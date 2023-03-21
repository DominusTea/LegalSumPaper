# -*- coding: utf-8 -*-


import argparse

def preproc_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'file_path',
        type=str,
        help="Path to dataset",
        )
    parser.add_argument(
        'mode',
        type=str,
        choices=["train",'eval', 'both'],
        default='both'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        default="model_is_saved/",
        help="where to save the model"
    )
    parser.add_argument(
        '--model_load_path',
        type=str,
        default="model_is_saved/",
        help="where to load the model from. Only for eval!"
    )
    parser.add_argument(
        '--eval_output_file_path',
        type=str,
        default="eval_results/",
        help="Where to store inference results"
    )
    parser.add_argument(
        '--reorder',
        action='store_true',
        default=False,
        help="Whether to reorder caselaw data"
    )
    parser.add_argument(
        '--remove_info',
        action='store_true',
        default=False,
        help="Whether to remove unecessary text"
    )
    parser.add_argument(
        '--put_categs_at_start',
        action='store_true',
        default=False,
        help="Whether to include category data"
    )




    return parser

parser=preproc_argparser()
args=parser.parse_args()

file_path=args.file_path #'AreiosPagosFullWithCategsSubsetsReduced.csv'

from transformers import BertConfig, BertGenerationDecoder
from transformers import AutoModel, AutoTokenizer

from transformers import EncoderDecoderModel

from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.models.bart.modeling_bart import BartDecoder
import torch

custom_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("custom device is", custom_device)

"""## Batch Processing for BERT model compatibility"""

def process_data_to_model_inputs(text_batch, summary_batch,
                                 encoder_max_length=512,
                                 decoder_max_length=128,
                                 tokenizer=None,
                                 include_decoder_inputs=False,
                                 ignore_pad_token=True,
                                 device=custom_device):
  # tokenize the inputs and labels
  # inputs = tokenizer(text_batch, padding="max_length", truncation=True, max_length=encoder_max_length)
  # outputs = tokenizer(summary_batch, padding="max_length", truncation=True, max_length=decoder_max_length)

  # print("!!!!!!!!!")
  # print('text:', text_batch)
  # print('summary:', summary_batch)
  # print("type of text_batch, text_batch[0]:", type(text_batch), type(text_batch[0]))
  # print("type of summary_batch, summary_batch[0]:",type(summary_batch), type(summary_batch[0]))
  # print('~~~~~~~~~~~~~~~S')
  inputs = tokenizer(text_batch)
  outputs = tokenizer(summary_batch)

  # print("ref sum[0]:",summary_batch[0])
  # print("ref sum[1]:",summary_batch[1])


  batch = dict()
  batch["input_ids"] = inputs.input_ids.to(device)
  batch["attention_mask"] = inputs.attention_mask.to(device)
  if include_decoder_inputs:
    batch["decoder_input_ids"] = outputs.input_ids.to(device)
    batch["decoder_attention_mask"] = outputs.attention_mask.to(device)
  batch["labels"] = outputs.input_ids

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
  # We have to make sure that the PAD token is ignored
  if ignore_pad_token:
    batch["labels"] = torch.LongTensor([[-100 if token == 0 else token for token in labels] for labels in batch["labels"]])
  batch["labels"] = batch["labels"].to(device)


  return batch


def custom_collator(tokenizer, batch, return_dict=True, device='cuda:0'):
    # print("item[0]:",batch[0][0])
    # print("item[1]:",batch[0][1])
    # encode data and target to huggingface form ([CLS], huggingface tokenizer)
    if return_dict:
        text_batch = [item[0] for item in batch]
        summary_batch = [item[1] for item in batch]
        return process_data_to_model_inputs(text_batch, summary_batch, tokenizer=tokenizer, device=device)


    data = [tokenizer(item[0])['input_ids'] for item in batch]
    target = [tokenizer(item[1])['input_ids'] for item in batch]
    # print("target after tokenization ", target)
    # target = torch.LongTensor(target)
    return [data, target]




def harmonize_configs(model, tokenizer):
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size
    return model

"""## Dataset Class Definition"""

def remove_text_relative_to_substring(fulltext, substring, remove_text='before'):
  '''
  Removes text before/after the occurance of a substring
  '''
  split_text = fulltext.split(substring)
  N_occurs = len(split_text)
  # print(N_occurs)
  if remove_text=='before':
      if N_occurs == 1:
          # print(split_text[0][0:9000])
          # print("~~~``")
          return split_text[0]
      elif N_occurs== 2:
          return split_text[1].strip()
      else:
          return ("".join(split_text[1:])).strip()
  elif remove_text =='after':
      if N_occurs == 1:
          return split_text[0]
      elif N_occurs == 2:
          return split_text[0].strip()
      else:
          return ("".join(split_text[0:N_occurs-1])).strip()
  else:
      raise ValueError("remove_text must be one of ['before'|'after']")


def remove_text_relative_to_strings(fulltext, strings, remove_text='before'):
    '''
    Like remove_text_relative_to_substring but strings can be a list of potential (substngs, remove_text)
    '''
    if isinstance(strings, str):
        return remove_text_relative_to_substring(fulltext, strings, remove_text)
    else:
        processed = fulltext
        for substring, remove_text_pos in strings:
            processed = remove_text_relative_to_substring(processed,
                                                          substring=substring,
                                                          remove_text=remove_text_pos)
        return processed


def remove_AreiosPagos_start():
    starting_options = [("ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ", 'before'),
                    ("ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟΝ ΝΟΜΟ", 'before'),
                    ("ΣΚΕΦΤΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ", 'before'),
                    ("Σκέφθηκε σύμφωνα με το νόμο", 'before'),
                    ("Σκέφθηκε σύμφωνα με στείλουν νόμο", 'before'),
                    ("Σ Κ Ε Φ Θ Η Κ Ε  Σ Υ Μ Φ Ω Ν Α  Μ Ε  Τ Ο  Ν Ο Μ Ο", "before")
                    ]

    return starting_options


def removeAreiosPagos_end():
    ending_options= [("Κρίθηκε και αποφασίσθηκε", 'after'),
                ("ΚΡΙΘΗΚΕ, αποφασίσθηκε", 'after'),
                ("ΚΡΙΘΗΚΕ και αποφασίσθηκε", 'after'),
                ]
    return ending_options

def AreiosPagosClearingFunc():
    substring_options = remove_AreiosPagos_start() + removeAreiosPagos_end()
    return lambda x : remove_text_relative_to_strings(fulltext=x,
                                                      strings=substring_options,
                                                      )

'''
Contains code for pytorch Dataset creation

'''
import glob
from torch.utils.data import Dataset, random_split
import torch

import os
from transformers import AutoTokenizer

import pandas as pd
import re


import csv


def read_pd_from_file(path, dataisDict=False, limit=None):
    '''
    returns dataframe from given csv or json file.
    dataisDict (bool): True when reading from a file that contains a single row
    '''
    filetype = path.split(".")[-1]
    if filetype == "csv":
        df = pd.read_csv(path,nrows=limit)
    elif filetype == "json":
        if dataisDict:
            df = pd.read_json(path, typ='dictionary', nrows=limit)
        else:
            df = pd.read_json(path, nrows=limit)
    else:
        raise ValueError("file must be of type: [csv|json]. Given file is of type ", filetype)
    # print(df)
    return df


def read_pd_from_file1(path, dataisDict=False, limit=None):
    '''
    returns dataframe from given csv or json file.
    dataisDict (bool): True when reading from a file that contains a single row
    '''
    try:
      filetype = path.split(".")[-1]
      if filetype == "csv":
          df = pd.read_csv(path,nrows=limit)
      elif filetype == "json":
          if dataisDict:
              df = pd.read_json(path, typ='dictionary', nrows=limit)
          else:
              df = pd.read_json(path, nrows=limit)
      else:
          raise ValueError("file must be of type: [csv|json]. Given file is of type ", filetype)
      # print(df)
      return df
    except:
      return pd.DataFrame(data={'text':"",'summary':""})


def reorder_results(s, query="ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ"):
  '''
  searches for a certain substring and if found places it and each text that follows at the start of the string
  '''
  split_text = s.split(query)
  if len(split_text) == 1:
    # the query string is not in the
    return s
  else:
    # construct reordered string
    # ...find the result string
    res = query + " " + split_text[-1]
    #...find the rest
    rest = split_text[0] if len(split_text) == 1 else " ".join(split_text[:-1])
    return res + " " + rest

def isSpecialToken(input):
    '''
    returns True if input string is a (transformers) special token
    '''
    return input in ['[SEP]', '[MASK]', '[CLS]', '[UNK]', '[PAD]']

def truncate_text(text, maxLen, tokenizer, keepSpecialTokens=False):
    '''
    truncates text to maxLen number of tokens.
    Text is truncated according to tokenization (using the greek bert AutoTokenizer)

    maxLen:             int
    tokenizer:          transformers AutoTokenizer
    keepSpecialTokens:  boolean
    '''
    text_ids = tokenizer.encode(text)
    text_tokens_lst = tokenizer.covert_ids_to_tokens(text_ids)

    if not keepSpecialTokens:
        text_tokens_lst = filter(lambda token: not isSpecialToken(token),
                                 text_tokens_lst)
    truncated = text_tokens_lst[0:maxLen]
    return(" ".join(truncated))

def get_file_index(filename):
    '''
    Returns file index from file name
    Input (str): A filename string (like dataset/preprocessed/AreiosPagos/caselaw/417.json)
    Output (int): An integer (like 417 in the example used above)
    '''
    # omit filetype from string
    s = filename.split('.')[0]
    # omit filepath to keep only filename. This should be an integers
    if os.sep in s:
        s = s.split(os.sep)[-1]
    return int(s)





class CaseLawDataset(Dataset):
    '''
    Code for torch dataset object.
    Doesn't tokenize or do any preprocessing/padding.
    if maxLen is defined, each text is truncated to maxLen length (using GreekBert tokenizer)
    if keepSpecialTokens is True, text contains special tokens [SEP], [MASK], [CLS],[UNK], [PAD]
    '''
    def __init__(self, path,
                 maxLen=None,
                 keepSpecialTokens=False,
                 limit=None,
                 include_tags=False,
                 include_urls=False,
                 putResultsAtStart=False,
                 removeDuplicates=True,
                 removeStopWords=False,
                 respect_subset_labels=None,
                 putCategsAtStart=False,
                 removeDuplWhitespaces=False,
                 removeStartAndEndInformation=False):
        """
            path (str)         : Either path to file or path to directory containing caselaw, summary folders
            texts (list)       : Python list of strings that contain each document's text.
            Summaries (list)   : Python list that contains the label for each sequence (each label must be an integer)
            limit (int)        : If not None, construct dataset containing the first "limit" number of text-summary pairs.
            include_tags (bool): If True, then tags are included in each dataset's item.
            include_urls (bool): If True, then urls are included in each dataset's item
            respect_subset_labels( int): If not None then create dataset by keeping only the rows where column subset = respect_subset_labels
            removeStartAndEndInformation (bool): If true, removes start and end of the court main text that correspond to general information.
        """
        self.include_tags = include_tags
        self.include_urls = include_urls
        if removeStopWords:
          self.nlp=spacy.load("el_core_news_sm")
          self.stopwords = set(self.nlp.Defaults.stop_words)

        assert(isinstance(path, str))
        # check if path points to single file or folder containing multiple files
        if os.path.isfile(path):
            if False:
              df=pd.DataFrame(columns=['text','summary','case_tags','url','subset'])
              with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for i,row in enumerate(reader):
                  # print("readig subset = ", row['subset'])
                  try:
                    sb=int(row['subset'])
                    df.loc[len(df)] = [row['text'],row['summary'],row['case_tags'], row['url'],sb]
                  except:
                    print("error subset ", row['subset'], " with row contents ", row)
                    sb=0
                  df.loc[len(df)] = [row['text'],row['summary'],row['case_tags'], row['url'],sb]

            else:
              df = read_pd_from_file(path, limit=limit)
            print("read df with length ", len(df), " and columns ", df.columns)
            df = df.dropna(axis=0, subset=['text','summary'],)
            print(" length after empty col removal is ", len(df))
            if respect_subset_labels is not None:
              df = df[df['subset'] == respect_subset_labels]
              print(" length after selecting subset = ",respect_subset_labels, " is ", len(df))
            if removeStartAndEndInformation:
              df['text']=df['text'].map(AreiosPagosClearingFunc())
            if putResultsAtStart:
                df['text'] = df['text'].map(self.reorder_results)
            if removeDuplWhitespaces:
                df['text'] = df['text'].map(self.remove_dupl_whitespace)
            if removeDuplicates:
                print("length before dropping Ερημοδικία....", len(df))
                df.drop(df.loc[df['case_category'] == 'Ερημοδικία_αναιρεσείοντος'].index, inplace=True)
                df.drop(df.loc[df['case_category'] == 'Ερημοδικία αναιρεσείοντος'].index, inplace=True)
                print("length after dropping Ερημοδικία....", len(df))
            if putCategsAtStart:
              df['text'] = df['case_tags'].map(lambda x : x.replace("_"," ")) + " [SEP] " + df['text']
            if removeStopWords:
              df['text'] = df['text'].map(self.remove_stopwords)


            if self.include_tags:
                self.case_tags = df['case_category'].tolist()
            if self.include_urls:
                self.case_url = df['url'].tolist()

            self.texts = df['text'].tolist()
            self.summaries = df['summary'].tolist()

        elif os.path.isdir(path):
            all_text_files = glob.glob(os.path.join(path, "caselaw/*.csv")) + glob.glob(os.path.join(path, "caselaw/*.json"))
            all_summary_files = glob.glob(os.path.join(path, "summary/*.csv")) + glob.glob(os.path.join(path, "summary/*.json"))

            # sort by number
            all_text_files = sorted(all_text_files, key = get_file_index )
            all_summary_files = sorted(all_summary_files, key = get_file_index )

            # Apply dataset size constraint if limit is not None
            if limit is not None:
                all_text_files=all_text_files[0:limit]
                all_summary_files=all_summary_files[0:limit]

            # get reading function
            read_fun = lambda x: read_pd_from_file(x, True)
            df_caselaws = pd.concat(map(read_fun, all_text_files))
            df_summaries = pd.concat(map(read_fun, all_summary_files))
            if putResultsAtStart:
                df_caselaws['text'] = df_caselaws['text'].map(reorder_results)
            # df = pd.concat([df_caselaws, df_summaries], axis=1)
            if removeDuplicates:
                raise Warning("Currently duplicate removal not implemented on folder format")
            self.texts = df_caselaws['text'].tolist()
            self.summaries = df_summaries['summary'].tolist()
            if self.include_tags:
                self.case_tags = df_summaries['case_category'].tolist()
            if self.include_urls:
                self.case_url = df_summaries['url'].tolist()

        # length truncation
        if maxLen is not None:
            self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
            self.maxLen = (maxLen-1) if keepSpecialTokens else maxLen
            # truncate each text
            df['text'] = df['text'].map(
                    lambda text:
                truncate_text(text, maxLen, self.tokenizer, keepSpecialTokens))



    def __getitem__(self, item):
        # print("type1", type((self.texts[item], self.summaries[item])))
        # result_tupl += self.case_url[item],
        #
        # print("type2", type(result_tupl))
        # return self.texts[item], self.summaries[item]
        return self.texts[item], self.summaries[item]
        metadata_tupl = ()
        if self.include_tags:
            metadata_tupl += (self.case_tags[item],)
        if self.include_urls:
            metadata_tupl += (self.case_url[item],)
        return self.texts[item], self.summaries[item], metadata_tupl

    def __len__(self):
        return len(self.texts)

    def get_data(self):
        '''
        Returns the model's texts and summaries into a list form
        Output: [list of text strings], [list of summary strings]
        '''
        result = self.texts, self.summaries
        if self.include_tags:
            result += self.case_tags,
        if self.include_urls:
            result += self.case_url
        return result

    def get_texts(self):
        return self.texts

    def set_texts(self, input_texts):
        self.texts = input_texts

    def remove_stopwords(self,t):
        # try:
        #   doc = self.nlp(t)
        #   filtered = [a.text for a in doc if not a.is_stop]
        #   return " ".join(filtered)
        # except:
        #   return t
        try:
          doc = t.split(" ")
          filtered = [word for word in doc if not word in self.stopwords]
          return " ".join(filtered)
        except:
          return t



    def split_dataset(self, lengths, seed=42):
        '''
        Splits input dataset into 2 parts, each having number of elements given by the lengths input list
        '''
        # if len(lengths) == 2:
        #     train_set, test_set = random_split(self,
        #                                        lengths,
        #                                        generator=torch.Generator().manual_seed(seed))
        #     return train_set, test_set
        # elif len(lengths) == 3:
        #     print("here")
        #     train_set, dev_set, test_set = random_split(self,
        #                                                 lengths,
        #                                                 generator=torch.Generator().manual_seed(seed))
        #     return train_set, dev_set, test_set
        # else:
        return random_split(self,
                            lengths,
                            generator=torch.Generator().manual_seed(seed))




    @staticmethod
    def remove_dupl_whitespace(s):
      '''
      Removes duplicate whitespace every where in the string
      '''
      symbol_list=[" ", "\n", "\t"]
      for symbol in symbol_list:
          s = re.sub(symbol+'+', symbol, s)
      return s

    @staticmethod
    def reorder_results(s, query="ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ"):
        '''
        searches for a certain substring and if found places it and each text that follows at the start of the string
        '''
        split_text = s.split(query)
        if len(split_text) == 1:
            # the query string is not in the string
            return s
        else:
            # construct reordered string
            # ...find the result string
            res = query + " " + split_text[-1]
            #...find the rest
            rest = split_text[0] if len(split_text) == 1 else " ".join(split_text[:-1])
            return res + " " + rest


"""## BERT Model definition & harmonization"""

model_str = 'nlpaueb/bert-base-greek-uncased-v1'

tokenizer_greek = AutoTokenizer.from_pretrained(model_str)
custom_tokenizer= lambda x: tokenizer_greek(x,
                                            return_tensors='pt',
                                            padding='max_length',
                                            truncation=True,
                                            max_length=512)


# config_decoder = BertConfig(is_decoder=True)
# decoder=BertGenerationDecoder(config_decoder)

# model_str = "patrickvonplaten/bert2bert-cnn_dailymail-fp16" #



if args.mode in ["train", 'both']:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_str,
                                                                model_str
                                                                #"facebook/bart-base",
                                                                #BartDecoder.from_pretrained("facebook/bart-base")
                                                                )

    # model = EncoderDecoderModel.from_pretrained('drive/MyDrive/diplomatiki/legal_summarization/models/checkpoint-5200-full_dataset-correct_hyper')


    model = harmonize_configs(model, tokenizer_greek)
    model = model.to(custom_device)
# simple model for BERT training tasks
# model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

# encoder decoder model. Only encoder should be pretrained

"""## Model Training

"""

# removeDuplicates=True,
#                  removeStopWords=False,
#                  respect_subset_labels=None,
#                  putCategsAtStart=True,
#                  removeDuplWhitespaces=False

putResultsAtStart=args.reorder
removeDuplicates=True
removeStopWords=False
removeDuplWhitespaces=True
removeStartAndEndInformation=args.remove_info
putCategsAtStart=args.put_categs_at_start


"""## Dataset construction"""



# from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     sampler=ImbalancedDatasetSampler(train_dataset),
#     batch_size=args.batch_size,
#     **kwar)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
  '''
  Custom Hugginface Seq2Seq Trainer that overrides the random sampling dataloader with a sampler tailored to imbalanced datasets
  '''
  def __init__(self, *args, **kwargs):
    super(CustomSeq2SeqTrainer, self).__init__(*args, **kwargs)

  def get_train_dataloader(self):
    if self.train_dataset is None:
      raise ValueError("Trainer: training requires a train_dataset.")

    # train_sampler = self._get_train_sampler()
    print("constructing sampler")
    train_sampler = ImbalancedDatasetSampler(self.train_dataset)
    print("returning dataloader")
    return DataLoader(
        self.train_dataset,
        batch_size=self.args.train_batch_size,
        sampler=train_sampler,
        collate_fn=self.data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
        pin_memory=self.args.dataloader_pin_memory,
    )


# encoder decoder model. Only encoder should be pretrained



train_dataset = CaseLawDataset(file_path,
                         putResultsAtStart=putResultsAtStart,
                         removeDuplicates=removeDuplicates,
                         removeStopWords=removeStopWords,
                         removeDuplWhitespaces=removeDuplWhitespaces,
                         removeStartAndEndInformation=removeStartAndEndInformation,
                         putCategsAtStart=putCategsAtStart,
                         respect_subset_labels=0
                         )

val_dataset = CaseLawDataset(file_path,
                         putResultsAtStart=putResultsAtStart,
                         removeDuplicates=removeDuplicates,
                         removeStopWords=removeStopWords,
                         removeDuplWhitespaces=removeDuplWhitespaces,
                         removeStartAndEndInformation=removeStartAndEndInformation,
                         putCategsAtStart=putCategsAtStart,
                         respect_subset_labels=1
                         )
test_dataset = CaseLawDataset(file_path,
                         putResultsAtStart=putResultsAtStart,
                         removeDuplicates=removeDuplicates,
                         removeStopWords=removeStopWords,
                         removeDuplWhitespaces=removeDuplWhitespaces,
                         removeStartAndEndInformation=removeStartAndEndInformation,
                         putCategsAtStart=putCategsAtStart,
                         respect_subset_labels=2
                         )

print(len(train_dataset))

if args.mode in ['train','both']:
    training_args = Seq2SeqTrainingArguments(
                                      output_dir="here",
                                      # output_dir="drive/MyDrive/diplomatiki/legal_summarization/models/",
                                      do_train=True,
                                      do_eval=True,
                                      # gradient_accumulation_steps=2,
                                      num_train_epochs=3,
                                      logging_steps=70,
                                      eval_steps=900,
                                      evaluation_strategy='steps',
                                      learning_rate=4e-5,
                                      # log_level='debug',
                                      # debug="underflow_overflow",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      dataloader_pin_memory=False,
                                      save_total_limit = 1,
                                      save_steps=1800,
                                      # fp16=True,
                                      )
    trainer = Seq2SeqTrainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=lambda x: custom_collator(tokenizer=custom_tokenizer,
                                                              batch=x,
                                                              device=custom_device),

                      # optimizers=(custom_optimizer, custom_scheduler) #custom_scheduler,
                      )

# model = EncoderDecoderModel.from_pretrained('drive/MyDrive/diplomatiki/legal_summarization/models/ourmodel_7_10_results_at_start_startend_info_removed_lexranked', output_attentions=True, output_hidden_states=False).to(custom_device)
if args.mode in ['train','both']:
    try:
        trainer.train(
                      # resume_from_checkpoint=True
                      # resume_from_checkpoint='drive/MyDrive/diplomatiki/legal_summarization/models/checkpoint-5400'
                      )
    except:
        import IPython ; IPython.embed() ; exit(1)
# model.save_pretrained("ourmodel")
# model.save_pretrained("drive/MyDrive/diplomatiki/legal_summarization/models/default_model")

# model.save_pretrained("drive/MyDrive/diplomatiki/legal_summarization/models/ourmodel_8_10_results_at_start_startend_info_removed_categs_lexranked")
    try:
        model.save_pretrained(args.model_save_path)
    except:
        import IPython ; IPython.embed() ; exit(1)


i=1
import json
import time
if not os.path.exists(args.eval_output_file_path):
  os.makedirs(args.eval_output_file_path)
for data in test_dataset:
  # print(data[1])
  # time.sleep(1)
  if args.mode =='eval':
      model=EncoderDecoderModel.from_pretrained(args.model_load_path).to(custom_device)

  encoding = custom_tokenizer([data])
  output=model.generate(input_ids=encoding['input_ids'].to(custom_device),
                        #  decoder_input_ids=encoding['input_ids'].to('cuda:0'),
                        max_length=513,
                        min_length=8,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        length_penalty=1.3
                        )
  trainer_outputs = tokenizer_greek.batch_decode(output, skip_special_tokens=True)
  # ref_output = summary
  print("generated:",trainer_outputs[0])
  print("reference:",data[1])
  print(f"~~~~{i}~~~~~~")

  with open(f"{args.eval_output_file_path}/{i}.json", "w+", encoding='utf-8') as f:
    json.dump({"summary": trainer_outputs[0], "reference_summary": data[1]},
              f,
              ensure_ascii=False)
  i+=1
