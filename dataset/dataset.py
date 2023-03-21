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
                 removeDuplWhitespaces=False,
                 respect_subset_labels=None,
                 include_subsets=False):
        """
            path (str)         : Either path to file or path to directory containing caselaw, summary folders
            texts (list)       : Python list of strings that contain each document's text.
            Summaries (list)   : Python list that contains the label for each sequence (each label must be an integer)
            limit (int)        : If not None, construct dataset containing the first "limit" number of text-summary pairs.
            include_tags (bool): If True, then tags are included in each dataset's item.
            include_urls (bool): If True, then urls are included in each dataset's item
            respect_subset_labels( int | None): If not None then create dataset by keeping only the rows where column subset = respect_subset_labels

        """
        self.include_tags = include_tags
        self.include_urls = include_urls
        self.include_subsets = include_subsets

        assert(isinstance(path, str))
        # check if path points to single file or folder containing multiple files
        if os.path.isfile(path):
            df = read_pd_from_file(path, limit=limit)
            df=df.dropna(axis=0,subset=['text', 'summary'])
            if putResultsAtStart:
                df['text'] = df['text'].map(self.reorder_results)
            if removeDuplicates:
                if 'case_category' in df.columns:
                    df.drop(df.loc[df['case_category'] == 'Ερημοδικία_αναιρεσείοντος'].index, inplace=True)
                    df.drop(df.loc[df['case_category'] == 'Ερημοδικία αναιρεσείοντος'].index, inplace=True)
                else:
                    print("Creating dataset without removing duplicates due to non existent case_category column")
            if removeDuplWhitespaces:
                df['text'] = df['text'].map(self.remove_dupl_whitespace)
            if self.include_tags:
                try:
                    self.case_tags = df['case_category'].tolist()
                except:
                    self.case_tags = df['case_tags'].tolist()
            if self.include_urls:
                self.case_url = df['url'].tolist()
            if self.include_subsets:
                self.subsets = df['subset'].astype(int).tolist()
                print("!!!!!!!!!\n",self.subsets)
            if respect_subset_labels is not None:
                print("DSDSDS", df.columns)
                try:
                    df = df[df["subset"] == respect_subset_labels]
                except:
                    raise ValueError('No "subset" column found in the file')

            self.texts = df['text'].tolist()
            self.summaries = df['summary'].tolist()
            self.df=df

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
        metadata_tupl = ()
        if self.include_tags:
            metadata_tupl += (self.case_tags[item],)
        if self.include_urls:
            metadata_tupl += (self.case_url[item],)
        if self.include_subsets:
            metadata_tupl += (self.subsets[item],)
        return self.texts[item], self.summaries[item], metadata_tupl

    def __len__(self):
        return len(self.texts)


    def get_df(self):
        return self.df

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
