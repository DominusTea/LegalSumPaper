'''
Runs a summarizer on a dataset and stores produced summaries
Usage: python3 summarize.py [-o path/to/outputs/dir] path/to/dataset.csv
'''

import warnings
import argparse
import os
import json
import pickle
import tqdm

from dataset.dataset import CaseLawDataset
from dataset.dataloader import CustomDataloader, CustomDataloaderWithTags

from utils.dataset import dataset_splitter

from models.graph_based.TextRank.textrank import TextRank_Summarizer
from models.graph_based.LexRank.lexrank import LexRank_Summarizer
from models.simple.RandomExtractor import RandomExtractor_Summarizer

from utils.output import output_summaries

import pandas as pd

def summarizer_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file',
        type=str,
        help="Path to dataset file.")
    parser.add_argument(
        '-m',
        '--model',
        default='lexrank',
        type=str,
        nargs='?',
        choices=['textrank', 'lexrank', 'random_sentence', 'biased_lexrank'],
        help="Model to be used for summarization."
    )
    parser.add_argument(
        '--saved_model',
        default=None,
        type=str,
        nargs='?',
        help="Path to saved model file. Currently applicable for lexrank/biased_lexrank"
    )
    parser.add_argument(
        '-o',
        '--output',
        default='output_summaries/',
        nargs='?',
        type=str,
        help = "Path to output the summaries"
    )
    parser.add_argument(
        '--output_filetype',
        default='json',
        type=str,
        choices=['json','txt'],
        help = "Filetype to store the output (.json|.txt)"
    )
    parser.add_argument(
        '-l',
        '--limit',
        default=None,
        type=int,
        help="Limit of summaries to produce"
    )
    parser.add_argument(
        '--bias_word',
        default=None,
        type=str,
        help="Word to bias the graph-based centrality algorithms. If True, use case tags as a bias word."
    )
    parser.add_argument(
        '--limit_summary_length',
        default=None,
        type=float,
        help="Force generated summaries to have up to --limit_summary_length*len(referene summary) number of tokens"
    )
    parser.add_argument(
        '--split_dataset',
        default=None,#"0.7_0.15_0.15",
        type=str,
        help="If flag is not used, then the whole dataset is summarized. Else, split percentages for train/dev/test | train/test must be provided. In both cases only test subset is summarized."
    )
    parser.add_argument(
        '--include_urls',
        action='store_true',
        default=False,
        help="Whether to include urls in the summary files. Default False."
    )
    parser.add_argument(
        '--include_tags',
        action='store_true',
        default=False,
        help="Whether to include tags in the summary files. Default False."
    )
    parser.add_argument(
        '--include_subsets',
        action='store_true',
        default=False,
        help="Whether to include subset id in the summary files. Default False."
    )
    parser.add_argument(
        '--put_results_at_start',
        action='store_true',
        default=False,
        help="Whether to reorder the judicial decision to be at the start of the caselaw. This is used when the summarizr has a fixed input length (e.g in BERT)"
    )
    parser.add_argument(
        '--remove_duplicates',
        action='store_true',
        default=False,
        help="Whether to remove duplicate dataset entries."
    )

    parser.add_argument(
        '--lexrank_similarity',
        default='idf_mod_cosine',
        type=str,
        choices=['idf_mod_cosine', 'word2vec_sim', 'common_words', 'bert_sim'],
        help="Which similarity function will be used by the lexrank algorithm"
    )
    parser.add_argument(
        '--subset',
        default=None,
        type=int,
        choices=[0,1,2],
        help="Which subset of the dataset to summarize. Works when input is a csv/json file. Default: summarize all subsets"
    )
    parser.add_argument(
        '--single_output_file',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--limit_on_text',
        action='store_true',
        default=False,
        help="True when limit_summary_length applies to text not reference summary."
    )


    return parser



if __name__=="__main__":

    parser = summarizer_argparser()
    args = parser.parse_args()
    print(args)
    # make output directory if it doesnt exist and open it
    if os.path.exists(args.output):
        warnings.warn(f"Output directed to already existing directory: {args.output}")


    else:
        print("Creating output directory", args.output)
        try:
            os.mkdir(args.output)
        except OSError as err:
            print(err)
    # add potentialy missing ("/" in Linux "\" in windows) os independent file seperator
    if args.output[-1] != os.path.sep:
        args.output = args.output + os.path.sep

    # raise ValueError(f"given path: {args.output} is not a directory")


    # if args.bias_word is not None:
    #     # construct dataset
    #     caselaw_dataset = CaseLawDataset(path=args.file, limit=args.limit, include_tags=True)
    #     caselaw_dataset = caselaw_dataset if args.split_dataset is None else dataset_splitter(caselaw_dataset, args.split_dataset)[1]
    #     # construct dataloader
    #     caselaw_dataloader = CustomDataloaderWithTags(caselaw_dataset, batch_size=1)
    # else:
        # construct dataset
    caselaw_dataset = CaseLawDataset(path=args.file,
                                     limit=args.limit,
                                     include_tags=args.include_tags,
                                     include_urls=args.include_urls,
                                     putResultsAtStart=args.put_results_at_start,
                                     removeDuplicates=args.remove_duplicates,
                                     respect_subset_labels=args.subset,
                                     include_subsets=args.include_subsets)
    # caselaw_dataset = caselaw_dataset if args.split_dataset is None else dataset_splitter(caselaw_dataset, args.split_dataset)[1]
    caselaw_dataset = caselaw_dataset if args.split_dataset is None else dataset_splitter(caselaw_dataset, args.split_dataset)[1]
    print("dataset contains: ", len(caselaw_dataset), " samples.")
    caselaw_dataloader = CustomDataloaderWithTags(caselaw_dataset, batch_size=1)

    # ~~~~~~~~~~~~~TEXTRANK~~~~~~~~~~~
    if args.model == "textrank":
        # get summarizer model
        model = TextRank_Summarizer()
        # iterate over dataloader
        for index, data in enumerate(caselaw_dataloader, 1):
            print(index)
            text, summary, metadata = data
            generated_summary = model.summarize(text[0], limit_sentences=4)
            # output to file
            output_summaries(directory=f"{args.output}{index}.txt",
                             filetype=args.output_filetype,
                             summary= generated_summary,
                             metadata=metadata)

            if args.limit is not None and index == args.limit:
                break;

    if args.model == "lexrank":
        if args.saved_model is None:
            # get documents to produce TF-IDF representations
            print("getting all caselaw data")
            docs = caselaw_dataset.get_texts()
            # get summarizer model
            print("constructing model")
            # model = LexRank(documents=docs[:round(len(docs)/100)])
            model = LexRank_Summarizer(documents=docs[0:args.limit],
                                       similarity_metric='idf_mod_cosine')
        else:
            # just load pickled idf scores, idf default value LexRank_Summarizer object.
            # This skips the IDF representations construction step.
            print("loading serialized lexrank model")
            f = open(args.saved_model, 'rb')
            serialized_scores = pickle.load(f)
            idf_scores  = serialized_scores['idf_dict']
            idf_default_value = serialized_scores['idf_default_value']
            f.close()
            model = LexRank_Summarizer(similarity_metric=args.lexrank_similarity)
            model.load_idf_scores(idf_scores, idf_default_value)

        print("finished model construction")
        if args.single_output_file:
            output_cols=['text','summary'] + (['case_tags'] if args.include_tags else []) + (['url'] if args.include_urls else []) + (['subset'] if args.include_subsets else [])
            df = pd.DataFrame(columns=output_cols)
            for index, data in enumerate(tqdm.tqdm(caselaw_dataloader), 1):
                text, summary, metadata = data
                print(index)
                generated_summary = model.summarize(text[0],
                                                    limit_sentences=0.5,
                                                    ref_summary=summary[0] if args.limit_summary_length is not None else None,
                                                    limit_tokens=args.limit_summary_length,
                                                    threshold=None,
                                                    bias_word=None,#'έγκλημα'
                                                    limit_on_text=args.limit_on_text
                                                    )
                tags_data = metadata[0] if args.include_tags else []
                url_data = metadata[1] if args.include_urls else []
                subset_data = metadata[2] if args.include_subsets else []

                print(output_cols)

                # data_for_insertion = [generated_summary, summary[0]] + [tags_data[0]] +[url_data[0]] + [subset_data[0].item()]
                data_for_insertion = [generated_summary, summary[0]]
                if args.include_tags:
                    data_for_insertion += [tags_data[0]]
                if args.include_urls:
                    data_for_insertion += [url_data[0]]
                if args.include_subsets:
                    data_for_insertion += [subset_data[0]]

                print(metadata)
                df.loc[index] = data_for_insertion
                # df = pd.concat([df, pd.DataFrame([data_for_insertion], columns=df.columns)],ignore_index=True)
                if args.limit is not None and index == args.limit:
                    break;

            df.to_csv(args.output+".csv", index=False)
        else:
            print("emiting output to multiple files")
            for index, data in enumerate(tqdm.tqdm(caselaw_dataloader), 1):
                text, summary, metadata = data
                print(index)
                generated_summary = model.summarize(text[0],
                                                    limit_sentences=8,
                                                    ref_summary=summary[0] if args.limit_summary_length is not None else None,
                                                    limit_tokens=args.limit_summary_length,
                                                    threshold=None,
                                                    bias_word=None#'έγκλημα',
                                                    )
                # output to file
                print("reference summary is: ", summary)
                output_summaries(directory=f"{args.output}{index}.txt",
                                 filetype=args.output_filetype,
                                 summary= generated_summary,
                                 metadata=metadata,
                                 reference_summary=summary
                                 )


                if args.limit is not None and index == args.limit:
                    break;
    if args.model == "biased_lexrank":
        if args.bias_word is None:
            raise RuntimeError("using biased lexrank without a bias word")

        if args.saved_model is None:
            # get documents to produce TF-IDF representations
            print("getting all caselaw data")
            docs = caselaw_dataset.get_texts()
            # get summarizer model
            print("constructing model")
            # model = LexRank(documents=docs[:round(len(docs)/100)])
            model = LexRank_Summarizer(documents=docs[0:args.limit],
                                       similarity_metric=args.lexrank_similarity)
        else:
            # just load pickled idf scores, idf default value LexRank_Summarizer object.
            # This skips the IDF representations construction step.
            print("loading serialized lexrank model")
            f = open(args.saved_model, 'rb')
            serialized_scores = pickle.load(f)
            idf_scores  = serialized_scores['idf_dict']
            idf_default_value = serialized_scores['idf_default_value']
            f.close()
            model = LexRank_Summarizer(similarity_metric=args.lexrank_similarity)
            model.load_idf_scores(idf_scores, idf_default_value)

        print("finished model construction")

        for index, data in enumerate(caselaw_dataloader, 1):
            text, summary, metadata = data
            # tag = metadata[0]
            tag = metadata[0][0]
            print("metadata is ", metadata)

            if args.bias_word == "True":
                bias_word=tag.split(',')
                print("here bias word is ", bias_word)
                # if isinstance(tag, str):
                #     bias_word = tag if len(tag.split('_')) == 1 else tag.split('_')
                # elif isinstance(tag, list):
                #     bias_word = tag[0] if len(tag[0].split('_')) == 1 else tag[0].split('_')
                # elif isinstance(tag, tuple):
                #     bias_word = tag[0] if len(tag[0].split('_')) == 1 else tag[0].split('_')
                # else:
                #     raise ValueError("Tag must be [str|list|tuple]")
            else:
                bias_word=args.bias_word
            print(index, tag, bias_word)
            generated_summary = model.summarize(text[0],
                                                limit_sentences=9,
                                                ref_summary=summary[0] if args.limit_summary_length is not None else None,
                                                threshold=None,
                                                limit_tokens=args.limit_summary_length,
                                                bias_word=bias_word#'έγκλημα',
                                                )
            # output to file
            output_summaries(directory=f"{args.output}{index}",
                             filetype=args.output_filetype,
                             summary=generated_summary,
                             metadata=metadata,
                             reference_summary=summary
                             )


            if args.limit is not None and index == args.limit:
                break;

    if args.model == "random_sentence":
        # get summarizer model
        model = RandomExtractor_Summarizer()
        # iterate over dataloader
        # print("!!!!!!!!!",caselaw_dataset[0], type(caselaw_dataset[0]))
        for index, data in enumerate(tqdm.tqdm(caselaw_dataloader), 1):
            # print(index)
            text, summary, metadata = data
            generated_summary = model.summarize(text[0],
                                                limit_sentences=6,
                                                ref_summary=summary[0] if args.limit_summary_length is not None else None,
                                                limit_tokens=args.limit_summary_length)
            # output to file
            output_summaries(directory=f"{args.output}{index}.txt",
                             filetype=args.output_filetype,
                             summary= generated_summary,
                             reference_summary=summary,
                             metadata=metadata)

            if args.limit is not None and index == args.limit:
                break;
