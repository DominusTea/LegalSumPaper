'''
preprocess raw files and outputs them in a format compatible with
the summarization models
'''

import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from utils.text import AreiosPagosClearingFunc, STEClearingFunc, removeSTE_substring,remove_dupl_whitespace

def read_pd_from_file(path,usecols=None, limit=None):
    '''
    returns dataframe from given csv or json file.
    '''
    filetype = path.split(".")[-1]
    if filetype == "csv":
        reader_func = pd.read_csv
    elif filetype == "json":
        reader_func =pd.read_json
    else:
        raise ValueError("file must be of type: [csv|json]. Given file is of type ", filetype)

    if usecols is not None:
        df = reader_func(path, usecols=usecols,nrows=limit)
    else:
        df = reader_func(path,nrows=limit)

    return df

def preproc_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'raw_data',
        type=str,
        help="Path to raw data file. \n \
        If given file is a directory then preproc data from all .csv and .json files (CURRENTLY NOT IMPLEMENTED)"
        )
    parser.add_argument(
        '-d',
        '--dataset_type',
        type=str,
        choices=["STE", "AreiosPagos"],
        default="AreiosPagos",
        help="Type of dataset of which we preprocess the raw data.\
         [STE|AreiosPagos]"
         )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help = "Path to output the preprocessed data."
    )
    parser.add_argument(
        '--output_filetype',
        type=str,
        choices=["json", "csv"],
        default='json',
        help = "Type of file for the preprocessing output. Supported: [.csv|.json]."
    )
    parser.add_argument(
        '-m',
        '--model_type',
        default='TextRank',
        nargs='?',
        type=str,
        help = "Preprocess data to be compatible with given model type"
    )
    parser.add_argument(
        '-l',
        '--limit',
        default=None,
        type=int,
        help="Upper limit of number of caselaw to preprocess"
    )
    parser.add_argument(
        '--single_file',
        action='store_true',
        default=False,
        help="Store output summary, caselaw pairs into a single file"
    )
    parser.add_argument(
        '--include_urls',
        action='store_true',
        default=False,
        help="Whether to include urls in the preprocessed output file"
    )
    parser.add_argument(
        '--include_tags',
        action='store_true',
        default=False,
        help="Whether to include urls in the preprocessed output file"
    )

    parser.add_argument(
        '--remove_newlines',
        actions='store_true',
        default=False,
        help="Whether all newline characters are to be removed from the text"
    )

    return parser

def get_desired_columns(data_type="AreiosPagos", model_type="TextRank"):
    '''
    Returns which columns to keep according to dataset type and summarization model
    '''
    if data_type=="AreiosPagos":
        default_cols = ['summary', 'text', 'case_category']
        if args.include_urls:
            default_cols += ['url']
        if args.include_tags:
            default_cols += ['case_tags']
        print(default_cols)
    if data_type=="STE":
        default_cols = ['case_num', 'date', 'metadata_url', 'summary', 'text', ]
        if args.include_urls:
            default_cols += ['url']

    return default_cols



def remove_unnecessary_text(df, data_type, model_type):
    '''
    Removes unnecessary text from dataframe according to data_type and model_type
    '''
    print("given data_type is", data_type)
    if data_type == "AreiosPagos":
        print("applying AreiosPagos unnecessary text clearning")
        df['text'] = df['text'].transform(AreiosPagosClearingFunc())

    if data_type =="STE":
        tqdm.pandas()
        print("applying STE unnecessary text cleaning")
        df['text'] = df['text'].str.replace(removeSTE_substring(), " ",regex=True)
        print("removed unecessary string")
        df['text'] = df['text'].progress_apply(STEClearingFunc())
        print("removed unecessary parts")

    df['text'] = df['text'].progress_apply(remove_dupl_whitespace)
    print("removed duplicate whitespaces")
    return df

def preprocess_df(df, data_type, model_type, remove_newlines=False):
    '''
    Preprocess each column in the df, according to dataset and summarization model type
    '''
    if remove_newlines:
        print("removing all newlines from output file")
        df['text']=df['text'].str.replace('\n','')

    if data_type == "AreiosPagos":
        df = df.dropna(subset=['text','summary'])
        df = remove_unnecessary_text(df,
                                     data_type=data_type,
                                     model_type=model_type)
    else:
        df=df.dropna(subset=['text'])
        df = df[df['text'] != 'Document is empty!']
        print("df cols are ",df.columns, len(df))
        # dropping older cases since there are major transcription errors
        print("llererernenr is ", len(df.iloc[250]))
        if len(df) <= 250:
            start_idx=0
            print("Warning: Due to --limit option / insufficient dataset size, the older cases will not be dropped from the preprocessed output")
        else:
            start_idx=250
            print("dropping older cases since there are major transcription errors")
        df = df.sort_values(by=['date'],key=lambda x: x.str.split('/',expand=True)[2]).iloc[start_idx:]
        df = remove_unnecessary_text(df,
                                     data_type=data_type,
                                     model_type=model_type)

    return df



if __name__=="__main__":
    parser = preproc_argparser()
    args = parser.parse_args()

    path = args.raw_data
    print(args.output, type(args.output))
    if not args.single_file:
        # check if output directory exists
        if not(os.path.exists(args.output)):
            print(f"Directory {args.output} not found. Making it...")
            os.mkdir(args.output)
        else:
            #path exists, check if it is a directory
            if not(os.path.isdir(args.output)):
                raise ValueError("output argument is not a directory")
        # check if subdirectories exist
        if not(os.path.exists(args.output+"/caselaw")):
            print(f"Sub-directory {args.output}/caselaw not found. Making it...")
            os.mkdir(args.output+"/caselaw")
        if not(os.path.exists(args.output+"/summary")):
            print(f"Sub-directory {args.output}/summary not found. Making it...")
            os.mkdir(args.output+"/summary")

    # load the data
    if os.path.isfile(path):
        # get desired columns
        desired_columns = get_desired_columns(data_type=args.dataset_type, model_type=args.model_type)
        # read dataframe from file
        df = read_pd_from_file(path,usecols=desired_columns, limit=args.limit)

    elif os.path.isdir(path):
        raise NotImplementedError("Preprocessing data from directory is not currently implemented")
    else:
        raise FileNotFoundError("Input file not found (?)")

    # preprocess each column
    preprocessed_df = preprocess_df(df=df,
                                    data_type = args.dataset_type,
                                    model_type =  args.model_type,
                                    remove_newlines = args.remove_newlines
                                    )
    if args.single_file:
        # emit to a single file
        f = open(args.output, 'w+', encoding='utf-8')
        print("outputing to single file: ", args.single_file)
        if args.output_filetype == 'csv':
            preprocessed_df.to_csv(f, index=False,)
        elif args.output_filetype == 'json':
            preprocessed_df.to_json(f, ensure_ascii=False)
        else:
            raise ValueError("Not recognised filetype")
    else:
        # emit df into multiple files indexed by the row index. Caselaw folder contains only the text, while summary folder contains the summary and the rest of the metadata
        for index, row in preprocessed_df.iterrows():
            with open(f"{args.output}/caselaw/{index+1}.json", "w+", encoding='utf-8') as f:
                # get caselaw file series
                caselaw_file_series = row['text']
                # get corresponding json from string
                caselaw_file_json = json.dumps({'text': caselaw_file_series},
                                               ensure_ascii=False)
                # emit caselaw text to file
                f.write(caselaw_file_json)
                # caselaw_file_json.dump(caselaw_file_json,
                #                        f,
                #                        ensure_ascii = False)

            with open(f"{args.output}/summary/{index+1}.json", "w+", encoding='utf-8') as f:
                # get summary file series
                summary_file_series = row.drop(labels=['text'])
                # emite summary (and metadata) to file
                summary_file_series.to_json(f, force_ascii=False)

                # # get all column names
                # cols = row.index.to_list()
                # print(cols)
                # # find all except 'text', to store in the summary folder
                # summary_file_cols = cols.copy()
                # summary_file_cols = summary_file_cols.remove('text')
                # summary_file_series = row([summary_file_cols])

            if args.limit is not None and index == args.limit-1.:
                break
