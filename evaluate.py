import pyrouge # (https://github.com/bheinzerling/pyrouge) (A python wrapper over the original rouge script - NOT IMPLEMENTED)
import rouge # (https://github.com/Diego999/py-rouge) (A python reimplementation of the original rouge script)

from rouge_score import rouge_scorer #(https://pypi.org/project/rouge-score/)


from nltk import meteor, word_tokenize

import argparse
import os
import pandas as pd
import warnings
import glob
import json


from dataset.dataset import CaseLawDataset
from utils.dataset import dataset_splitter



import numpy as np


def get_file_index(filename):
    '''
    Returns file index from file name
    Input (str): A filename string (like dataset/preprocessed/AreiosPagos/caselaw/417.json)
    Output (int): An integer (like 417 in the example used above)
    '''
    # omit filetype from string
    s = ".".join(filename.split('.')[:-1])
    # omit filepath to keep only filename. This should be an integers
    if os.sep in s:
        s = s.split(os.sep)[-1]
    s=s.replace('.txt','')
    return int(s)


def read_str_from_file(path, sum_col_label='summary'):
    '''
    returns summary string from given csv or json file.
    Txt files must contain the summary in the first line
    Json files must contain the summary under the key 'summary'
    Input:
        path (str): Path to file containg the summary
    Output:
        Summary (str)
    '''
    filetype = path.split(".")[-1]
    if filetype == "txt":
        file = open(path, 'r')
        summary = file.read()
        file.close()
    elif filetype == "json":
        with open(path,'r',encoding='utf-8') as file:
            print(file)
            data = json.load(file)
            summary=data[sum_col_label]
            # print("path is ", path)
            # data = json.load(file)
            # for key,val in data.items():
            #     print("key: ", key)
            #     print("\nval: ", val)
            # summary = data[sum_col_label]
    else:
        raise ValueError("file must be of type: [csv|json]. Given file is of type ", filetype)
    return summary


def read_summaries_from_directory(dir,sum_col_label='summary',limit=None):
    '''
    returns list of summaries in the given directory
    Input:
        dir (str): path to directory containing the summaries
        sum_col_label (str): Name of the column containing the summary in each file
        limit (int): Upper limit of summaries to return
    '''
    if os.path.isfile(dir):
        # we read from a csv
        caselaw_dataset = CaseLawDataset(path=dir,
                                         limit=None,
                                         removeDuplicates=args.remove_duplicates
                                         )
        caselaw_dataset = caselaw_dataset if args.split_dataset is None else dataset_splitter(caselaw_dataset, args.split_dataset)[1]
        summaries = [item[1] for item in caselaw_dataset]
        return summaries

    # if usecols is not None:
    #     df = reader_func(path, usecols=usecols)
    # else:
    #     df = reader_func(path)
    all_files = glob.glob(os.path.join(dir, "*.csv")) + glob.glob(os.path.join(dir,"*.json"))
    # print(all_files)
    all_files = sorted(all_files, key = get_file_index)
    limit = len(all_files) if limit is None else min(len(all_files), limit)
    return [read_str_from_file(file, sum_col_label=sum_col_label) for file in all_files[:limit]]

def prepare_results_per_metric(m, p, r, f, stds=None):
    '''
    Used for pretty printing of the rouge package results per metric
    '''
    if stds == None:
        return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
    else:
        return  f"\t{m}\t P: {p}+/-{stds[0]}\t R: {r}+/-{stds[1]}\t F1: {f}+/-{stds[2]}"

def print_rouge(metric, results):
    '''
    Used for pretty printing of the rougepackage results
    '''
    for hypothesis_id, results_per_ref in enumerate(results):
        nb_references = len(results_per_ref['p'])
        for reference_id in range(nb_references):
            print( prepare_results_per_metric(metric,results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))

def get_metrics_from_results(results):
    '''
    Returns metric results from score calculated using the rouge package.
    Input must be a one-to-one rouge metric score
    Output: a matrix of all metric results for this pair of generated/reference summary (N_metrics x 3)
    '''
    # metrics used
    metrics=list(results.keys())
    # submetrics used (e.g f1, accuracy,...)
    submetrics = list(results[metrics[0]].keys())
    # sort results by the metric identifier: ROUGE-1 < ROUGE-N < .. ROUGE-L < ROUGE-W
    results = sorted(scores.items(), key=lambda x: x[0])
    # keep only the metric values and not the identifier
    results = np.array([list(x[1][0].values()) for x in results])
    # squeeze result from (Nmetric x Nsubmetric x 1) to (Metric x Nsubmetric)
    return np.squeeze(results)

def print_aggregate_statistics(results,metrics, submetrics):
    '''
    Prints aggregate statistics from the rouge evaluation metrics over the whole dataset
    Expects reults to be a numpy array of the form Nsamples x Nmetrics x Nsubmetrics
    '''
    print("Aggregate dataset statistics")
    mean_res = np.mean(results, axis=0)
    std_res = np.std(results, axis=0)
    print("metrics is", metrics)
    for m_idx, metric in enumerate(metrics):
        print(prepare_results_per_metric(metric,
                                         mean_res[m_idx][1],
                                         mean_res[m_idx][2],
                                         mean_res[m_idx][0],
                                         [std_res[m_idx][1],std_res[m_idx][2],std_res[m_idx][0]]
                                         )
              )


def evaluator_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'reference_sum',
        type=str,
        help="Path to dataset case law reference summaries."
    )
    parser.add_argument(
        'generated_sum',
        type=str,
        help="Path to dataset case law generated summaries."
    )
    parser.add_argument(
        '-o',
        '--output',
        default='evaluation_results/',
        nargs='?',
        type=str,
        help = "Path to output the evaluation results."
    )
    parser.add_argument(
        '-m',
        '--metric',
        default='all',
        nargs='?',
        type=str,
        help = "Which evaluation metric to use."
    )
    parser.add_argument(
        '-l',
        '--limit',
        default=None,
        type=int,
        help="Number of generated-reference summary pairs to evaluate."
    )
    parser.add_argument(
        '--ignore_stopwords',
        action='store_true',
        default=False,
        help="Ignore stopwords from texts before evaluating."
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help = "Do not print evaluation result for every pair."
    )
    parser.add_argument(
        '--stem',
        # required=True,
        action='store_true',
        default=False,
        # type=bool,
        help="Enable word stemming preprocessing before evaluating metric scores."
    )

    parser.add_argument(
        '--split_dataset',
        default=None,#"0.7_0.15_0.15",
        type=str,
        help="If flag is not used, then the whole dataset is summarized. Else, split percentages for train/dev/test | train/test must be provided. In both cases only test subset is summarized."
    )

    parser.add_argument(
        '--remove_duplicates',
        action='store_true',
        default=False,
        help="Whether to remove duplicate dataset entries."
    )

    parser.add_argument(
        '--summaries_together',
        action='store_true',
        default=False,
        help= "whether generated and reference summary are present in the same json file. This works only summary path is a folder and not a single file."
    )


    return parser

if __name__=="__main__":

    parser = evaluator_argparser()
    args = parser.parse_args()
    print(args)

    if args.summaries_together:
        reference_summaries = read_summaries_from_directory(args.reference_sum, sum_col_label='reference_summary')
        generated_summaries = read_summaries_from_directory(args.generated_sum, sum_col_label='summary')

    else:
        reference_summaries =  read_summaries_from_directory(args.reference_sum)
        generated_summaries = read_summaries_from_directory(args.generated_sum)

    # print(reference_summaries[0],"!!!!\n\n!!!!")
    # produce warning if number of generated summaries is not equal to number of reference
    if len(generated_summaries) != len(reference_summaries):
        warnings.warn(f"Number of generated summaries ({len(generated_summaries)}) \
        is not equal to number of reference summaries ({len(reference_summaries)})")

    N = min(len(generated_summaries), len(reference_summaries))
    if args.limit is not None:
        N = min(N, args.limit)

    if args.metric in ["ROUGE", 'all']:
        # -----------starting with rouge package------------
        print("args.stem is ", args.stem)

        MAX_N = 3
        metrics=['rouge-n', 'rouge-l', 'rouge-w']
        all_metrics = ['rouge-'+str(i) for i in range(1,MAX_N+1)] + ['rouge-l', 'rouge-w']
        all_submetrics= ['f1', 'p', 'r']
        N_METRICS = len(metrics) - 1 + MAX_N

        # construct evaluator
        evaluator = rouge.Rouge(metrics=metrics,
                               max_n=MAX_N, # Up to which N to calculate ROUGE-N
                               limit_length=True,
                               length_limit=100,
                               length_limit_type='words',
                               apply_avg=False,
                               apply_best=False,
                               alpha=0.5, # Default F1_score
                               weight_factor=1.2,
                               stemming=args.stem, ignore_stopwords=args.ignore_stopwords,
                               ensure_thesis_compatibility=True)


        # results per metric and per generated & reference pair x 3 (accuracy, recall, f1)
        dataset_results = np.zeros((N, N_METRICS, 3))
        # dataset_results = np.zeros(N)

        for index in range(N):
            try:
                scores =  evaluator.get_scores(generated_summaries[index],
                                           reference_summaries[index])
            except Exception as e:
                scores = {'rouge-2': [{'f': [0.0], 'p': [0.0], 'r': [-0.0]}], 'rouge-1': [{'f': [0.0], 'p': [0.0], 'r': [0.0]}], 'rouge-3': [{'f': [0.0], 'p': [0.0], 'r': [-0.0]}], 'rouge-l': [{'f': [0.0], 'p': [0.0], 'r': [0.0]}], 'rouge-w': [{'f': [0.0], 'p': [0.0], 'r': [0.0]}]}
                warnings.warn(f"error in evaluating metric for pair: {index}\n Ref: {reference_summaries[index]}\n Generated: {generated_summaries[index]}", e)
            # print(scores)

            if not args.quiet:
                print("Generated & Reference pair: ", index)
                print(scores)

            dataset_results[index, :, :] = get_metrics_from_results(dict([(metric, val[0]) for metric, val in scores.items()]))
            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                if not args.quiet:
                    print_rouge(metric, results); print()

        # print whole dataset aggregate statistics
        print_aggregate_statistics(dataset_results,
                                   metrics=all_metrics,
                                   submetrics=all_submetrics)

    elif args.metric in['rouge_score', 'all']:
        N_METRICS=2
        # results per metric and per generated & reference pair x 3 (accuracy, recall, f1)
        dataset_results = np.zeros((N, N_METRICS, 3))
        # dataset_results = np.zeros(N)

        scorer = rouge_scorer.RougeScorer( ['rougeLsum'], use_stemmer=True)
        for index in range(N):

            try:
                scores = scorer.score(generated_summaries[index],
                                           reference_summaries[index])
                # scores =  evaluator.get_scores(generated_summaries[index],
                #                            reference_summaries[index])
            except Exception as E:
                print(E)
                scores = {'rouge-2': [{'f': [0.0], 'p': [0.0], 'r': [-0.0]}], 'rouge-1': [{'f': [0.0], 'p': [0.0], 'r': [0.0]}], 'rouge-3': [{'f': [0.0], 'p': [0.0], 'r': [-0.0]}], 'rouge-l': [{'f': [0.0], 'p': [0.0], 'r': [0.0]}], 'rouge-w': [{'f': [0.0], 'p': [0.0], 'r': [0.0]}]}
                warnings.warn(f"error in evaluating metric for pair: {index}")

            if not args.quiet:
                print("Generated & Reference pair: ", index)
                print(scores)

            print(scores)
            # dataset_results[index, :, :] = get_metrics_from_results(dict([(metric, val[0]) for metric, val in scores.items()]))
            # for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            #     if not args.quiet:
            #         print_rouge(metric, results); print()



    elif args.metric in ['METEOR', 'all']:
        # results per generated & reference pair x 3 (accuracy, recall, f1)
        dataset_results = np.zeros((N, 4))

        for index in range(N):
            if not args.quiet:
                print("Generated & Reference pair: ", index)
            print(word_tokenize(reference_summaries[index]))

            precision = meteor(references=[word_tokenize(reference_summaries[index])],
                               hypothesis=[word_tokenize(generated_summaries[index])],
                               alpha=1)
        #     recall = meteor(references=reference_summaries[index],
        #                     hypothesis=generated_summaries[index],
        #                     alpha=0)
        #     f1 = meteor(references=reference_summaries[index],
        #                 hypothesis=generated_summaries[index],
        #                 alpha=0.5)
        #     default = meteor(references=reference_summaries[index],
        #                        hypothesis=generated_summaries[index])
        #
        #     dataset_results[index,:] = [precision, recall, f1, default]
        #     if not args.quiet:
        #         print(f"P: {round(precision,4)} \t R: {round(recall,4)} \t F1: {round(f1,4)} \t DEF: {round(default,4)} ")
        #
        # aggregate_results=np.mean(dataset_results, axis=0)
        # print(f"Aggregate Results")
        # print(f"P: {round(aggregate_results[0],4)}\t R: {round(aggregate_results[1],4)}\t F1: {round(aggregate_results[2],4)}\t DEF: {round(aggregate_results[3],4)} ")
