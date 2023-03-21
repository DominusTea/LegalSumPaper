'''
analyzes data crawled from the dataset crawler spiders
Usage: python3 analysis.py filapath_to.csv
'''
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#~~~DROP DUPLICATES ~~~~~~
import nltk
from nltk.util import ngrams
# import spacy
from fuzzywuzzy import fuzz

#~~~LENGTH STATS ~~~~~~
import os
from utils.text import adapt_spacy, count_tokens, count_sentences, count_tokens_per_sentence, remove_dupl_whitespace
import spacy
import pandas as pd
# import swifter
from tqdm import tqdm
import time

#~~~~ GRUSKY STATS~~~~~~
from utils.fragments import Fragments
#~~~~~~~~~~~~~~~~~~~

#--------GRUSKY PLOTS-----
import seaborn as sns
#-----------------------

DROP_NULL_TEXTORSUM=False
DROP_DUPLICATES=False
LENGTH_STATS=False
GRUSKY_STATS=False
PLOT_GRUSKY_STATS=True

def drop_null_text_or_sum(df):
    df = df[~df['summary'].isna()]
    df = df[~df['text'].isna()]
    return df

def faster_df_apply_whole_row(df, func):
    cols = list(df.columns)
    data, index = [], []
    for row in df.itertuples(index=True):
        row_dict = {f:v for f,v in zip(cols, row[1:])}
        data.append(func(row_dict))
        index.append(row[0])
    return pd.Series(data, index=index)

def faster_df_apply(df, func):
    # cols = list(df.columns)
    data, index = [], []
    for row in df.items():

        data.append(func(row[1]))
        index.append(row[0])
    return pd.Series(data, index=index)



def stringMatching_deduplicate(text_lst):
    '''
    returns indices list of deduplicated text list, after applying string matching
    '''
    if len(text_lst) == 1:
        return [0]

    kept_text_idx_lst = [i for i in range(len(text_lst))] #list of indexes for the texts we will keep
    # print(kept_text_idx_lst)
    N_text = len(text_lst)
    for idx,text in enumerate(text_lst):
        for comp_idx, comparison_text in enumerate(text_lst[idx:min(N_text-1,idx+10)]):
            if comp_idx == 0:
                pass
            else:
                comparison_len = min(len(text), len(comparison_text))
                start_comp_idx, stop_comp_idx = round(comparison_len*0.4), round(comparison_len*0.5)
                subtext=text[start_comp_idx:stop_comp_idx]
                sub_comp_text=comparison_text[start_comp_idx:stop_comp_idx]
                sim = fuzz.ratio(subtext,sub_comp_text)
                # print(sim)
                if sim > 92:
                    # print("sim: ", sim, "\n",subtext,"\n!!!!\n", sub_comp_text, "\n~~~~~~~\n")
                    try:
                        kept_text_idx_lst.remove(idx)
                        break
                    except:
                        # pass
                        print("tried to remove ", idx, "when comp_idx = ", comp_idx, " and len = ", len(text_lst))
    return [text_lst[i] for i in kept_text_idx_lst]


def n_gram_analysis(text_lst):
    cluster_lexicon = set()
    # get cluster's token vocabulary
    for text in text_lst:
        # by taking the union of each text's vocabulary
        text_tokens = set(get_tokens_from_text(row))
        cluster_lexicon = cluster_lexicon.union(text_tokens)

        # create index for each token
        cluster_lexicon_lst = list(cluster_lexicon)
        cluster_token_idx_dict ={token:idx for idx,token in enumerate(cluster_lexicon_lst)}
        #vocab size
        vocab_size = len(cluster_token_idx_dict)
        # create N-gram sparse representations

def get_tokens_from_text(txt,n=3):
    '''
    returns list of unique n-gram tokens from text
    '''
    assert(n>0)
    print(type(txt), type(txt[0]))
    tokenized_txt = nltk.word_tokenize(txt[0])
    if n==1:
        return tokenized_txt
    elif n<5:
        res =  list(ngrams(tokenized_txt, n))
        print(res); return res
    else:
        raise NotImplementedError("implemented ngrams up to n=4")

def dedupl_df(df):

    #----Deduplication-----
    df1 = df[['url']]
    duplicates= df[df1.duplicated()]
    deduplicated = df.drop_duplicates(subset=['url'])

    print("Number of entries in the original Dataframe", df.shape[0])
    print("Number of entries in the duplicates Dataframe ",duplicates.shape[0] )
    print("Number of entries in the deduplicated Dataframe ",deduplicated.shape[0] )

    #----Text length statistics-------

    # ignore or not erimodikia
    # deduplicated.drop(deduplicated.loc[df['case_category'] == 'Ερημοδικία_αναιρεσείοντος'].index, inplace=True)
    deduplicated=deduplicated[ deduplicated['case_category'] != 'Ερημοδικία_αναιρεσείοντος' ]
    deduplicated=deduplicated[ deduplicated['case_category'] != 'Ερημοδικία αναιρεσείοντος' ]

    return deduplicated


if __name__=="__main__":

    try:
        input_file = sys.argv[1]
    except FileNotFoundError as err:
        print("Could not find file: ", err)
    except IndexError:
        print("Must provide inputfile")
    print(input_file)
    filetype = input_file.split(".")[-1]
    file_dir = os.sep.join(input_file.split(os.sep)[:-1])
    print("file directory is ", file_dir)
    if filetype == "csv":
        df = pd.read_csv(input_file)
    elif filetype == "json":
        df = pd.read_json(input_file)
    else:
        raise ValueError("file must be of type: [csv|json]. Given file is of type ", filetype)

    if DROP_NULL_TEXTORSUM:
        print("Total number of entries: ", len(df))
        df = df[~df['summary'].isna()]
        df = df[~df['text'].isna()]
        print("Total number of entries after removing entries without text or summary: ", len(df))


    if DROP_DUPLICATES:
        deduplicated=dedupl_df(df)
        try:
            N_categ = deduplicated['case_category'].nunique()
            print("Number of unique case categories in the deduplicated Dataframe ", N_categ)
            # ----case-law category statistics-----

            categ_df=deduplicated['case_category'].str.replace('_'," "  )
            # print(categ_df)
            freq_df = categ_df.value_counts()/categ_df.size
            print(freq_df)
            mean_freq, std_freq = freq_df.mean(), freq_df.std()
            print("mean class frequency: ", mean_freq, " class frequency std: ", std_freq)

            fig = plt.figure(figsize=(19.20,9.83))
            plt.rcParams.update({'font.size': 22})
            # plt.rcParams.update({'axes.font.size': 22})
            plt.rc('xtick', labelsize=17)
            plt.rc('ytick', labelsize=17)
            plt.ylabel("Category relative frequency")

            ax = freq_df.iloc[0:10].plot(kind="bar",
                         title="Class category frequency of the 10 most frequent classes"
                         )
            #
            # plt.bar(x=["average frequency"],
            #             height=[mean_freq],
            #             yerr=[std_freq],
            #             color='orange'
            #             )

            # plt.savefig()
            plt.xticks(rotation=25)
            plt.savefig(os.path.join(file_dir,'categories.png'), bbox_inches='tight')
            plt.show()
            print(freq_df)


            #----incluster category statistics----


            categ_text_lst_df = deduplicated.groupby(['case_category'])['text'].apply(list)
            # print(categ_text_lst_df)
            # iterate over each category class
            removed=0
            i=0
            for category, row in categ_text_lst_df.items():
                print("category index: ", i, " category: ", category)
                i+=1
                # print(len(row))
                # print(type(row))
                # find proportion of kept texts
                deduplicated_texts = stringMatching_deduplicate(row)
                removed += len(row) - len(deduplicated_texts)
                print("kept percentage: ",round(len(deduplicated_texts)/len(row), 4) )
            print("removed: " , removed)

        except:
            raise ValueError("categories column is missing or contains only null values")


    if LENGTH_STATS:
        deduplicated=dedupl_df(df)
        deduplicated=drop_null_text_or_sum(deduplicated)
        print("~~~~~~LENGTH STATS~~~~~~~~~~~~~")
        print("loaded df with length: ", len(deduplicated))
        spacy_model='el_core_news_sm'
        tqdm.pandas(desc="my bar!")
        nlp = spacy.load(spacy_model)#, disable=["textcat", "ner", "tok2vec"])  #disable=["parser", "ner","textcat", "tok2vec"])
        nlp = adapt_spacy(nlp)
        print("spacy model loaded and adapted to legal domain")


        count_tokens_fn = lambda x: count_tokens(x, nlp)
        count_sentences_fn = lambda x: count_sentences(x, nlp)
        count_tokens_per_sentence_fn = lambda x: count_tokens_per_sentence(x, nlp)

        deduplicated=deduplicated.dropna(subset=['text','summary'])
        texts = deduplicated['summary'].apply(remove_dupl_whitespace)
        # texts = deduplicated['text'].apply(remove_dupl_whitespace)
        print("removed uncessecary whitespaces")


        df_lengths = texts.apply(len)
        text_mean_len, text_len_std = df_lengths.mean(), df_lengths.std()
        print("calculated character level stats")
        print("Average length of each document's text in characters: ", text_mean_len, " +/- ", text_len_std)

        # s_time = time.time()
        df_tokens=texts.progress_apply(count_tokens_fn)
        # # df_tokens = texts.loc[0:100].swifter.apply(count_tokens_fn)
        # # df_tokens = faster_df_apply(texts.loc[0:100], count_tokens_fn)
        # # df_tokens=texts.loc[0:100].apply(count_tokens_fn)
        # print("time is ", time.time()-s_time)
        text_token_mean_len, text_token_len_std = df_tokens.mean(), df_tokens.std()
        print("calculated token level stats")
        print("Average length of each document's text in tokens: ", text_token_mean_len, " +/- ", text_token_len_std)
        #


        df_sentences = texts.progress_apply(count_sentences_fn)
        text_sentence_mean_len, text_sentence_len_std = df_sentences.mean(), df_sentences.std()
        print("calculated sentence level stats")
        print("Average number of sentences in  document: ", text_sentence_mean_len, " +/- ", text_sentence_len_std)


        df_tokens_per_sentence = texts.progress_apply(count_tokens_per_sentence_fn)
        text_token_per_sentence_mean_len, text_token_per_sentence_len_std = df_tokens_per_sentence.mean(), df_tokens_per_sentence.std()
        print("calculated token per sentence level stats")
        print("Average number of tokens per sentence per document: ", text_token_per_sentence_mean_len, " +/- ", text_token_per_sentence_len_std)

        with open(file_dir+"stats.txt", "w+") as f:
            f.writelines("Average length of each document's text in characters: " + str(text_mean_len) +  " +/- " + str(text_len_std))
            f.writelines("Average length of each document's text in tokens: " + str(text_token_mean_len) + " +/- " + str(text_token_len_std))
            f.writelines("Average number of sentences in  document: " + str(text_sentence_mean_len) + " +/- " + str(text_sentence_len_std))
            f.writelines("Average number of tokens per sentence per document: " + str(text_token_per_sentence_mean_len) + " +/- " + str(text_token_per_sentence_len_std))

    if GRUSKY_STATS:
        deduplicated=dedupl_df(df)
        deduplicated=drop_null_text_or_sum(deduplicated)
        text_idx, summary_idx = list(deduplicated.columns).index('text'),list(deduplicated.columns).index('summary')

        tqdm.pandas(desc="my bar!")
        all_stats_fn = lambda x: Fragments(x[text_idx], x[summary_idx]).all_stats()
        coverage_stats_fn = lambda x: Fragments(x[text_idx], x[summary_idx]).coverage()
        density_stats_fn = lambda x: Fragments(x[text_idx], x[summary_idx]).density()
        compression_stats_fn = lambda x: Fragments(x[text_idx], x[summary_idx]).compression()
        deduplicated= deduplicated.dropna(how="any",subset=['text','summary'])
        # stats_fn = lambda x: x[3]
        deduplicated['text'] = deduplicated['text'].apply(remove_dupl_whitespace)

        df_grusky_stats=pd.DataFrame()
        df_grusky_stats['all_stats'] = deduplicated.progress_apply(all_stats_fn,axis=1)
        # import IPython ; IPython.embed() ; exit(1)
        df_grusky_stats=pd.concat([df_grusky_stats['all_stats'], df_grusky_stats['all_stats'].str.split(',', expand=True)], axis=1)


        df_grusky_stats.to_csv(os.path.join(file_dir,'grusky_stats.csv'))

        # for i in tqdm(range(10)):
        #
        #     text, summary = deduplicated['text'][i], deduplicated['summary'][i]
        #     fragments = Fragments(summary, text)
        #     cov = fragments.coverage()
        # #     print("Coverage:",    fragments.coverage())
        # #     print("Density:",     fragments.density())
        # #     print("Compression:", fragments.compression())

    if PLOT_GRUSKY_STATS:

        sns.set_style("white")
        df_grusky_stats=pd.read_csv(os.path.join(file_dir,'grusky_stats.csv'))
        gfg=sns.kdeplot(x=df_grusky_stats['0'], y=df_grusky_stats['1'], cmap="Oranges", shade=True,gridsize=750)
        gfg.set_xlabel("Extractive Fragment Coverage")
        gfg.set_ylabel("Extractive Fragment Density")
        gfg.set_ylim(0,4)
        gfg.set_xlim(0,1)
        plt.text(0.1, 3.5, "AreiosPagos",horizontalalignment='left', size='medium', color='black')
        plt.text(0.1, 3.0, "n = 8,395",horizontalalignment='left', size='medium', color='black')
        plt.text(0.1, 2.5, "c = 37:1",horizontalalignment='left', size='medium', color='black')
        # plt.savefig(os.path.join(file_dir, "grusky_stats_plot.png"))
        plt.show()
