import math
from collections import Counter, defaultdict

import numpy as np
import scipy

from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize, new_tokenize, adapt_spacy

import spacy
import pickle
import tqdm
import warnings
from transformers import AutoModel, AutoTokenizer
import torch

class LexRank:
    def __init__(
        self,
        documents=None,
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True,
        sim='idf_mod_cosine',
    ):
        self.sim = sim
        self.nlp = adapt_spacy(spacy.load('el_core_news_sm'))
        self.embeddings_nlp = spacy.load('test/full.5epochs.AP.spacy.greeklaw')
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords
        self.bias_word = None
        self.do_bias = False

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words

        if sim in ['bert_sim']:
            self.bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
            self.bert_tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        if sim in ['idf_mod_cosine']:
            if documents is None:
                warnings.warn("No documents provided. Idf scores will be initialized to default values")
                self.idf_default_value = 0
                self.idf_score = defaultdict(lambda: self.idf_default_value)

            else:
                print("starting calculating idf scores")
                self.idf_score, self.idf_default_value = self._calculate_idf(documents)
                print("finished calculating idf scores")
            # with open('models/assets/lexrank_with_idfs_precomputed', 'wb') as f:
            #     pickle.dump(dict(self.idf_score), f)

    def get_summary(
        self,
        sentences,
        summary_size=None,
        threshold=.03,
        fast_power_method=True,
        bias_word=None,
        bias_embedding='spaCy'
    ):
        if not (isinstance(summary_size, int) or isinstance(summary_size,float)):
            if summary_size is None:
                warnings.warn("No Summary size provided. The full text will be returned in order of LexRank importance")
            else:
                raise ValueError('\'summary_size\' should be a positive integer or float')

        self.num_sentences = len(sentences)

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
            bias_word=bias_word
        )
        # print("summar size", summary_size)
        print(lex_scores)
        if bias_word is None:
            sorted_ix = np.argsort(lex_scores)[::-1]
        else:
            sorted_ix = sorted(lex_scores, key=lex_scores.get)[::-1]
        for i in range(len(sentences)):
            print(f"~~~~~~~~~~~~{i}-----------\n",sentences[i])
        # print("sorted indexes is ", sorted_ix)
        # print("lex score is ", lex_scores)
        if isinstance(summary_size, float):
            summary = [sentences[i] if (i in sorted_ix[:int(len(sorted_ix)*summary_size)]) else '' for i in range(len(sorted_ix))]
        else:
            summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
        bias_word=None
    ):
        print("ranking sentences")
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )


        similarity_matrix = self._calculate_similarity_matrix(sentences,self.sim)

        self.do_bias = False if bias_word is None else True#(bias_word in self.embeddings_nlp.vocab) or not(self.nlp(bias_word)[0].is_oov)
        if self.do_bias:
            bias_sentence_similarity_arr = self._calculate_bias_similarity_array(sentences,
                                                                                 bias_word = bias_word)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
            damp=True,
            bias_sim_arr = bias_sentence_similarity_arr if self.do_bias else None
        )

        return scores

    def sentences_similarity(self, sentence_1, sentence_2):
        tf_1 = Counter(self.tokenize_sentence(sentence_1))
        tf_2 = Counter(self.tokenize_sentence(sentence_2))

        similarity = self._idf_modified_cosine([tf_1, tf_2], 0, 1)

        return similarity

    def tokenize_sentence(self, sentence, new_tokenizer=True, tokenizer=None):
        '''
        Returns list of tokens from sentence.
        If new_tokenizer == True: returns results similar to PyTextRank
        tokenizer: spacy pipeline which is used to tokenize each sentence. If None
        the 'el_core_news_sm' model is loaded every time.
        '''
        if tokenizer == None:
            tokenizer = self.nlp
        if new_tokenizer:
            tokens = new_tokenize(
                sentence,
                self.stopwords,
                keep_numbers=self.keep_numbers,
                keep_emails=self.keep_emails,
                keep_urls=self.keep_urls,
                nlp = tokenizer
                )
        else:
            tokens = tokenize(
                sentence,
                self.stopwords,
                keep_numbers=self.keep_numbers,
                keep_emails=self.keep_emails,
                keep_urls=self.keep_urls,
            )

        return tokens

    def _calculate_idf(self, documents):
        bags_of_words = []
        # print("documents is of type ", type(documents))
        # print("documents length is ", len(documents))
        for doc in tqdm.tqdm(documents):
            # print("doc's type is ", type(doc))
            # print("doc's length is ", len(doc))
            doc_words = set()

            for sentence in doc:
                # print("doc's sentence is ", sentence)
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)
            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = math.log(doc_number_total + 1)

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score, default_value

    def _calculate_similarity_matrix(self, sentences, similarity_metric):
        if similarity_metric != 'bert_sim':
            tf_scores = [
                Counter(self.tokenize_sentence(sentence)) for sentence in sentences
                ]
        if similarity_metric == 'bert_sim':
            sent = sentences[0]
            print("1")
            print(self.bert_tokenizer.encode(sent))
            print("2")
            # print(sentences)
            # print(self.bert_model(torch.tensor(self.bert_tokenizer.encode(sent)).view(1,-1))[0].shape)
            tokenized_sentences = self.bert_tokenizer(sentences, return_tensors='pt', truncation=True, padding='max_length', max_length=512)['input_ids']
            print("3")
            # print(tokenized_sentences.shape)

            sentence_embeddings = [self.bert_model(sent.view(1,-1)) for sent in tokenized_sentences]
            # sentence_embeddings = torch.mean(sentence_embeddings,axis=0)
            print(sentence_embeddings.shape, "!!!!!", len(sentences))
        length = len(sentences)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                if similarity_metric == 'idf_mod_cosine':
                    similarity = self._idf_modified_cosine(tf_scores, i, j)
                elif similarity_metric == 'common_words':
                    similarity = self._common_words_normalized([tf_scores[i],tf_scores[j]])
                elif similarity_metric ==  'word2vec_sim':
                    similarity = self._word_embeddings_similarity(tf_scores[i], tf_scores[j])
                elif similarity_metric == 'bert_sim':
                    similarity = self._bert_similarity(sentence_embeddings[i],sentence_embeddings[j])

                else:
                    raise ValueError("Similarity metric is not recognised")

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _bert_similarity(self, sent1, sent2):
        print("AAAAAA")
        emb1 = self.bert_model(torch.tensor(self.bert_tokenizer.encode(sent1)).view(1,-1))[0]
        emb2 = self.bert_model(torch.tensor(self.bert_tokenizer.encode(sent2)).view(1,-1))[0]


        norm_emb1 = torch.mean(emb1,axis=1)
        norm_emb2 = torch.mean(emb2,axis=1)
        print("BBBBBB")
        with torch.no_grad():
            cos_similarity = torch.nn.CosineSimilarity()
            print("CCCCCCCCC")
            return cos_similarity(norm_emb1, norm_emb2).item()


    def _word_embeddings_similarity(self, sent_tf_scores_1, sent_tf_scores_2):
        sentence1_length = sum(sent_tf_scores_1.values())
        sentence2_length = sum(sent_tf_scores_2.values())

        if sentence1_length == 0 or sentence2_length == 0:
            return 0

        sentence_1_word_embedded = np.array([count*self.idf_score[word]*self.embeddings_nlp.vocab.get_vector(word)
                                           for word, count
                                           in zip(sent_tf_scores_1.keys(), sent_tf_scores_1.values())])

        sentence_2_word_embedded = np.array([count*self.idf_score[word]*self.embeddings_nlp.vocab.get_vector(word)
                                           for word, count
                                           in zip(sent_tf_scores_2.keys(), sent_tf_scores_2.values())])



        # Add word embeddings of each sentence and
        # normalize sentence embeddings by their token length
        sentence_1_word_embedded = np.sum(sentence_1_word_embedded, axis=0)/sentence1_length
        sentence_2_word_embedded = np.sum(sentence_2_word_embedded, axis=0)/sentence2_length

        sentences_word_embeddings_similarity = 1 - scipy.spatial.distance.cosine(sentence_1_word_embedded, sentence_2_word_embedded)

        return sentences_word_embeddings_similarity




    def _calculate_bias_similarity_array(self, sentences, bias_word, use_counts=True):
        # get similarity between each sentence and the bias word
        if use_counts:
            def find_occurences_in_counter(bw_list, c):
                res = 0
                for bw in bw_list:
                    res += c[bw]
                return res

            tf_scores = [
            Counter(self.tokenize_sentence(sentence)) for sentence in sentences
            ]
            if isinstance(bias_word, str):
                bias_word_nlp_lst = self.tokenize_sentence(bias_word)
                if len(bias_word_nlp_lst) == 0:
                    return np.ones(len(tf_scores))/len(tf_scores)

                bias_sentences_sim = [find_occurences_in_counter(bias_word_nlp_lst,sent_tf_scores)/sum(sent_tf_scores.values()) if sum(sent_tf_scores.values())>0 else 0 for sent_tf_scores in tf_scores]
            elif isinstance(bias_word, list) or isisntance(bias_word, tuple):

                bias_word_nlp_lst = [self.tokenize_sentence(bw) for bw in bias_word]
                bias_word_nlp_lst = [item for sublist in bias_word_nlp_lst for item in sublist]

                bias_sentences_sim = [ find_occurences_in_counter(bias_word_nlp_lst, sent_tf_scores)/sum(sent_tf_scores.values()) if sum(sent_tf_scores.values()) >0 else 0  for sent_tf_scores in tf_scores]
            else:
                raise ValueError("Bias word should be either an iterable or a strng")
            bias_sentences_sim = np.array(bias_sentences_sim)
            if np.sum(bias_sentences_sim) >0:
                bias_sentences_sim = bias_sentences_sim/np.sum(bias_sentences_sim)
            else:
                # print("bias word noxt found returning uniform array")
                bias_sentences_sim = np.ones(len(tf_scores))/len(tf_scores)
            print("tf scores is ", tf_scores)
            print("returning ", bias_sentences_sim, len(tf_scores))
            return bias_sentences_sim

        else:
            tf_scores = [
                Counter(self.tokenize_sentence(sentence)) for sentence in sentences
                ]
            bias_sentences_sim_lst = [self._sentence_bias_similarity(sent_tf_scores, bias_word)
                                      for sent_tf_scores in tf_scores]
            # create numpy array and normalize it
            bias_sentences_sim = np.array(bias_sentences_sim_lst)
            bias_sentences_sim = bias_sentences_sim/np.sum(bias_sentences_sim)
            return bias_sentences_sim

    def _common_words_normalized(self, tf_scores):
        '''
        Counts number of common words and normalizes by sum of sentences length
        tf_score (collections.Counter object) {'word':number of occurences in sentence}
        '''
        # print(tf_scores)
        # get length of each sentence
        lengths = [sum(counter.values()) for counter in tf_scores]
        # print(lengths)
        if 0 in lengths:
            return 0
        # get denominator (log10 sum of sentences' length)
        normalization = sum(math.log10(length) for length in lengths)
        if normalization ==  0:
            return int(tf_scores[0] == tf_scores[1])
        # get common words (common keys in the counters object)
        # works only when calculating common word of 2 sentences
        common = tf_scores[0] & tf_scores[1]
        # get number of common words
        num_common_words = sum([tf_scores[0][word]+ tf_scores[1][word] for word in common])

        return num_common_words/normalization

    def load_idf_scores(self, scores, default_val=0):
        '''
        sets idf scores from input
        '''
        default_val_func = lambda:default_val
        print("default val", default_val)
        print("default_val_func", default_val_func)
        print("type of idf_scores", type(scores))
        # print("1", idf_scores[1])
        # print("idf_scores['ο']", idf_scores['ο'])
        self.idf_score = defaultdict(default_val_func, scores )

    def get_idf_scores(self):
        '''
        returns idf scores from object
        '''
        return self.idf_score, self.idf_default_value

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

    def _sentence_bias_similarity(self, sentence_tf_scores, bias_word, bias_embedding='spaCy'):
        '''
        Calculates similarity (cosine distance) of sentence vector embedding with
        a given bias word embedding
        Input:
            sentence_tf_scores (collections.Counter) {'word': count_of_occurences}
            bias word (str)
        '''
        # print(bias_word)
        sentence_length = sum(sentence_tf_scores.values())
        if self.do_bias:
            if len(sentence_tf_scores) == 0:
                return 0
            if isinstance(bias_word, str):
                # single word as bias
                if bias_embedding =='spaCy':
                    if bias_word in self.embeddings_nlp.vocab:
                        bias_embedded = self.embeddings_nlp.vocab.get_vector(bias_word)
                        sentence_word_embedded = np.array([count*self.embeddings_nlp.vocab.get_vector(word)
                                                           for word, count
                                                           in zip(sentence_tf_scores.keys(), sentence_tf_scores.values())])
                    else:
                        spacy_word = self.nlp(bias_word)[0]
                        bias_embedded = spacy_word.vector
                        sentence_word_embedded = np.array([count*self.nlp(word)[0].vector
                                                       for word, count
                                                       in zip(sentence_tf_scores.keys(), sentence_tf_scores.values())])
                else:
                    raise ValueError('Only spaCy bias embedding supported')
            else:
                # multiple words as bias
                assert(isinstance(bias_word, tuple) or isinstance(bias_word, list))
                # print("bias word = ",bias_word)
                sentence_word_embedded = np.array([count*self.embeddings_nlp.vocab.get_vector(word)
                                                   for word, count
                                                   in zip(sentence_tf_scores.keys(), sentence_tf_scores.values())])
                bias_embedded = np.sum([self.embeddings_nlp.vocab.get_vector(b) for b in bias_word], axis=0)
                bias_embedded /= len(bias_embedded)

            sentence_embedded = np.sum(sentence_word_embedded, axis=0)/sentence_length

            sentence_bias_similarity = 1 - scipy.spatial.distance.cosine(sentence_embedded, bias_embedded)
        else:
            raise ValueError("_sentence_bias_similarity should not be called when self.do_bias is calculated to be False")
        return sentence_bias_similarity

    def _markov_matrix(self, similarity_matrix):
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)
        # the following should not be needed since the sum of similarities should be >0 (self similarity is always non zero)
        row_sum = np.where(row_sum == 0, 1, row_sum )
        return similarity_matrix / row_sum

    def _markov_matrix_discrete(self, similarity_matrix, threshold):
        markov_matrix = np.zeros(similarity_matrix.shape)

        for i in range(len(similarity_matrix)):
            columns = np.where(similarity_matrix[i] > threshold)[0]
            markov_matrix[i, columns] = 1 / len(columns)

        return markov_matrix
