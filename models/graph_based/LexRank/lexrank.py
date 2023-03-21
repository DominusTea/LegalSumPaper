import spacy

# from models.graph_based.LexRank.lexrank_dev.lexrank.LexRank import LexRank as lexrank
from lexrank import LexRank as lexrank
from lexrank import STOPWORDS

import time
import warnings

# for adapt_spacy
from spacy.symbols import ORTH, NORM
from nltk.tokenize import sent_tokenize


def adapt_spacy(nlp):
    '''
    Adapts spacy pipeline to AreiosPagos caselaw dataset
    Should, probably, be moved elsewere after testing
    '''
    # legal text exceptions
    nlp.tokenizer.add_special_case('παρ.', [{ORTH: 'παρ.', NORM:'παράγραφος'  }])

    nlp.tokenizer.add_special_case('Π.Κ.', [{ORTH: 'Π.', NORM: 'ποινικος'},
                                             {ORTH: 'Κ.', NORM: 'κώδικας'}])
    nlp.tokenizer.add_special_case('Κ.Ποιν.Δ.',
                                   [{ORTH: 'Κ.', NORM:'κώδικας'  },
                                    {ORTH: 'Ποιν.', NORM: 'ποινικός'},
                                    {ORTH: 'Δ.', NORM: 'δικονομία'}
                                    ])


    nlp.tokenizer.add_special_case('Κ.Π.Δ.',
                                   [{ORTH: 'Κ.', NORM:'κώδικας'  },
                                    {ORTH: 'Π.', NORM: 'ποινικός'},
                                    {ORTH: 'Δ.', NORM: 'δικονομία'}
                                    ])

    nlp.tokenizer.add_special_case('ΚΠΔ.',
                                   [{ORTH: 'Κ', NORM:'κώδικας'  },
                                    {ORTH: 'Π', NORM: 'ποινικός'},
                                    {ORTH: 'Δ.', NORM: 'δικονομία'}
                                    ])

    nlp.tokenizer.add_special_case('ΚΠοινΔ.',
                                   [{ORTH: 'Κ', NORM:'κώδικας'  },
                                    {ORTH: 'Ποιν', NORM: 'ποινικός'},
                                    {ORTH: 'Δ.', NORM: 'δικονομία'}
                                    ])


    nlp.tokenizer.add_special_case('ΠΚ.', [{ORTH: 'Π', NORM: 'ποινικός'},
                                            {ORTH: 'Κ.', NORM: 'κώδικας'}])

    nlp.tokenizer.add_special_case('αρ.', [{ORTH: 'αρ.', NORM:'αριθμός'  }])
    nlp.tokenizer.add_special_case('αρθρ.', [{ORTH: 'αρθρ.', NORM:'άρθρο'  }])

    nlp.tokenizer.add_special_case('Αρ.', [{ORTH: 'Αρ.', NORM:'αριθμός'  }])
    nlp.tokenizer.add_special_case('Αρθρ.', [{ORTH: 'Αρθρ.', NORM:'άρθρο'  }])

    nlp.tokenizer.add_special_case('κεφ.', [{ORTH: 'κεφ.', NORM:'κεφάλαιο'  }])
    nlp.tokenizer.add_special_case('Κεφ.', [{ORTH: 'Κεφ.', NORM:'κεφάλαιο'  }])

    nlp.tokenizer.add_special_case('στοιχ.', [{ORTH: 'στοιχ.', NORM:'στοιχείο'  }])

    nlp.tokenizer.add_special_case('ν.', [{ORTH: 'ν.', NORM:'νόμος'  }])
    nlp.tokenizer.add_special_case('Ν.', [{ORTH: 'Ν.', NORM:'νόμος'  }])

    nlp.tokenizer.add_special_case('εδ.', [{ORTH: 'εδ.', NORM:'εδάφιο'  }])

    nlp.tokenizer.add_special_case('αριθ. κατ.',
                                   [{ORTH: 'αριθ. ', NORM:'αριθμός'  },
                                    {ORTH: 'κατ.', NORM: 'καταχώρηση'}
                                    ])

    nlp.tokenizer.add_special_case('Μον. Πλημ.',
                                   [{ORTH: 'Μον. ', NORM:'μονομελές'  },
                                    {ORTH: 'Πλημ.', NORM: 'πλημμελειοδικων'},
                                    ])

    nlp.tokenizer.add_special_case('περ.', [{ORTH: 'περ.', NORM:'περίπτωση'  }])
    nlp.tokenizer.add_special_case('υποπερ.', [{ORTH: 'υποπερ.', NORM:'υποπερίπτωση'  }])

    nlp.tokenizer.add_special_case('Ολ.', [{ORTH: 'Ολ.', NORM:'Ολομέλεια'  }])
    nlp.tokenizer.add_special_case('ΑΠ', [{ORTH: 'Α', NORM:'Άρειος'  },
                                          {ORTH: 'Π', NORM: 'Πάγος'}
                                          ])
    nlp.tokenizer.add_special_case('Ολ.ΑΠ', [{ORTH: 'Ολ.', NORM:'Ολομέλεια'  },
                                             {ORTH: 'Α', NORM:'Άρειος'  },
                                             {ORTH: 'Π', NORM: 'Πάγος'}
                                            ])


    nlp.tokenizer.add_special_case('αριθ. πρωτ.',
                                   [{ORTH: 'αριθ. ', NORM:'αριθμός'  },
                                    {ORTH: 'πρωτ.', NORM: 'πρωτόκολλο'}
                                    ])

    # Institutions
    nlp.tokenizer.add_special_case('Ο.Α.Ε.Δ', [{ORTH: 'Ο.Α.Ε.Δ'}])



    # reducted text exception
    nlp.tokenizer.add_special_case('...', [{ORTH: '...', NORM: '[RDCTD]'}])


    # name initials
    greek_capital_letters=[chr(c) for c in  range(0x391, 0x3aa) if chr(c).isalpha()]
    # get every possible capital letter pair (X. X)
    capitals_pairs = [a + ". "+ b+ ". " for idx, a in enumerate(greek_capital_letters) \
     for b in greek_capital_letters[idx + 1:]]
    for pair in capitals_pairs:
        nlp.tokenizer.add_special_case(pair, [{ORTH: pair}])

    return nlp

class LexRank_Summarizer():
    def __init__(self, documents=None, spacy_model='el_core_news_sm', similarity_metric='idf_mod_cosine'):
        self.nlp = spacy.load(spacy_model)
        # adapt spaCy pipeline
        self.nlp = adapt_spacy(self.nlp)
        # get training documents from list of strings to list of list of strings (each string corresponds to a sentence)
        start = time.time()
        # documents = [self._segment_to_sentences(text) for text in documents]
        documents = self._segment_docs_to_sentences(documents)
        end = time.time()
        print("sentence segmentation elapsed time: ", end-start)
        # get lexrank summarizer - train idf scores
        self.lxr = lexrank(documents,
                           stopwords=STOPWORDS['el'],
                           sim=similarity_metric)

    def get_idf_scores(self):
        return self.lxr.get_idf_scores()

    def load_idf_scores(self,scores,default_val=0):
        self.lxr.load_idf_scores(scores, default_val)

    def _segment_to_sentences(self, text):
        '''
        segments input text (str) to sentences (str) using the spacy pipeline
        Input: text(str)
        Output: list(str)
        '''
        print("starting sentence segmentation")

        return list(d.text for d in self.nlp(text).sents)

    def _segment_docs_to_sentences(self, documents):
        '''
        like _segment_to_sentences but works for list of texts and produces
        a list of similar results
        '''
        if documents is None:
            warnings.warn("Segmenting None documents")
            return documents
        else:
            docs = self.nlp.pipe(documents,
                                 disable=["ner"],
                                 )
            res = [list(sent.text for sent in doc.sents) for doc in docs]
            # res = [sent.text for sent in doc.sents for doc in docs]
            # res = [ sent.text for sent in  [doc.sents for doc in docs] ]
            return res

    def _truncate_summary(self, summary, N_tokens):
        '''
        Truncates summary to N_tokens number of tokens
        Input:
            summary (str): Summary string to truncate
            N_tokens (int): Upper limit of tokens in the truncated summary
        Output:
            (str): The truncated Summary
        '''
        doc=self.nlp(summary)
        truncated_doc= doc[:N_tokens] if len(doc) > N_tokens else doc
        return truncated_doc.text

    def summarize(self, text,
                  ref_summary=None,
                  threshold=0.1,
                  generate_serializable=True,
                  limit_tokens=None,
                  limit_sentences=None,
                  bias_word=None,
                  limit_on_text=False):
        '''
        Summarizes given text
        Inputs:
            text (str):   Input text
            ref_summary (str): Text's reference summary. Not needed unless limit_tokens is not None
            threshold (int): Value of (lower bound) similarity threshold.
                             If None then idf-modified cosine is used.
            generate_serializable (bool): If True generates summary in string, otherwise list of strings
            limit_sentences (int): Number of sentences to keep for extractive summary
            limit_size (int): Number of tokens to keep
            limit_on_text (bool): whether limit_tokens applies to the reference summary (False) or the main text (True)
        '''
        # # apply spaCy nlp pipeline to segment text to sentences
        # doc = self.nlp(text)
        # sents = [sent.text for sent in doc]

        # sents = sent_tokenize(text)
        sents = list(d.text for d in self.nlp(text).sents)

        print("number of sentences : ", len(sents))
        # find correct threshold val:


        if limit_sentences is not None:
            summary = self.lxr.get_summary(sents,
                                           summary_size=limit_sentences,
                                           threshold = threshold,
                                           bias_word=bias_word,
                                           )
        else:
            summary = self.lxr.get_summary(sents,
                                           threshold = threshold,
                                           bias_word=bias_word)

        print(type(summary))
        summary_str = (" ".join(summary) if generate_serializable else summary).strip()

        # truncate summary string into limit_
        limit_base_text = ref_summary if not(limit_on_text) else text
        print("1. limt tokens is ", limit_tokens)
        if ref_summary is not None:
            print(self.nlp(ref_summary))
        if limit_tokens is not None:
            print("trying to limit tokens to ")
            try:
                limit_tokens = int(len(self.nlp(limit_base_text))*limit_tokens)
                # limit_tokens = int(len(self.nlp(ref_summary))*limit_tokens)
            except:
                limit_tokens=1
            print("limit tokens is ", limit_tokens)
            summary_str = self._truncate_summary(summary_str, N_tokens=limit_tokens)

        return summary_str
