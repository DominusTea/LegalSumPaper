import spacy
import random

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

    # remove unecessary pipeline components
    nlp.remove_pipe("ner")
    return nlp

class RandomExtractor_Summarizer():
    def __init__(self, spacy_model='el_core_news_sm'):
        self.nlp = spacy.load(spacy_model)
        # adapt spaCy pipeline
        self.nlp = adapt_spacy(self.nlp)

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

    def summarize(self, text, generate_serializable=True, limit_sentences=None, ref_summary=None, limit_tokens=None):
        # parse document
        doc = self.nlp(text)
        # .. and get its sentences
        sentences = list(doc.sents)
        # print(sentences, len(sentences))
        # get number of sentences in text
        N_sentences = len(sentences)
        if limit_sentences is not None:
            limit_sentences = min(N_sentences, limit_sentences)
        else:
            limit_sentences=N_sentences
        # get random sentence indexes
        idxs = random.sample(range(N_sentences), limit_sentences)
        selected_sentences = [sentences[idx].text for idx in idxs]

        summary_str = (" ".join(selected_sentences) if generate_serializable else selected_sentences).strip()

        # if generate_serializable:
        #     selecteted_str =  " ".join(selected_sentences)
        # else:
        #     selecteted_str = selected_sentences
        # truncate summary string into limit_
        if limit_tokens is not None:
            try:
                limit_tokens = int(len(self.nlp(ref_summary))*limit_tokens)
            except:
                limit_tokens = 1
            summary_str = self._truncate_summary(summary_str, N_tokens=limit_tokens)


        return summary_str
