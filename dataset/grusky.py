'''
analyzes data crawled from the dataset crawler spiders
Usage: python3 analysis.py filapath_to.csv
'''
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import os
#~~~~ GRUSKY STATS~~~~~~

#~~~~~~~~~~~~~~~~~~~

################################################################################

import html as _html
import itertools as _itertools
import random as _random

from collections import namedtuple as _namedtuple

import spacy as _spacy
from spacy.symbols import ORTH, NORM
from os import system as _system
import re
import time
#### ORIGINAL CODE FROM https://github.com/lil-lab/newsroom/tree/master/newsroom/ ####
### COmplimentary to paper Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies
### SLIGHTLY adapt to greek legal texts
################################################################################

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


class Fragments(object):

    Match = _namedtuple("Match", ("summary", "text", "length"))

    @classmethod
    def _load_model(cls):
        model_str="el_core_news_sm"
        if not hasattr(cls, "_en"):

            try:

                cls._en = adapt_spacy(_spacy.load(model_str))

            except:

                _system("python -m spacy download "+model_str)
                cls._en = adapt_spacy(_spacy.load(model_str))

    def __init__(self, summary, text, tokenize = True, case = False):

        self._load_model()

        self._tokens = tokenize

        self.summary = self._tokenize(summary) if tokenize else summary.split()
        self.text    = self._tokenize(text)    if tokenize else text.split()

        self._norm_summary = self._normalize(self.summary, case)
        self._norm_text    = self._normalize(self.text, case)

        self._match(self._norm_summary, self._norm_text)


    def _tokenize(self, text):

        """
        Tokenizes input using the fastest possible SpaCy configuration.
        This is optional, can be disabled in constructor.
        """

        # return self._en(text, disable = ["tagger", "parser", "ner", "textcat"])
        return self._en(text)

    def _normalize(self, tokens, case = False):

        """
        Lowercases and turns tokens into distinct words.
        """

        return [
            str(t).lower()
            if not case
            else str(t)
            for t in tokens
        ]


    def overlaps(self):

        """
        Return a list of Fragments.Match objects between summary and text.
        This is a list of named tuples of the form (summary, text, length):
            - summary (int): the start index of the match in the summary
            - text (int): the start index of the match in the reference
            - length (int): the length of the extractive fragment
        """

        return self._matches


    def strings(self, min_length = 0, raw = None, summary_base = True):

        """
        Return a list of explicit match strings between the summary and reference.
        Note that this will be in the same format as the strings are input. This is
        important to remember if tokenization is done manually. If tokenization is
        specified automatically on the raw strings, raw strings will automatically
        be returned rather than SpaCy tokenized sequences.
        Arguments:
            - min_length (int): filter out overlaps shorter than this (default = 0)
            - raw (bool): return raw input rather than stringified
                - (default = False if automatic tokenization, True otherwise)
            - summary_base (true): strings are based of summary text (default = True)
        Returns:
            - list of overlaps, where overlaps are strings or token sequences
        """

        # Compute the strings against the summary or the text?

        base = self.summary if summary_base else self.text

        # Generate strings, filtering out strings below the minimum length.

        strings = [
            base[i : i + length]
            for i, j, length
            in self.overlaps()
            if length > min_length
        ]

        # By default, we just return the tokenization being used.
        # But if they user wants a raw string, then we convert.
        # Mostly, this will be used along with spacy.

        if self._tokens and raw:

            for i, s in enumerate(strings):
                strings[i] = str(s)

        # Return the list of strings.

        return strings

    def all_stats(self, summary_base=True):
        return str(self.coverage())+','+ str(self.density()) + ',' + str(self.compression())

    def coverage(self, summary_base = True):

        """
        Return the COVERAGE score of the summary and text.
        Arguments:
            - summary_base (bool): use summary as numerator (default = True)
        Returns:
            - decimal COVERAGE score within [0, 1]
        """

        numerator = sum(o.length for o in self.overlaps())

        if summary_base: denominator = len(self.summary)
        else:            denominator = len(self.reference)

        if denominator == 0: return 0
        else:                return numerator / denominator


    def density(self, summary_base = True):

        """
        Return the DENSITY score of summary and text.
        Arguments:
            - summary_base (bool): use summary as numerator (default = True)
        Returns:
            - decimal DENSITY score within [0, ...]
        """

        numerator = sum(o.length ** 2 for o in self.overlaps())

        if summary_base: denominator = len(self.summary)
        else:            denominator = len(self.reference)

        if denominator == 0: return 0
        else:                return numerator / denominator


    def compression(self, text_to_summary = True):

        """
        Return compression ratio between summary and text.
        Arguments:
            - text_to_summary (bool): compute text/summary ratio (default = True)
        Returns:
            - decimal compression score within [0, ...]
        """

        ratio = [len(self.text), len(self.summary)]

        try:

            if text_to_summary: return ratio[0] / ratio[1]
            else:               return ratio[1] / ratio[0]

        except ZeroDivisionError:

            return 0


    def _match(self, a, b):

        """
        Raw procedure for matching summary in text, described in paper.
        """

        self._matches = []

        a_start = b_start = 0

        while a_start < len(a):

            best_match = None
            best_match_length = 0

            while b_start < len(b):

                if a[a_start] == b[b_start]:

                    a_end = a_start
                    b_end = b_start

                    while a_end < len(a) and b_end < len(b) \
                            and b[b_end] == a[a_end]:

                        b_end += 1
                        a_end += 1

                    length = a_end - a_start

                    if length > best_match_length:

                        best_match = Fragments.Match(a_start, b_start, length)
                        best_match_length = length

                    b_start = b_end

                else:

                    b_start += 1

            b_start = 0

            if best_match:

                if best_match_length > 0:
                    self._matches.append(best_match)

                a_start += best_match_length

            else:

                a_start += 1


    def _htmltokens(self, tokens):

        """
        Carefully process tokens to handle whitespace and HTML characters.
        """

        return [
            [
                _html.escape(t.text).replace("\n", "<br/>"),
                _html.escape(t.whitespace_).replace("\n", "<br/>")
            ]

            for t in tokens
        ]


    def annotate(self, min_length = 0, text_truncation = None, novel_italics = False):

        """
        Used to annotate fragments for website visualization.
        Arguments:
            - min_length (int): minimum length overlap to count (default = 0)
            - text_truncation (int): tuncated text length (default = None)
            - novel_italics (bool): italicize novel words (default = True)
        Returns:
            - a tuple of strings: (summary HTML, text HTML)
        """

        start = """
            <u
            style="color: {color}; border-color: {color};"
            data-ref="{ref}" title="Length: {length}"
            >
        """.strip()

        end = """
            </u>
        """.strip()

        # Here we tokenize carefully to preserve sane-looking whitespace.
        # (This part does require text to use a SpaCy tokenization.)

        summary = self._htmltokens(self.summary)
        text = self._htmltokens(self.text)

        # Compute novel word set, if requested.

        if novel_italics:

            novel = set(self._norm_summary) - set(self._norm_text)

            for word_whitespace in summary:

                if word_whitespace[0].lower() in novel:
                    word_whitespace[0] = "<em>" + word_whitespace[0] + "</em>"

        # Truncate text, if requested.
        # Must be careful later on with this.

        if text_truncation is not None:
            text = text[:text_truncation]

        # March through overlaps, replacing tokens with HTML-tagged strings.

        colors = self._itercolors()

        for overlap in self.overlaps():

            # Skip overlaps that are too short.

            if overlap.length < min_length:
                continue

            # Reference ID for JavaScript highlighting.
            # This is random, but shared between corresponding fragments.

            ref = _random.randint(0, 1e10)
            color = next(colors)

            # Summary starting tag.

            summary[overlap.summary][0] = start.format(
                color = color,
                ref = ref,
                length = overlap.length,
            ) + summary[overlap.summary][0]

            # Text starting tag.

            text[overlap.text][0] = start.format(
                color = color,
                ref = ref,
                length = overlap.length,
            ) + text[overlap.text][0]

            # Summary ending tag.

            summary[overlap.summary + overlap.length - 1][0] += end

            # Text ending tag.

            text[overlap.text + overlap.length - 1][0] += end

        # Carefully join tokens and whitespace to reconstruct the string.

        summary = " ".join("".join("".join(tw) for tw in summary).split())
        text = " ".join("".join("".join(tw) for tw in text).split())

        # Return the tuple.

        return summary, text


    def _itercolors(self):

        # Endlessly cycle through these colors.

        return _itertools.cycle((

            "#393b79",
            "#5254a3",
            "#6b6ecf",
            "#9c9ede",
            "#637939",
            "#8ca252",
            "#b5cf6b",
            "#cedb9c",
            "#8c6d31",
            "#bd9e39",
            "#e7ba52",
            "#e7cb94",
            "#843c39",
            "#ad494a",
            "#d6616b",
            "#e7969c",
            "#7b4173",
            "#a55194",
            "#ce6dbd",
            "#de9ed6",

        ))

################################################################################



GRUSKY_STATS=True

def remove_dupl_whitespace(s):
    '''
    Removes duplicate whitespace every where in the string
    '''
    symbol_list=[" ", "\n", "\t"]
    for symbol in symbol_list:
        s = re.sub(symbol+'+', symbol, s)
    return s


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




    if GRUSKY_STATS:
        tqdm.pandas(desc="my bar!")
        all_stats_fn = lambda x: Fragments(x[2],x[3]).all_stats() if(x[2]!="" and x[3]!="") else ""
        coverage_stats_fn = lambda x: Fragments(x[2],x[3]).coverage()
        density_stats_fn = lambda x: Fragments(x[2],x[3]).density()
        compression_stats_fn = lambda x: Fragments(x[2],x[3]).compression()
        # stats_fn = lambda x: x[3]

        deduplicated= deduplicated.dropna(how="any",subset=['text','summary'])
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
