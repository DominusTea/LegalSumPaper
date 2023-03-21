import regex
from urlextract import URLExtract

import spacy
from spacy.symbols import ORTH, NORM

EMAIL_REGEX = regex.compile(
    r'[\p{L}0-9]+[\p{L}0-9_.+-]*[\p{L}0-9_+-]+@[\p{L}0-9]+[\p{L}0-9.-]*\.\p{L}+'  # noqa
)
PUNCTUATION_SIGNS = set('.,;:¡!¿?…⋯&‹›«»\"“”[]()⟨⟩}{/|\\')

url_extractor = URLExtract()


def clean_text(text, allowed_chars='- '):
    text = ' '.join(text.lower().split())
    text = ''.join(ch for ch in text if ch.isalnum() or ch in allowed_chars)

    return text


def contains_letters(word):
    return any(ch.isalpha() for ch in word)


def contains_numbers(word):
    return any(ch.isdigit() for ch in word)


def filter_words(words, stopwords, keep_numbers=False):
    if keep_numbers:
        words = [
            word for word in words
            if (contains_letters(word) or contains_numbers(word)) and
            word not in stopwords
        ]

    else:
        words = [
            word for word in words
            if contains_letters(word) and not contains_numbers(word) and
            word not in stopwords
        ]

    return words


def separate_punctuation(text):
    text_punctuation = set(text) & PUNCTUATION_SIGNS

    for ch in text_punctuation:
        text = text.replace(ch, ' ' + ch + ' ')

    return text


def tokenize(
    text,
    stopwords,
    keep_numbers=False,
    keep_emails=False,
    keep_urls=False,
):
    tokens = []
    # print("got text: ", text)
    for word in text.split():
        emails = EMAIL_REGEX.findall(word)

        if emails:
            if keep_emails:
                tokens.append(emails[0])

            continue

        urls = url_extractor.find_urls(word, only_unique=True)

        if urls:
            if keep_urls:
                tokens.append(urls[0].lower())

            continue

        cleaned = clean_text(separate_punctuation(word)).split()
        cleaned = filter_words(cleaned, stopwords, keep_numbers=keep_numbers)

        tokens.extend(cleaned)

    return tokens

def new_tokenize(
    text,
    stopwords,
    keep_numbers=False,
    keep_emails=False,
    keep_urls=False,
    nlp = spacy.load('el_core_news_sm')
):
    tokens = []

    doc = nlp(text)
    # print("sentence length", len(doc))
    # print(text)
    for word in doc:
        if word.pos_ in ['ADJ', 'NOUN', 'PROPN', 'VERB'] and word.text not in stopwords:
            tokens.append(word.lemma_.lower().strip())

    return tokens


def adapt_spacy(nlp):
    '''
    Adapts spacy pipeline to AreiosPagos caselaw dataset
    Should, probably, be moved elsewere after testing
    '''
    # legal text exceptions
    nlp.tokenizer.add_special_case('παρ.', [{ORTH: 'παρ.', NORM:'παράγραφος'  }])
    nlp.tokenizer.add_special_case('παρ. ', [{ORTH: 'παρ. ', NORM:'παράγραφος'  }])



    nlp.tokenizer.add_special_case('Π.Κ.', [{ORTH: 'Π.', NORM: 'ποινικός'},
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
                                   [{ORTH: 'Μον. ' , NORM:'μονομελές'  },
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
