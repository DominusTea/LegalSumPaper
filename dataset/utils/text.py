import spacy
from spacy.symbols import ORTH, NORM
import re

def remove_text_relative_to_substring(fulltext, substring, remove_text='before'):
    '''
    Removes text before/after the occurance of a substring
    '''
    split_text = fulltext.split(substring)
    N_occurs = len(split_text)
    # print(N_occurs)
    if remove_text=='before':
        if N_occurs == 1:
            # print(split_text[0][0:9000])
            # print("~~~``")
            return split_text[0]
        elif N_occurs== 2:
            return split_text[1].strip()
        else:
            return ("".join(split_text[1:])).strip()
    elif remove_text =='after':
        if N_occurs == 1:
            return split_text[0]
        elif N_occurs == 2:
            return split_text[0].strip()
        else:
            return ("".join(split_text[0:N_occurs-1])).strip()
    else:
        raise ValueError("remove_text must be one of ['before'|'after']")

def remove_text_relative_to_strings(fulltext, strings, remove_text='before'):
    '''
    Like remove_text_relative_to_substring but strings can be a list of potential (substngs, remove_text)
    '''
    if isinstance(strings, str):
        return remove_text_relative_to_substring(fulltext, strings, remove_text)
    else:
        processed = fulltext
        for substring, remove_text_pos in strings:
            processed = remove_text_relative_to_substring(processed,
                                                          substring=substring,
                                                          remove_text=remove_text_pos)
        return processed


def remove_AreiosPagos_start():
    starting_options = [("ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ", 'before'),
                    ("ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟΝ ΝΟΜΟ", 'before'),
                    ("ΣΚΕΦΤΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ", 'before'),
                    ("Σκέφθηκε σύμφωνα με το νόμο", 'before'),
                    ("Σκέφθηκε σύμφωνα με στείλουν νόμο", 'before'),
                    ("Σ Κ Ε Φ Θ Η Κ Ε  Σ Υ Μ Φ Ω Ν Α  Μ Ε  Τ Ο  Ν Ο Μ Ο", "before")
                    ]

    return starting_options

def removeAreiosPagos_end():
    ending_options= [("Κρίθηκε και αποφασίσθηκε", 'after'),
                 ("ΚΡΙΘΗΚΕ, αποφασίσθηκε", 'after'),
                 ("ΚΡΙΘΗΚΕ και αποφασίσθηκε", 'after'),
                 ]
    return ending_options

def AreiosPagosClearingFunc():
    substring_options = remove_AreiosPagos_start() + removeAreiosPagos_end()
    return lambda x : remove_text_relative_to_strings(fulltext=x,
                                                      strings=substring_options,
                                                      )
def removeSTE_substring():
    res="(Αριθμός *(\d)+/(\d)+ -\d-)|(\r()*\n)"
    return res

def removeSTE_start():
    starting_options=[("Σκεφθέν κατά τον νόμον", 'before'),
                        ('Α φ ο ύ\xa0μ ε λ έ τ η σ ε\xa0τ α\xa0σ χ ε τ ι κ ά\xa0έ γ γ ρ α φ α\n\nΣ κ έ φ θ η κ ε\xa0κ α τ ά\xa0τ ο ν\xa0Ν ό μ ο', 'before'),
                        ('Α φ ο ύ μ ε λ έ τ η σ ε τ α σ χ ε τ ι κ ά έ γ γ ρ α φ α\n\nΣ κ έ φ θ η κ ε κ α τ ά τ ο ν Ν ό μ ο', 'before'),
                        ('Α φ ο ύ μ ε λ έ τ η σ ε τ α σ χ ε τ ι κ ά έ γ γ ρ α φ α \nΣ κ έ φ θ η κ ε κ α τ ά τ ο ν ν ό μ ο \n', 'before'),    
    ]
    return starting_options

def removeSTE_end():
    ending_options =[
                    ("Η διάσκεψη έγινε", 'after'),
                     ("Η διάσκεψη   \nέγινε", 'after'),
                     ("Κρίθηκε και αποφασίσθηκε στην Αθήνα", 'after'),
                     ("Εκρίθη και απεφασίσθη εν Αθήναις", 'after'),
                     ("και η απόφαση δημοσιεύθηκε σε δημόσια", 'after'),

                     ]
    return ending_options

def STEClearingFunc():
    substring_options = removeSTE_start() + removeSTE_end()
    return lambda x : remove_text_relative_to_strings(fulltext=x,
                                                      strings=substring_options,
                                                      )

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

def remove_dupl_whitespace(s):
    '''
    Removes duplicate whitespace every where in the string
    '''
    symbol_list=[" ", "\n", "\t"]
    for symbol in symbol_list:
        s = re.sub(symbol+'+', symbol, s)
    return s

def count_tokens(s,nlp):
    # cleared_text = remove_dupl_whitespace(s)
    try:
        # print("!")
        return len(nlp(s))
    except:
        return 0

def count_sentences(s, nlp):
    # cleared_text=remove_dupl_whitespace(s)
    sent_lst= list(d.text for d in nlp(s).sents)
    return len(sent_lst)

def count_tokens_per_sentence(s, nlp):
    # cleared_text=remove_dupl_whitespace(s)
    count_per_sent_lst= list( len(d) for d in nlp(s).sents)
    return sum(count_per_sent_lst)/len(count_per_sent_lst)

def do_all(s, nlp, do_clear=False):
    t = nlp(s)
    s_list = list(len(d) for d in t)
    return len(t), len(s_list), sum(s_list)/len(s_list)
