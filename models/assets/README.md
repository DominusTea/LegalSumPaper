#Assets  
Assets used for the creation of stopword files.
- We have extracted the greek [stopwords](stopwords_el_lexrank.txt) used in the [lexrank](graph_based/LexRank/lexrank_master) model. Those stopwords contain modern greek diacritics.
- We have also extracted the greek [stopwords](stopwords_el_spacy.txt) used in the greek spaCy model 'el_core_news_sm'.  
## stopword merging  
We remove all diacritics, apart from  "´", from the lexrank stopwords and we merge them with the spaCy stopwords into a single [file](stopwords_el.txt)
