import os

from accents_mapping import unaccentify #removes every diacretic execept regular modern greek tonation

stopwords_file_1 = "../assets/stopwords_el_lexrank.txt"
stopwords_file_2 = "../assets/stopwords_el_spacy.txt"
outputfile = "../assets/stopwords_el.txt"

# get set of stopwords from each file
s1 = set()
s2 = set()
with open(stopwords_file_1, 'r') as f:
    for line in f:
        s1.add(unaccentify(line))
with open(stopwords_file_2, 'r') as f:
    for line in f:
        s2.add(line)

# print((s2))
# get set union
s = s1.union(s2)

# write them to file assets/stopwords_el.txt
with open(outputfile, 'w+', encoding='utf-8') as f:
    for stopword in sorted(s):
        f.write(str(stopword))

# assemble them to lexrank model
# this will work on the pip installed lexrank.
# if lexrank wasn't installed using pip install -e /path/to/our/lexrank/fork
# then import lexrank will not load our lexrank fork, but will have our lexrank stopwords
os.system('lexrank_assemble_stopwords --source_dir '+outputfile)
