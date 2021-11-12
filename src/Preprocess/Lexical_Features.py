##Import
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np


"""
PYTHON MODULE TO COMPUTE LEXICAL FEATURES 
"""



#def addColumnsXXX(df : pd.DataFrame):
#    for current in ["left", "right"]:
#        df [current + '_strip'] = df[current].apply(lambda x : x.XXX())

#
# ###Jaccard
#
# ##FEATURE 1: of strip_tokenized
# df ['jacckard_strip_tokenized'] = df.apply(lambda it : m.jaccard_similarity (it['left_strip_tokenized'],
#                 it['right_strip_tokenized']), axis = 1)
#
# ##FEATURE 2: strip_tokenized_noPunct_noStopWords = content_words
# df ['jacckard_strip_tokenized_noPunct_lemmat_noStopWords'] = df.apply(lambda it : m.jaccard_similarity (it['left_strip_tokenized_noPunct_lemmat_noStopWords'],
#                 it['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis = 1)
#
# ###FEATURE 3: strip_tokenized_noPunct
# df ['jacckard_strip_tokenized_noPunct'] = df.apply(lambda it : m.jaccard_similarity (it['left_strip_tokenized_noPunct'],
#                 it['right_strip_tokenized_noPunct']), axis = 1)
#
# ##Difference in length
# ###FEATURE 4 AND 5: left-right and right-left
#
# df ['left_strip_tokenized_len']=df.left_strip_tokenized.apply(len)
# df ['right_strip_tokenized_len']=df.right_strip_tokenized.apply(len)
#
# df['left-right'] = df.apply(lambda x: m.substract(x['left_strip_tokenized_len'], x['right_strip_tokenized_len']), axis = 1)
# df['right-left'] = df.apply(lambda x: m.substract(x['right_strip_tokenized_len'],x['left_strip_tokenized_len']), axis = 1)
#
# print (df)
#


