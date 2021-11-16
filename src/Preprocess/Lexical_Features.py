import pandas as pd

"""
PYTHON MODULE TO COMPUTE LEXICAL FEATURES 
"""

# JACCARD OVERLAP STRIP TOKENIZED
def addColumnsJaccardStripTokenized(df: pd.DataFrame):
    df ['jaccard_strip_tokenized'] = df.apply(lambda it : jaccard_similarity (it['left_strip_tokenized'],
                 it['right_strip_tokenized']), axis = 1)

# JACCARD OVERLAP OF CONTENT WORDS
def addColumnsJaccardContentWords (df: pd.DataFrame):
    df ['jaccard_strip_tokenized_noPunct_lemmat_noStopWords'] =df.apply(lambda it : jaccard_similarity (it['left_strip_tokenized_noPunct_lemmat_noStopWords'],
                 it['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis = 1)

# JACCKARD OVERLAP OF STOPWORDS
def addColumnsJaccardStopwords (df: pd.DataFrame):
    df ['jacckard_strip_tokenized_noPunct']=df.apply (lambda it : jaccard_similarity (it['left_strip_tokenized_noPunct'],
                 it['right_strip_tokenized_noPunct']), axis = 1)

# LENGTH
def addColumnsLength (df: pd.DataFrame):
    for current in ["left", "right"]:
        df [current + '_strip_tokenized_len'] = df [current + '_strip_tokenized'].apply(len)

# DIFFERENCE LEFT-RIGHT
def addColumnsLeftRight (df: pd.DataFrame):
    df ['left-right']= df.apply(lambda x: substract(x['left_strip_tokenized_len'], x['right_strip_tokenized_len']), axis = 1)

# DIFFERENCE RIGHT-LEFT
def addColumnsRightLeft (df: pd.DataFrame):
    df ['right-left']=df.apply(lambda x: substract(x['right_strip_tokenized_len'],x['left_strip_tokenized_len']), axis = 1)

def substract(a, b):
    return a - b

def jaccard_similarity (list1, list2):
    s1 = set(list1)
    s2 = set(list2)

    try:
        respuesta=float(len(s1.intersection(s2)) / len(s1.union(s2)))
    except:
        respuesta=float (0)

    return respuesta


