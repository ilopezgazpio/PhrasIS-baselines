#coding: utf-8
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .WordNet_Features import getWnPos

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

"""
PREPROCESSING UTILITIES 
"""

lemmatizer = WordNetLemmatizer()


def readDataset(path : str, nrows=None):
    df = pd.read_csv(path, delimiter="\t", header=None, nrows=nrows)
    df.columns = ['STS', 'NLI', 'left', 'right', 'left_num', 'right_num', 'id']
    return df


def saveDatasetCSV(df : pd.DataFrame, savePath : str, sep = ","):
    df.to_csv( savePath, sep=sep )


def loadDatasetCSV(path : str, sep = ",") -> pd.DataFrame:
    return pd.read_csv( path, sep=sep )


def saveDatasetPickle(df : pd.DataFrame, savePath : str):
    df.to_pickle( savePath )

def loadDatasetPickle( path : str) -> pd.DataFrame:
    return pd.read_pickle( path )

#LOWER AND PUNCTUATION ISSUES
def addColumnsLower (df: pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_lower'] = df[current].apply(lambda x: x.lower())
        df[current+ '_lower'] = df[current+ '_lower'].apply(lambda x: x.replace("-", " "))
        df[current+ '_lower'] = df[current+ '_lower'].apply(lambda x: x.replace(".", " "))

# STRIP
def addColumnsStrip(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip']= df[current+ '_lower'].apply(lambda x : x.strip())

# TOKENIZED
def addColumnsTokenized(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip_tokenized'] = df.apply(lambda x: nltk.word_tokenize(x[current + '_strip']), axis=1)

# NO PUNCTUATIONS
def addColumnsNoPunctuations(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip_tokenized_noPunct'] = df [current + '_strip_tokenized'].apply(lambda x: [word for word in x if word.isalnum()])

# LEMMATIZED
def addColumnsLemmatized(df : pd.DataFrame):
    for current in ["left", "right"]:
        # We also need POS column to lemmatize correctly
        # df [current + '_strip_tokenized_noPunct_lemmat']= df[current + '_strip_tokenized_noPunct'].apply(lambda x: [lemmatizer.lemmatize(w, getWnPos(w)) for w in x])
        column1 = current + '_strip_tokenized_noPunct'
        column2 = current + '_POS_tags'
        df[current + '_strip_tokenized_noPunct_lemmat']= df.apply(lambda x: [ lemmatizer.lemmatize(token, getWnPos(pos_tag)) for (token, pos_tag) in zip(x[column1], x[column2])], axis=1)

# NO STOP WORDS (CONTENT WORDS)
def addColumnsContentWords(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip_tokenized_noPunct_lemmat_noStopWords'] = df [current + '_strip_tokenized_noPunct_lemmat'].apply(lambda x: [word for word in x if not word in stopwords.words('english')])

 # STOP WORDS
def addColumnsStopWords(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip_tokenized_noPunct_StopWords'] = df [current + '_strip_tokenized_noPunct'].apply(lambda x: [word for word in x if word in stopwords.words('english')])


# POS TAGS
def addColumnsPOStags(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_POS_tags'] = df [current + '_strip_tokenized_noPunct'].apply(lambda tokens: list(   zip(*nltk.pos_tag(tokens)))[1] if len(tokens) else [] )

# nltk.pos_tag(["I", " dog"])
# [('I', 'PRP'), (' dog', 'VBP')]
# list(zip(*nltk.pos_tag(["I", " dog"])))
# [('I', ' dog'), ('PRP', 'VBP')]

def normalization (df: pd.DataFrame):

    drop= ['STS', 'NLI', 'left', 'right', 'left_num', 'right_num', 'id', 'left_lower', 'right_lower',
           'left_strip', 'right_strip', 'left_strip_tokenized', 'right_strip_tokenized', 'left_strip_tokenized_noPunct',
           'right_strip_tokenized_noPunct', 'left_POS_tags', 'right_POS_tags', 'left_strip_tokenized_noPunct_lemmat',
           'right_strip_tokenized_noPunct_lemmat', 'left_strip_tokenized_noPunct_lemmat_noStopWords', 'right_strip_tokenized_noPunct_lemmat_noStopWords',
           'left_strip_tokenized_noPunct_StopWords', 'right_strip_tokenized_noPunct_StopWords',
           'chunk1_maximum', 'chunk2_maximum', 'chunk1>chunk2', 'chunk2>chunk1']

    X = df.drop(drop, axis=1, inplace=False).values
    X = preprocessing.StandardScaler().fit_transform(X.astype(float))

    normalize_data = pd.DataFrame(X, columns=['jaccard_strip_tokenized', 'jaccard_strip_tokenized_noPunct_lemmat_noStopWords', 'jacckard_strip_tokenized_noPunct',
                                              'left_strip_tokenized_len', 'right_strip_tokenized_len','left-right', 'right-left', 'path_similarity', 'lch_similarity_nouns', 'lch_similarity_verbs',
                                              'jcn_similarity_brown_nouns', 'jcn_similarity_brown_verbs', 'jcn_similarity_genesis_nouns', 'jcn_similarity_genesis_verbs', 'wup_similarity',
                                              'path_similarity_root', 'lch_similarity_nouns_root', 'lch_similarity_verbs_root', 'wup_similarity_root',
                                              '|chunk1-chunk2|', 'minimum_difference', 'maximum_difference'])

    df.update(normalize_data)

