import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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


def readDatasetCSV(path : str, sep = ",") -> pd.DataFrame:
    return pd.read_csv( path, sep=sep )


def saveDatasetPickle(df : pd.DataFrame, savePath : str):
    df.to_pickle( savePath )

def readDatasetPickle( path : str) -> pd.DataFrame:
    return pd.read_pickle( path )

# STRIP
def addColumnsStrip(df : pd.DataFrame):
    for current in ["left", "right"]:
        df [current + '_strip'] = df[current].apply(lambda x : x.strip())

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
        df [current + '_strip_tokenized_noPunct_lemmat']= df[current + '_strip_tokenized_noPunct'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])

# NO STOP WORDS (CONTENT WORDS)
def addColumnsContentWords(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip_tokenized_noPunct_lemmat_noStopWords'] = df [current + '_strip_tokenized_noPunct_lemmat'].apply(lambda x: [word for word in x if not word in stopwords.words('english')])

 # STOP WORDS
def addColumnsStopWords(df : pd.DataFrame):
    for current in ["left", "right"]:
        df[current + '_strip_tokenized_noPunct_StopWords'] = df [current + '_strip_tokenized_noPunct'].apply(lambda x: [word for word in x if word in stopwords.words('english')])


