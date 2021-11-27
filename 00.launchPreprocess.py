import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet_ic')
nltk.download('genesis')
nltk.download('averaged_perceptron_tagger')


from src.Preprocess import Utils
from src.Preprocess import Lexical_Features
from src.Preprocess import WordNet_Features

# Set seed for all libraries
np.random.seed(123)

# To print the whole df
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Load the datasets
names = [
    'PhrasIS_test_h_n',
    'PhrasIS_test_h_p',
    'PhrasIS_test_i_n',
    'PhrasIS_test_i_p',
    'PhrasIS_train_h_n',
    'PhrasIS_train_h_p',
    'PhrasIS_train_i_n',
    'PhrasIS_train_i_p'
]

paths = [
    'dataset/PhrasIS.test.headlines.negatives.txt',
    'dataset/PhrasIS.test.headlines.positives.txt',
    'dataset/PhrasIS.test.images.negatives.txt',
    'dataset/PhrasIS.test.images.positives.txt',
    'dataset/PhrasIS.train.headlines.negatives.txt',
    'dataset/PhrasIS.train.headlines.positives.txt',
    'dataset/PhrasIS.train.images.negatives.txt',
    'dataset/PhrasIS.train.images.positives.txt',
]

# For development only
nrows=20
datasets = dict( {name : Utils.readDataset(path, nrows=nrows) for (name,path) in zip(names,paths)})

# Preprocess dataset
preprocess_pipeline = [
    Utils.addColumnsLower,
    Utils.addColumnsStrip,
    Utils.addColumnsTokenized,
    Utils.addColumnsNoPunctuations,
    Utils.addColumnsPOStags,
    Utils.addColumnsLemmatized,
    Utils.addColumnsContentWords,
    Utils.addColumnsStopWords
]

step=1
for name,dataset in datasets.items():
    for func in preprocess_pipeline:
        func(dataset)
    print("Processing dataset {}/{}".format(step, len(datasets.keys())))
    step+=1

# Compute lexical features
lexical_pipeline = [
    Lexical_Features.addColumnsJaccardStripTokenized,
    Lexical_Features.addColumnsJaccardContentWords,
    Lexical_Features.addColumnsJaccardStopwords,
    Lexical_Features.addColumnsLength,
    Lexical_Features.addColumnsLeftRight,
    Lexical_Features.addColumnsRightLeft
]

step=1
for name,dataset in datasets.items():
    for func in lexical_pipeline:
        func(dataset)
    print("Processing lexical features {}/{}".format(step, len(datasets.keys())))
    step+=1


# Compute wordnet features
wordnet_pipeline = [
    WordNet_Features.addColumnsPathSimilarity,
    WordNet_Features.addColumnsLchSimilarityNouns,
    WordNet_Features.addColumnsLchSimilarityVerbs,
    WordNet_Features.addColumnsJcnSimilarityBrownNouns,
    WordNet_Features.addColumnsJcnSimilarityBrownVerbs,
    WordNet_Features.addColumnsJcnSimilarityGenesisNouns,
    WordNet_Features.addColumnsJcnSimilarityGenesisVerbs,
    WordNet_Features.addColumnsWupSimilarity,
    WordNet_Features.addColumnsPathSimilarityRoot,
    WordNet_Features.addColumnsLchSimilarityNounsRoot,
    WordNet_Features.addColumnsLchSimilarityVerbsRoot,
    WordNet_Features.addColumnsWupSimilarityRoot,
    WordNet_Features.addColumnsChunkMaximum,
    WordNet_Features.addColumnsChunk1Specific,
    WordNet_Features.addColumnsChunk2Specific,
    WordNet_Features.addColumnsDifference,
    WordNet_Features.addColumnsMinimumDifference,
    WordNet_Features.addColumnsMaximumDifference
]

step=1
for name,dataset in datasets.items():
    for func in wordnet_pipeline:
        func(dataset)
    print("Processing wordnet features {}/{}".format(step, len(datasets.keys())))
    step+=1


# TODO -> NORMALIZATU


# Save files
saveFolder = "dirty"
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

for name, df in datasets.items():
    Utils.saveDatasetCSV(df, os.path.join( saveFolder, name + ".csv"))
    Utils.saveDatasetPickle(df, os.path.join( saveFolder, name + ".pickle"))





