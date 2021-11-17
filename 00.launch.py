import pandas as pd
from src.Preprocess import Utils
from src.Preprocess import Lexical_Features
from src.Preprocess import WordNet_Features
import nltk
nltk.download('stopwords')
nltk.download('punkt')

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
nrows=2
datasets = dict( {name : Utils.readDataset(path, nrows=nrows) for (name,path) in zip(names,paths)})

# Preprocess dataset

preprocess_pipeline = [
    Utils.addColumnsStrip,
    Utils.addColumnsTokenized,
    Utils.addColumnsNoPunctuations,
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

# Compute features
features_pipeline = [
    Lexical_Features.addColumnsJaccardStripTokenized,
    Lexical_Features.addColumnsJaccardContentWords,
    Lexical_Features.addColumnsJaccardStopwords,
    Lexical_Features.addColumnsLength,
    Lexical_Features.addColumnsLeftRight,
    Lexical_Features.addColumnsRightLeft
]

step=1
for name,dataset in datasets.items():
    for func in features_pipeline:
        func(dataset)
    print("Processing features dataset {}/{}".format(step, len(datasets.keys())))
    step+=1


# Compute wordnet
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
    WordNet_Features.addColumnsChunk1Maximum,
    WordNet_Features.addColumnsChunk2Maximum,
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
    print("Processing wordnet dataset {}/{}".format(step, len(datasets.keys())))
    step+=1


