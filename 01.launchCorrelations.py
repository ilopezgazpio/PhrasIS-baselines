import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from src.Preprocess import Utils
from src.Correlations import Correlations

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

saveFolder = "dirty"
if not os.path.exists(saveFolder):
    sys.exit("First preprocess the datasets... Exiting")

datasets = dict( {name : Utils.loadDatasetPickle( os.path.join(saveFolder, name+".pickle")) for name in names })

figuresFolder = "figures"
if not os.path.exists(figuresFolder):
    os.makedirs(figuresFolder)


lexical = ['jaccard_strip_tokenized', 'jaccard_strip_tokenized_noPunct_lemmat_noStopWords', 'jacckard_strip_tokenized_noPunct']
wordnet_path = ['path_similarity', 'path_similarity_root']
wordnet_lch = ['lch_similarity_nouns', 'lch_similarity_verbs', 'lch_similarity_nouns_root', 'lch_similarity_verbs_root']
wordnet_jcn = ['jcn_similarity_brown_nouns', 'jcn_similarity_brown_verbs', 'jcn_similarity_genesis_nouns', 'jcn_similarity_genesis_verbs']
wordnet_wup = ['wup_similarity', 'wup_similarity_root']
wordnet_depth = ['chunk1>chunk2', 'chunk2>chunk1', 'minimum_difference', 'maximum_difference']
length = ['left-right', 'right-left', '|chunk1-chunk2|']

all_features = lexical + wordnet_path + wordnet_lch + wordnet_jcn + wordnet_wup + wordnet_depth + length


''' Correlation Plot 1 : Color map correlation '''

titleName = "Correlation Matrix for "
titleName2= "Scatter Matrix"
figName1 = "ColorMapCorrelation_"
figName2= "SquareMapCorrelation_"
figName3= "ScatterMatrix"

for name, dataset in datasets.items():
    Correlations.CorrelationMatrix(dataset, all_features, titleName + name, fillNA=True, savePath=os.path.join(figuresFolder, figName1+name + ".png"))
    Correlations.CorrelationMatrix2(dataset, all_features, titleName + name, fillNA=True, savePath=os.path.join(figuresFolder, figName2+name + ".png"))

#Correlations.scatterMatrix(dataset, all_features, titleName2, fillNA=True, savePath=os.path.join(figuresFolder, figName3 + ".png"))




