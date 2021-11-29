import pandas as pd
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

from src.Constants.Constants import ALL_FEATURES

''' Correlation Plot 1 & 2 : Color map correlation '''

titleName = "Correlation Matrix for "
figName1 = "ColorMapCorrelation_"
figName2= "SquareMapCorrelation_"
figName3= "ScatterMatrix_"

for name, dataset in datasets.items():
    Correlations.CorrelationMatrix(dataset, ALL_FEATURES, titleName + name, fillNA=True, savePath=os.path.join(figuresFolder, figName1+name + ".png"))
    Correlations.CorrelationMatrix2(dataset, ALL_FEATURES, titleName + name, fillNA=True, savePath=os.path.join(figuresFolder, figName2+name + ".png"))



''' Correlation Plot 3: Scatter '''
titleName2= "Scatter Matrix for "

from src.Constants.Constants import LEXICAL_COLS
from src.Constants.Constants import WORDNET_PATH_COLS
from src.Constants.Constants import WORDNET_LCH_COLS
from src.Constants.Constants import WORDNET_JCN_COLS
from src.Constants.Constants import WORDNET_WUP_COLS
from src.Constants.Constants import WORDNET_DEPTH_COLS
from src.Constants.Constants import LENGTH_COLS


intra_groups = {
    "LEXICAL features" : LEXICAL_COLS ,
    "WORDNET PATH features" : WORDNET_PATH_COLS,
    "WORDNET LCH features" : WORDNET_LCH_COLS,
    "WORDNET JCN features" : WORDNET_JCN_COLS,
    "WORDNET WUP features" : WORDNET_WUP_COLS,
    "WORDNET DEPTH features" : WORDNET_DEPTH_COLS,
    "LENGTH features" : LENGTH_COLS
}

for name, col_group in intra_groups.items():
    Correlations.scatterMatrix(dataset, col_group, titleName2 + name, fillNA=True, savePath=os.path.join(figuresFolder, figName3+name.lower().replace(" ","_")+".png"))


'''
Compute some inter-group correlations, to see correlation among distinct kind of features
'''
inter_group1 = [LEXICAL_COLS[0], WORDNET_PATH_COLS[0], WORDNET_LCH_COLS[0], WORDNET_JCN_COLS[0], WORDNET_WUP_COLS[0], WORDNET_DEPTH_COLS[0], LENGTH_COLS[0]]
Correlations.scatterMatrix(dataset, inter_group1, titleName2 + "inter group features", fillNA=True, savePath=os.path.join(figuresFolder, figName3+"inter1"+".png"))


