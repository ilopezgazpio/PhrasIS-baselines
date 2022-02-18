import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer

from src.Preprocess import Utils
from src.MachineLearning import MachineLearning
from src.Constants.Constants import ALL_FEATURES
from src.Constants.Constants import LEXICAL_COLS

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

# Set seed for all libraries
np.random.seed(123)

# To print the whole df
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Step 1 -> load all datasets

datasetsFolder = "dataset/bin"
if not os.path.exists(datasetsFolder):
    sys.exit("First preprocess the datasets... Exiting")

all_names = [
    'PhrasIS_train_h_n',
    'PhrasIS_train_h_p',
    'PhrasIS_train_i_n',
    'PhrasIS_train_i_p',
    'PhrasIS_test_h_n',
    'PhrasIS_test_h_p',
    'PhrasIS_test_i_n',
    'PhrasIS_test_i_p',
]

all_datasets = dict( {name : Utils.loadDatasetPickle( os.path.join(datasetsFolder, name+".pickle")) for name in all_names })

# Step 2 -> create Bag of Word features with CountVectorizer (we need same number of features for all datasets, so we share CountVectorizer instance for all datasets)
cv = CountVectorizer()
common_strings = pd.DataFrame([], columns=['full_text'])

for name, dataset in all_datasets.items():
    common_strings = pd.concat( [common_strings, pd.DataFrame(dataset['left'] + "," + dataset['right'], columns=['full_text'])], axis=0, ignore_index=True)

# fit cv to get to know all words of all datasets
cv.fit(common_strings['full_text'])
# Check correctness with vc.vocabulary_

# Step 3 -> Process datasets, add BoW features with global countvectorizer and remove labels

# Now use trained cv to create BOW features in each dataset
target_sts = dict()
target_nli = dict()

for name, dataset in all_datasets.items():
    target_sts[name] = dataset['STS'].values
    target_nli[name] = dataset['NLI'].values

    # Remove supervised labels for training
    all_datasets[name].drop(columns = ['STS','NLI'])

    # Create dataset specific BoW features as new columns
    all_datasets[name] = np.concatenate( (cv.transform(dataset['left'] + "," + dataset['right']).toarray(), dataset[ALL_FEATURES].values), axis=1)
    # datasets are saved as numpy arrays from now on


# Step 4 - create scenarios for crossvalidation
crossValidation_datasets = list()

crossValidation_datasets.append("PhrasIS_train_h_p")
crossValidation_datasets.append("PhrasIS_train_i_p")
crossValidation_datasets.append("PhrasIS_train_h_p+PhrasIS_train_h_n")
crossValidation_datasets.append("PhrasIS_train_i_p+PhrasIS_train_i_n")
crossValidation_datasets.append("PhrasIS_train_h_p+PhrasIS_train_i_p")
crossValidation_datasets.append("PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n")

all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"] = np.concatenate( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_h_n"]) , axis=0)
all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"] = np.concatenate( (all_datasets["PhrasIS_train_i_p"], all_datasets["PhrasIS_train_i_n"]) , axis=0)
all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p"] = np.concatenate( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_i_p"]) , axis=0)
all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = np.concatenate( (all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"], all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"]), axis=0)

target_sts["PhrasIS_train_h_p+PhrasIS_train_h_n"] = np.concatenate( (target_sts["PhrasIS_train_h_p"], target_sts["PhrasIS_train_h_n"]) , axis=0)
target_sts["PhrasIS_train_i_p+PhrasIS_train_i_n"] = np.concatenate( (target_sts["PhrasIS_train_i_p"], target_sts["PhrasIS_train_i_n"]) , axis=0)
target_sts["PhrasIS_train_h_p+PhrasIS_train_i_p"] = np.concatenate( (target_sts["PhrasIS_train_h_p"], target_sts["PhrasIS_train_i_p"]) , axis=0)
target_sts["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = np.concatenate( (target_sts["PhrasIS_train_h_p+PhrasIS_train_h_n"], target_sts["PhrasIS_train_i_p+PhrasIS_train_i_n"]), axis=0)

target_nli["PhrasIS_train_h_p+PhrasIS_train_h_n"] = np.concatenate( (target_nli["PhrasIS_train_h_p"], target_nli["PhrasIS_train_h_n"]) , axis=0)
target_nli["PhrasIS_train_i_p+PhrasIS_train_i_n"] = np.concatenate( (target_nli["PhrasIS_train_i_p"], target_nli["PhrasIS_train_i_n"]) , axis=0)
target_nli["PhrasIS_train_h_p+PhrasIS_train_i_p"] = np.concatenate( (target_nli["PhrasIS_train_h_p"], target_nli["PhrasIS_train_i_p"]) , axis=0)
target_nli["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = np.concatenate( (target_nli["PhrasIS_train_h_p+PhrasIS_train_h_n"], target_nli["PhrasIS_train_i_p+PhrasIS_train_i_n"]), axis=0)

crossValidation_models = [
    tree.DecisionTreeClassifier()
]

# todo : fitxategi auxiliarreko sailkatzaileak sartu ere

# Step 5 -> Cross Validate models on NLI
classification_measures = ['accuracy', 'precision_micro', 'precision_macro','recall_micro','recall_macro','f1_micro','f1_macro']
result_names = ['test_' + name for name in classification_measures]


data = []
for dataset_name in crossValidation_datasets:
    for model in crossValidation_models:
        result = cross_validate(model, all_datasets[dataset_name], target_nli[dataset_name] , cv=5, scoring=classification_measures)
        results = [result[measure].mean() for measure in result_names]
        data.append([model, dataset_name] + results)


table_results_crossValidation = pd.DataFrame(data, columns = ["Model name", "CV Set"] + result_names)
print(table_results_crossValidation)


# Step 6 -> Cross Validate models on STS

# todo
# Bileran komentatzeko -> STS regresioa da eta ezin dira NLI (labelak) kasuko ebaluazio metrikak erabili
# ez dakit scikit-learn en zer dagoen, baino pearson, spearman edo R karratu moduko zerbait behar dugu, regresioa ebaluatzeko metrikak



# Step 7 -> Final training and evaluation on test set

# todo

# Hemen cross validation en erabili ditugun datu base berberak hartu behar dira, baina train bakarrik egiteko, ebaluatu gabe.
# ebaluatzeko bere test bersioa sortu beharko dugu

