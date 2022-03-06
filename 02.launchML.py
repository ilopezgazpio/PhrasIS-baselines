import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer

from src.Preprocess import Utils
from src.Constants.Constants import ALL_FEATURES
from src.Constants.Constants import LEXICAL_COLS
from src.ML import MachineLearning

from sklearn.model_selection import cross_validate, KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

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
# Check correctness with cv.vocabulary_

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
test_datasets=list()

crossValidation_datasets.append("PhrasIS_train_h_p")
crossValidation_datasets.append("PhrasIS_train_i_p")
crossValidation_datasets.append("PhrasIS_train_h_p+PhrasIS_train_h_n")
crossValidation_datasets.append("PhrasIS_train_i_p+PhrasIS_train_i_n")
crossValidation_datasets.append("PhrasIS_train_h_p+PhrasIS_train_i_p")
crossValidation_datasets.append("PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n")

test_datasets.append("PhrasIS_test_h_p")
test_datasets.append("PhrasIS_test_i_p")
test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_h_n")
test_datasets.append("PhrasIS_test_i_p+PhrasIS_test_i_n")
test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_i_p")
test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n")

all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"] = np.concatenate( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_h_n"]) , axis=0)
all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"] = np.concatenate( (all_datasets["PhrasIS_train_i_p"], all_datasets["PhrasIS_train_i_n"]) , axis=0)
all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p"] = np.concatenate( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_i_p"]) , axis=0)
all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = np.concatenate( (all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"], all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"]), axis=0)

all_datasets["PhrasIS_test_h_p+PhrasIS_test_h_n"] = np.concatenate( (all_datasets["PhrasIS_test_h_p"], all_datasets["PhrasIS_test_h_n"]) , axis=0)
all_datasets["PhrasIS_test_i_p+PhrasIS_test_i_n"] = np.concatenate( (all_datasets["PhrasIS_test_i_p"], all_datasets["PhrasIS_test_i_n"]) , axis=0)
all_datasets["PhrasIS_test_h_p+PhrasIS_test_i_p"] = np.concatenate( (all_datasets["PhrasIS_test_h_p"], all_datasets["PhrasIS_test_i_p"]) , axis=0)
all_datasets["PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n"] = np.concatenate( (all_datasets["PhrasIS_test_h_p+PhrasIS_test_h_n"], all_datasets["PhrasIS_test_i_p+PhrasIS_test_i_n"]), axis=0)

target_sts["PhrasIS_train_h_p+PhrasIS_train_h_n"] = np.concatenate( (target_sts["PhrasIS_train_h_p"], target_sts["PhrasIS_train_h_n"]) , axis=0)
target_sts["PhrasIS_train_i_p+PhrasIS_train_i_n"] = np.concatenate( (target_sts["PhrasIS_train_i_p"], target_sts["PhrasIS_train_i_n"]) , axis=0)
target_sts["PhrasIS_train_h_p+PhrasIS_train_i_p"] = np.concatenate( (target_sts["PhrasIS_train_h_p"], target_sts["PhrasIS_train_i_p"]) , axis=0)
target_sts["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = np.concatenate( (target_sts["PhrasIS_train_h_p+PhrasIS_train_h_n"], target_sts["PhrasIS_train_i_p+PhrasIS_train_i_n"]), axis=0)

target_sts["PhrasIS_test_h_p+PhrasIS_test_h_n"] = np.concatenate( (target_sts["PhrasIS_test_h_p"], target_sts["PhrasIS_test_h_n"]) , axis=0)
target_sts["PhrasIS_test_i_p+PhrasIS_test_i_n"] = np.concatenate( (target_sts["PhrasIS_test_i_p"], target_sts["PhrasIS_test_i_n"]) , axis=0)
target_sts["PhrasIS_test_h_p+PhrasIS_test_i_p"] = np.concatenate( (target_sts["PhrasIS_test_h_p"], target_sts["PhrasIS_test_i_p"]) , axis=0)
target_sts["PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n"] = np.concatenate( (target_sts["PhrasIS_test_h_p+PhrasIS_test_h_n"], target_sts["PhrasIS_test_i_p+PhrasIS_test_i_n"]), axis=0)

target_nli["PhrasIS_train_h_p+PhrasIS_train_h_n"] = np.concatenate( (target_nli["PhrasIS_train_h_p"], target_nli["PhrasIS_train_h_n"]) , axis=0)
target_nli["PhrasIS_train_i_p+PhrasIS_train_i_n"] = np.concatenate( (target_nli["PhrasIS_train_i_p"], target_nli["PhrasIS_train_i_n"]) , axis=0)
target_nli["PhrasIS_train_h_p+PhrasIS_train_i_p"] = np.concatenate( (target_nli["PhrasIS_train_h_p"], target_nli["PhrasIS_train_i_p"]) , axis=0)
target_nli["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = np.concatenate( (target_nli["PhrasIS_train_h_p+PhrasIS_train_h_n"], target_nli["PhrasIS_train_i_p+PhrasIS_train_i_n"]), axis=0)

target_nli["PhrasIS_test_h_p+PhrasIS_test_h_n"] = np.concatenate( (target_nli["PhrasIS_test_h_p"], target_nli["PhrasIS_test_h_n"]) , axis=0)
target_nli["PhrasIS_test_i_p+PhrasIS_test_i_n"] = np.concatenate( (target_nli["PhrasIS_test_i_p"], target_nli["PhrasIS_test_i_n"]) , axis=0)
target_nli["PhrasIS_test_h_p+PhrasIS_test_i_p"] = np.concatenate( (target_nli["PhrasIS_test_h_p"], target_nli["PhrasIS_test_i_p"]) , axis=0)
target_nli["PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n"] = np.concatenate( (target_nli["PhrasIS_test_h_p+PhrasIS_test_h_n"], target_nli["PhrasIS_test_i_p+PhrasIS_test_i_n"]), axis=0)

models_nli = [
    tree.DecisionTreeClassifier(),
    KNeighborsClassifier(n_neighbors=4),
    LogisticRegression(solver='saga'),
    svm.SVC(kernel='linear'),
    GaussianNB(), #bad results
    RandomForestClassifier()
]

models_sts = [
    tree.DecisionTreeClassifier(),
    KNeighborsClassifier(n_neighbors=4),
    LogisticRegression(solver='saga'),
    svm.SVC(kernel='linear'),
    RandomForestClassifier()
]

#kfold and grid search
kfold = KFold(n_splits=5)
#grid1=MachineLearning.dtree_grid_search(all_datasets["PhrasIS_train_h_p"], target_nli ["PhrasIS_train_h_p"], kfold)
#print (grid1)
#grid2=MachineLearning.dtree_grid_search(all_datasets["PhrasIS_train_i_p"], target_nli ["PhrasIS_train_i_p"], kfold)
#print (grid2)


# Step 5 -> Cross Validate models on NLI
classification_measures_nli = ['accuracy', 'precision_micro', 'precision_macro','recall_micro','recall_macro','f1_micro','f1_macro']
result_names_nli = ['test_' + name for name in classification_measures_nli]

data_nli = []
for dataset_name in crossValidation_datasets:
    for model in models_nli:
        result_nli = cross_validate(model, all_datasets[dataset_name], target_nli[dataset_name] , cv=kfold, scoring=classification_measures_nli)
        results_nli = [result_nli[measure].mean() for measure in result_names_nli]
        data_nli.append([model, dataset_name] + results_nli)

table_results_crossValidation_nli = pd.DataFrame(data_nli, columns = ["Model name", "CV Set"] + result_names_nli)
print ("Table of results NLI:")
print(table_results_crossValidation_nli)

from scipy import stats

# Step 6 -> Cross Validate models on STS
classification_measures_sts = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error','neg_mean_absolute_percentage_error'] #the best value 0
result_names_sts = ['test_' + name for name in classification_measures_sts]

result_name_pearson=result_names_sts.copy()
result_name_pearson.append('test_pearson')

data_sts = []
for dataset_name in crossValidation_datasets:
    for model in models_sts:
        #correlation
        target_sts_array=target_sts[dataset_name]
        model.fit(all_datasets[dataset_name], target_sts[dataset_name])
        y_test = model.predict(all_datasets[dataset_name])
        result_pearson=stats.pearsonr (target_sts_array, y_test)[0]

        result_sts = cross_validate(model, all_datasets[dataset_name], target_sts[dataset_name], cv=5, scoring=classification_measures_sts)
        results_sts = [result_sts[measure].mean() for measure in result_names_sts]

        results_sts.append(result_pearson)
        data_sts.append([model, dataset_name] + results_sts)

table_results_crossValidation_sts = pd.DataFrame(data_sts, columns = ["Model name", "CV Set"] + result_name_pearson) #result_name_pearson #result_names_sts
print ("Table of results STS:")
print(table_results_crossValidation_sts)

#confussion matrix
figuresFolder = "figures_conf_matrix_nli"
if not os.path.exists(figuresFolder):
    os.makedirs(figuresFolder)
figName="Conf_matrix_"

# Step 7 -> Final training and evaluation on test set
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

test_nli = []
for model in models_nli:
    for dataset_name in crossValidation_datasets:
        model.fit (all_datasets[dataset_name], target_nli [dataset_name])

    for dataset_name_test in test_datasets:
        y_test=model.predict(all_datasets[dataset_name_test])

        y_test_accuracy=accuracy_score(target_nli[dataset_name_test], y_test)
        y_test_precision_micro = precision_score(target_nli[dataset_name_test], y_test, average='micro')
        y_test_precision_macro = precision_score(target_nli[dataset_name_test], y_test, average='macro')
        y_test_recall_micro = recall_score(target_nli[dataset_name_test], y_test, average='micro')
        y_test_recall_macro = recall_score(target_nli[dataset_name_test], y_test, average='macro')
        y_test_f1_micro = f1_score(target_nli[dataset_name_test], y_test, average='micro')
        y_test_f1_macro = f1_score(target_nli[dataset_name_test], y_test, average='macro')

        result_test_nli=[y_test_accuracy,y_test_precision_micro,y_test_precision_macro,y_test_recall_micro,y_test_recall_macro,y_test_f1_micro,y_test_f1_macro]
        test_nli.append([model, dataset_name_test]+ result_test_nli)

        #confussion matrix
        actual_classes, predicted_classes, _ = MachineLearning.cross_val_predict(model, kfold, all_datasets[dataset_name_test],target_nli[dataset_name_test])
        MachineLearning.plot_confusion_matrix(actual_classes, predicted_classes,["UNR", "EQUI", "BACK", "FORW", "SIMI", "REL", "OPPO"], dataset_name_test, model, savePath=os.path.join(figuresFolder, figName + dataset_name_test + str(model) + ".png"))

table_results_test_nli = pd.DataFrame(test_nli, columns = ["Model name", "CV Set"] + result_names_nli)
print ("Table of results test NLI:")
print(table_results_test_nli)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error

figuresFolder_sts = "figures_conf_matrix_sts"
if not os.path.exists(figuresFolder_sts):
    os.makedirs(figuresFolder_sts)

test_sts=[]
for model in models_sts:
    for dataset_name in crossValidation_datasets:
        model.fit (all_datasets[dataset_name], target_sts[dataset_name])

    for dataset_name_test in test_datasets:
        y_test=model.predict(all_datasets[dataset_name_test])

        y_test_mean_abs_error=mean_absolute_error(target_sts[dataset_name_test], y_test)
        y_test_mean_sq_error= mean_squared_error (target_sts[dataset_name_test], y_test)
        y_test_mean_sq_log_error= mean_squared_log_error (target_sts[dataset_name_test], y_test)
        y_test_mean_abs_perc_error= mean_absolute_percentage_error (target_sts[dataset_name_test], y_test)

        result_test_sts=[y_test_mean_abs_error, y_test_mean_sq_error, y_test_mean_sq_log_error, y_test_mean_abs_perc_error]

        #correlation
        result_pearson = stats.pearsonr(target_sts[dataset_name_test], y_test)[0]

        result_test_sts.append(result_pearson)
        test_sts.append([model, dataset_name_test]+result_test_sts)

        # confussion matrix
        actual_classes, predicted_classes, _ = MachineLearning.cross_val_predict(model, kfold, all_datasets[dataset_name_test],target_sts[dataset_name_test])
        MachineLearning.plot_confusion_matrix(actual_classes, predicted_classes,[0,1,2,3,4,5], dataset_name_test, model, savePath=os.path.join(figuresFolder_sts,figName + dataset_name_test + str(model) + ".png"))

table_results_test_sts = pd.DataFrame(test_sts, columns = ["Model name", "CV Set"] + result_name_pearson)
print ("Table of results test STS:")
print(table_results_test_sts)

