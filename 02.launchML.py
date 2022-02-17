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

# Set seed for all libraries
np.random.seed(123)

# To print the whole df
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Step 1 -> load datasets
names = [
    'PhrasIS_train_h_n',
    'PhrasIS_train_h_p',
    'PhrasIS_train_i_n',
    'PhrasIS_train_i_p',
]

saveFolder = "dirty"
if not os.path.exists(saveFolder):
    sys.exit("First preprocess the datasets... Exiting")

datasets = dict( {name : Utils.loadDatasetPickle( os.path.join(saveFolder, name+".pickle")) for name in names })

# Step 2 -> instantiate models
columns_of_X = ['left', 'right']
column_y_sts = ['STS']
column_y_nli= ['NLI']

scores_sts_de=[]
scores_nli_de=[]
scores_sts_kn=[]
scores_nli_kn=[]
scores_sts_lr=[]
scores_nli_lr=[]
scores_sts_nb=[]
scores_nli_nb=[]
scores_sts_svm=[]
scores_nli_svm=[]

scores= {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "balanced_accuracy": "balanced_accuracy",
    "f1_weighted" : "f1_weighted"
}

for name, dataset in datasets.items():
    X_strings = dataset[columns_of_X]
    X_strings['cv'] = X_strings['left'] + "," + X_strings['right']
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_strings['cv'])
    X_train = np.concatenate((X_bow.toarray(), dataset[ALL_FEATURES].values), axis=1)
    y_sts = dataset[column_y_sts].values
    y_nli =dataset[column_y_nli].values

# Step 3 -> train models

    for col in scores:
        medias_sts_de, medias_nli_de = MachineLearning.DecisionTree(X_train, y_sts, y_nli, col)
        scores_sts_de.append(medias_sts_de)
        scores_nli_de.append(medias_nli_de)

        medias_sts_knn, medias_nli_knn= MachineLearning.Knn(X_train, y_sts, y_nli, col)
        scores_sts_kn.append(medias_sts_knn)
        scores_nli_kn.append(medias_nli_knn)

        medias_sts_svm, medias_nli_svm=MachineLearning.SVM(X_train, y_sts, y_nli, col)
        scores_sts_svm.append(medias_sts_svm)
        scores_nli_svm.append(medias_nli_svm)

        medias_sts_lr, medias_nli_lr = MachineLearning.LR(X_train, y_sts, y_nli, col)
        scores_sts_lr.append(medias_sts_lr)
        scores_nli_lr.append(medias_nli_lr)

        medias_sts_nb, medias_nli_nb = MachineLearning.NB(X_train, y_sts, y_nli, col)
        scores_sts_nb.append(medias_sts_nb)
        scores_nli_nb.append(medias_nli_nb)

# Step 4 -> evaluate models
res_sts_acc_de, res_sts_f1_ma_de, res_sts_f1_mi_de, res_sts_ba_acc_de, res_sts_f1_we_de, res_nli_acc_de, res_nli_f1_ma_de, res_nli_f1_mi_de, res_nli_ba_acc_de, res_nli_f1_we_de= MachineLearning.calculate(scores_sts_de, scores_nli_de, len(scores))
print ("\nDecision tree results: ")
print("STS: %0.3f accuracy" % res_sts_acc_de)
print("NLI: %0.3f accuracy" % res_nli_acc_de)
print("STS: %0.3f f1_macro" % res_sts_f1_ma_de)
print("NLI: %0.3f f1_macro" % res_nli_f1_ma_de)
print("STS: %0.3f f1_micro" % res_sts_f1_mi_de)
print("NLI: %0.3f f1_micro" % res_nli_f1_mi_de)
print("STS: %0.3f balanced_accuracy" % res_sts_ba_acc_de)
print("NLI: %0.3f balanced_accuracy" % res_nli_ba_acc_de)
print("STS: %0.3f f1_weighted" % res_sts_f1_we_de)
print("NLI: %0.3f f1_weighted" % res_nli_f1_we_de)

res_sts_acc_knn, res_sts_f1_ma_knn, res_sts_f1_mi_knn, res_sts_ba_acc_knn, res_sts_f1_we_knn, res_nli_acc_knn, res_nli_f1_ma_knn, res_nli_f1_mi_knn, res_nli_ba_acc_knn, res_nli_f1_we_knn = MachineLearning.calculate(scores_sts_kn, scores_nli_kn, len(scores))
print ("\nKnn results: ")
print("STS: %0.3f accuracy" % res_sts_acc_knn)
print("NLI: %0.3f accuracy" % res_nli_acc_knn)
print("STS: %0.3f f1_macro" % res_sts_f1_ma_knn)
print("NLI: %0.3f f1_macro" % res_nli_f1_ma_knn)
print("STS: %0.3f f1_micro" % res_sts_f1_mi_knn)
print("NLI: %0.3f f1_micro" % res_nli_f1_mi_knn)
print("STS: %0.3f balanced_accuracy" % res_sts_ba_acc_knn)
print("NLI: %0.3f balanced_accuracy" % res_nli_ba_acc_knn)
print("STS: %0.3f f1_weighted" % res_sts_f1_we_knn)
print("NLI: %0.3f f1_weighted" % res_nli_f1_we_knn)

res_sts_acc_lr, res_sts_f1_ma_lr, res_sts_f1_mi_lr, res_sts_ba_acc_lr, res_sts_f1_we_lr, res_nli_acc_lr, res_nli_f1_ma_lr, res_nli_f1_mi_lr, res_nli_ba_acc_lr, res_nli_f1_we_lr = MachineLearning.calculate(scores_sts_lr, scores_nli_lr, len(scores))
print ("\nLinear regresssion results: ")
print("STS: %0.3f accuracy" % res_sts_acc_lr)
print("NLI: %0.3f accuracy" % res_nli_acc_lr)
print("STS: %0.3f f1_macro" % res_sts_f1_ma_lr)
print("NLI: %0.3f f1_macro" % res_nli_f1_ma_lr)
print("STS: %0.3f f1_micro" % res_sts_f1_mi_lr)
print("NLI: %0.3f f1_micro" % res_nli_f1_mi_lr)
print("STS: %0.3f balanced_accuracy" % res_sts_ba_acc_lr)
print("NLI: %0.3f balanced_accuracy" % res_nli_ba_acc_lr)
print("STS: %0.3f f1_weighted" % res_sts_f1_we_lr)
print("NLI: %0.3f f1_weighted" % res_nli_f1_we_lr)

res_sts_acc_nb, res_sts_f1_ma_nb, res_sts_f1_mi_nb, res_sts_ba_acc_nb, res_sts_f1_we_nb, res_nli_acc_nb, res_nli_f1_ma_nb, res_nli_f1_mi_nb, res_nli_ba_acc_nb, res_nli_f1_we_nb = MachineLearning.calculate(scores_sts_nb, scores_nli_nb, len(scores))
print ("\nGaussian Naive Bayes results: ")
print("STS: %0.3f accuracy" % res_sts_acc_nb)
print("NLI: %0.3f accuracy" % res_nli_acc_nb)
print("STS: %0.3f f1_macro" % res_sts_f1_ma_nb)
print("NLI: %0.3f f1_macro" % res_nli_f1_ma_nb)
print("STS: %0.3f f1_micro" % res_sts_f1_mi_nb)
print("NLI: %0.3f f1_micro" % res_nli_f1_mi_nb)
print("STS: %0.3f balanced_accuracy" % res_sts_ba_acc_nb)
print("NLI: %0.3f balanced_accuracy" % res_nli_ba_acc_nb)
print("STS: %0.3f f1_weighted" % res_sts_f1_we_nb)
print("NLI: %0.3f f1_weighted" % res_nli_f1_we_nb)

res_sts_acc_svm, res_sts_f1_ma_svm, res_sts_f1_mi_svm, res_sts_ba_acc_svm, res_sts_f1_we_svm, res_nli_acc_svm, res_nli_f1_ma_svm, res_nli_f1_mi_svm, res_nli_ba_acc_svm, res_nli_f1_we_svm= MachineLearning.calculate(scores_sts_svm, scores_nli_svm, len(scores))
print ("\nSupport Vector Machine results: ")
print("STS: %0.3f accuracy" % res_sts_acc_svm)
print("NLI: %0.3f accuracy" % res_nli_acc_svm)
print("STS: %0.3f f1_macro" % res_sts_f1_ma_svm)
print("NLI: %0.3f f1_macro" % res_nli_f1_ma_svm)
print("STS: %0.3f f1_micro" % res_sts_f1_mi_svm)
print("NLI: %0.3f f1_micro" % res_nli_f1_mi_svm)
print("STS: %0.3f balanced_accuracy" % res_sts_ba_acc_svm)
print("NLI: %0.3f balanced_accuracy" % res_nli_ba_acc_svm)
print("STS: %0.3f f1_weighted" % res_sts_f1_we_svm)
print("NLI: %0.3f f1_weighted" % res_nli_f1_we_svm)

# Step 5 -> save results
