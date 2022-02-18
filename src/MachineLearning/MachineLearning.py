from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np

def DecisionTree (X_train, y_sts, y_nli, score):

    clf_sts = tree.DecisionTreeClassifier().fit(X_train,y_sts)
    scores_sts = cross_val_score(clf_sts, X_train, y_sts , cv=3, scoring=score)
    clf_nli= tree.DecisionTreeClassifier().fit (X_train, y_nli)
    scores_nli= cross_val_score(clf_nli, X_train, y_nli , cv=3, scoring=score)

    media_sts= scores_sts.mean()
    media_nli= scores_nli.mean()

    return media_sts, media_nli

def Knn (X_train, y_sts, y_nli, score):
    k = 4
    k4_sts = KNeighborsClassifier(n_neighbors=k, weights='distance').fit (X_train, y_sts)
    scores_sts=cross_val_score(k4_sts, X_train, y_sts, cv=3, scoring=score)
    k4_nli = KNeighborsClassifier(n_neighbors=k, weights='distance').fit (X_train, y_sts)
    scores_nli = cross_val_score(k4_nli, X_train, y_nli, cv=3, scoring=score)

    media_sts=scores_sts.mean()
    media_nli=scores_nli.mean()

    return media_sts, media_nli

def LR (X_train, y_sts, y_nli, score):

    if np.sum(y_sts)==0:
        return 1.0,1.0
    else:

        lr_sts = LogisticRegression(solver='saga').fit(X_train, y_sts)
        scores_sts= cross_val_score(lr_sts, X_train, y_sts, cv=3, scoring=score)
        lr_nli = LogisticRegression(solver='saga').fit(X_train, y_nli)
        scores_nli = cross_val_score(lr_nli, X_train, y_nli, cv=3, scoring=score)

        media_st = scores_sts.mean()
        media_nl = scores_nli.mean()

        return media_st, media_nl

def NB (X_train, y_sts, y_nli, score):

    gnb_sts = GaussianNB().fit(X_train, y_sts)
    gnb_nli = GaussianNB().fit (X_train, y_nli)
    scores_sts=cross_val_score(gnb_sts, X_train, y_sts, cv=3, scoring=score)
    scores_nli=cross_val_score(gnb_nli, X_train, y_nli, cv=3, scoring=score)

    media_sts=scores_sts.mean()
    media_nli=scores_nli.mean()

    return media_sts,media_nli

def SVM (X_train, y_sts, y_nli, score):

    if np.sum(y_sts)==0:
        return 1.0,1.0

    else:
        clf_sts = svm.SVC(kernel='linear').fit(X_train, y_sts)
        clf_nli = svm.SVC(kernel='linear').fit(X_train, y_nli)
        scores_sts = cross_val_score(clf_sts, X_train, y_sts, cv=3, scoring=score)
        scores_nli = cross_val_score(clf_nli, X_train, y_nli, cv=3, scoring=score)

        media_sts = scores_sts.mean()
        media_nli = scores_nli.mean()

        return media_sts, media_nli

def calculate(media_sts, media_nli, length):

    res_sts_acc=(sum(media_sts[i] for i in range(0, len(media_sts), length))/(len(media_sts)/length))
    res_sts_f1_ma=(sum(media_sts[i] for i in range(1, len(media_sts), length))/(len(media_sts)/length))
    res_sts_f1_mi=(sum(media_sts[i] for i in range(2, len(media_sts), length))/(len(media_sts)/length))
    res_sts_bal_acu=(sum(media_sts[i] for i in range(3, len(media_sts), length))/(len(media_sts)/length))
    res_sts_f1_we= (sum(media_sts[i] for i in range(4, len(media_sts), length))/(len(media_sts)/length))
    res_nli_acc=(sum(media_nli[i] for i in range(0, len(media_nli), length))/(len(media_nli)/length))
    res_nli_f1_ma= (sum(media_nli[i] for i in range(1, len(media_nli), length))/(len(media_nli)/length))
    res_nli_f1_mi= (sum(media_nli[i] for i in range(2, len(media_nli), length))/(len(media_nli)/length))
    res_nli_bal_acu= (sum(media_nli[i] for i in range(3, len(media_nli), length))/(len(media_nli)/length))
    res_nli_f1_we= (sum(media_nli[i] for i in range(4, len(media_nli), length))/(len(media_nli)/length))

    return res_sts_acc, res_sts_f1_ma, res_sts_f1_mi, res_sts_bal_acu, res_sts_f1_we, res_nli_acc, res_nli_f1_ma, res_nli_f1_mi, res_nli_bal_acu, res_nli_f1_we