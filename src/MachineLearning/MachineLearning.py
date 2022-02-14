from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np

def DecisionTree (X_train, y_sts, y_nli, score):

    clf = tree.DecisionTreeClassifier()
    scores_sts = cross_val_score(clf, X_train, y_sts , cv=3, scoring=score)
    scores_nli= cross_val_score(clf, X_train, y_nli , cv=3, scoring=score)

    media_sts= scores_sts.mean()
    media_nli= scores_nli.mean()

    return media_sts, media_nli

def Knn (X_train, y_sts, y_nli, score):
    k = 4
    k4_sts = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores_sts=cross_val_score(k4_sts, X_train, y_sts, cv=3, scoring=score)
    k4_nli = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores_nli = cross_val_score(k4_nli, X_train, y_nli, cv=3, scoring=score)

    media_sts=scores_sts.mean()
    media_nli=scores_nli.mean()

    return media_sts, media_nli

def LR (X_train, y_sts, y_nli, score):

    if np.sum(y_sts)==0:
        #error: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
        #hago esto porque no puede haber un test solo de una clase. Y en esos casos la media siempre es 1 porque se acierta siempre
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

    gnb = GaussianNB()
    scores_sts=cross_val_score(gnb, X_train, y_sts, cv=3, scoring=score)
    scores_nli=cross_val_score(gnb, X_train, y_nli, cv=3, scoring=score)

    media_sts=scores_sts.mean()
    media_nli=scores_nli.mean()

    return media_sts,media_nli

def SVM (X_train, y_sts, y_nli, score):

    if np.sum(y_sts)==0:
        return 1.0,1.0

    else:
        clf = svm.SVC(kernel='linear')
        scores_sts = cross_val_score(clf, X_train, y_sts, cv=3, scoring=score)
        scores_nli = cross_val_score(clf, X_train, y_nli, cv=3, scoring=score)

        media_sts = scores_sts.mean()
        media_nli = scores_nli.mean()

        return media_sts, media_nli

def calculate (media_sts, media_nli, length):

    #print (media_sts, media_nli)

    res_sts_acc_knn=(sum(media_sts[i] for i in range(0, len(media_sts), length))/(len(media_sts)/length))
    res_sts_f1_knn=(sum(media_sts[i] for i in range(1, len(media_sts), length))/(len(media_sts)/length)) #sum of elements in odd positions (pares)
    res_nli_acc_knn=(sum(media_nli[i] for i in range(0, len(media_nli), length))/(len(media_nli)/length))
    res_nli_f1_knn= (sum(media_nli[i] for i in range(1, len(media_nli), length))/(len(media_nli)/length))

    return res_sts_acc_knn, res_sts_f1_knn, res_nli_acc_knn, res_nli_f1_knn