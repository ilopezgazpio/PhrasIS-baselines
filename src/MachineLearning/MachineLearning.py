from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

def DecisionTree (X_train, y_sts, y_nli, score):
    clf = tree.DecisionTreeClassifier()
    scores_sts = cross_val_score(clf, X_train, y_sts , cv=3, scoring=score)
    scores_nli= cross_val_score(clf, X_train, y_nli , cv=3, scoring=score)

    media_sts= scores_sts.mean()
    media_nli= scores_nli.mean()

    return media_sts, media_nli

def Knn (X_train, y_sts, y_nli, score):
    k = 4
    k4_sts = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_sts)
    scores_sts=cross_val_score(k4_sts, X_train, y_sts, cv=3, scoring=score)
    k4_nli = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_nli)
    scores_nli = cross_val_score(k4_nli, X_train, y_sts, cv=3, scoring=score)

    media_sts=scores_sts.mean()
    media_nli=scores_nli.mean()

    return media_sts, media_nli

def calculate (media_sts, media_nli, length):

    #print (media_sts, media_nli)

    res_sts_acc_knn=(sum(media_sts[i] for i in range(0, len(media_sts), length))/(len(media_sts)/length))
    res_sts_f1_knn=(sum(media_sts[i] for i in range(1, len(media_sts), length))/(len(media_sts)/length)) #sum of elements in odd positions (pares)
    res_nli_acc_knn=(sum(media_nli[i] for i in range(0, len(media_nli), length))/(len(media_nli)/length))
    res_nli_f1_knn= (sum(media_nli[i] for i in range(1, len(media_nli), length))/(len(media_nli)/length))

    return res_sts_acc_knn, res_sts_f1_knn, res_nli_acc_knn, res_nli_f1_knn