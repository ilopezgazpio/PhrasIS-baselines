##Import
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet_ic
import pandas as pd
import numpy as np

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

def substract(a, b):
    return a - b

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def getMaxPath(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list

    sim= [wn.path_similarity(i,j, simulate_root=False) for i in synset1 for j in synset2]

    #remove None because of simulate_root=False
    sim = [x for x in sim if x is not None]

    try:
        max_sim=max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim=0
        s1='None'
        s2='None'

    return max_sim, s1, s2

def getMaxLch(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [wn.lch_similarity(i,j, simulate_root=False) for i in synset1 for j in synset2]

    ##remove None because of simulate_root=False
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2

def getMaxWup(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [wn.wup_similarity(i,j, simulate_root=False) for i in synset1 for j in synset2]

    ##remove None because of simulate_root=False
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2

def getMaxJcn_brown(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [i.jcn_similarity(j, brown_ic) for i in synset1 for j in synset2]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2

def getMaxJcn_semcor(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [i.jcn_similarity(j, semcor_ic) for i in synset1 for j in synset2]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2

def getMaxPath_root(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [wn.path_similarity(i,j, simulate_root=True) for i in synset1 for j in synset2]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2

def getMaxLch_root(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [wn.lch_similarity(i,j, simulate_root=True) for i in synset1 for j in synset2]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2

def getMaxWup_root(synset1,synset2):

    a= [i.name() for i in synset1 for j in synset2] # save the names from synsets1 into list
    b = [j.name() for i in synset1 for j in synset2] # save the names from synsets2 into list
    sim= [wn.wup_similarity(i,j, simulate_root=True) for i in synset1 for j in synset2]

    try:
        max_sim = max(sim, default=0)
        idx = np.argmax(sim)
        s1 = a[idx]  # get the name of synset1 for which path sim is max
        s2 = b[idx]  # get the name of synset2 for which path sim is max
    except ValueError:
        max_sim = 0
        s1 = 'None'
        s2 = 'None'

    return max_sim, s1, s2