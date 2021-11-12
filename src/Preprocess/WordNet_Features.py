##Import
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet_ic
import pandas as pd
import numpy as np

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

"""
PYTHON MODULE TO COMPUTE WORDNET FEATURES 
"""

# ####ONTHOLOGY
# ###FEATURE 6: path similarity
# synarray_right= [item for word in df ['right_strip_tokenized_noPunct_lemmat_noStopWords'] for item in word]
# print (synarray_right)
# synarray_left= [item for word in df ['left_strip_tokenized_noPunct_lemmat_noStopWords'] for item in word]
# print (synarray_left)
# response =[m.getMaxPath(wn.synsets(x),wn.synsets(y)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response)
#
# max_values = [x[0] for x in response]
# max_value=np.max(max_values)
# print ('Max value of path_similarity simulate_root=False is: '+str(max_value))
#
#
# ###Feature 7: lch similarity
# ###be careful!!!! we cannot compare a verb with a noun for example.
# #Nouns:
# response_lch =[m.getMaxLch(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_lch)
#
# max_values_lch = [x[0] for x in response_lch]
# max_value_lch=np.max(max_values_lch)
# print ('Max value of lch_similarity (NOUNS) simulte_root=False between nouns is: '+str(max_value_lch))
#
# #verbs
# response_lch_verb =[m.getMaxLch(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_lch_verb)
#
# max_values_lch_verb = [x[0] for x in response_lch_verb]
# max_value_lch_verb=np.max(max_values_lch_verb)
# print ('Max value of lch_similarity (VERBS) simulte_root=False between nouns is: '+str(max_value_lch_verb))
#
#
# ###Feature : wup similarity
# response_wup =[m.getMaxWup(wn.synsets(x),wn.synsets(y)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_wup)
#
# max_values_wup = [x[0] for x in response_wup]
# max_value_wup=np.max(max_values_wup)
# print ('Max value of wup_similarity simulate_root=False is: '+str(max_value_wup))
#
# ####No existe el simulate_root
# ###Feature 8: jcn similarity with BROWN_IC
# ###nouns
# response_jcn_brown =[m.getMaxJcn_brown(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_jcn_brown)
#
# max_values_jcn_brown = [x[0] for x in response_jcn_brown]
# max_value_jcn_brown=np.max(max_values_jcn_brown)
# print ('Max value of jcn_similarity with brown_ic between nouns is: '+str(max_value_jcn_brown))
#
# ##verb
# response_jcn_brown_verb =[m.getMaxJcn_brown(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_jcn_brown_verb)
#
# max_values_jcn_brown_verb = [x[0] for x in response_jcn_brown_verb]
# max_value_jcn_brown_verb=np.max(max_values_jcn_brown_verb)
# print ('Max value of jcn_similarity with brown_ic between verbs is: '+str(max_value_jcn_brown_verb))
#
# #With SEMCOIR_IC
# #nouns
# response_jcn_semcor =[m.getMaxJcn_semcor(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_jcn_semcor)
#
# max_values_jcn_semcor = [x[0] for x in response_jcn_semcor]
# max_value_jcn_semcor=np.max(max_values_jcn_semcor)
# print ('Max value of jcn_similarity with semcor_ic between nouns is: '+str(max_value_jcn_semcor))
#
# #verbs
# response_jcn_semcor_verb =[m.getMaxJcn_semcor(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_jcn_semcor_verb)
#
# max_values_jcn_semcor_verb = [x[0] for x in response_jcn_semcor_verb]
# max_value_jcn_semcor_verb=np.max(max_values_jcn_semcor_verb)
# print ('Max value of jcn_similarity with semcor_ic between verbs is: '+str(max_value_jcn_semcor_verb))
#
#
# ##Feature 9: path_similarity but simulate_root:True
# response_root =[m.getMaxPath_root(wn.synsets(x),wn.synsets(y)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_root)
#
# max_values_root = [x[0] for x in response_root]
# max_value_root=np.max(max_values_root)
# print ('Max value of path_similarity with simulate_root=True is: '+str(max_value_root))
#
#
# ###Feature 10: lch similarity but simulate_root=True
# ###noun
# response_lch_root =[m.getMaxLch_root(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_lch_root)
#
# max_values_lch_root = [x[0] for x in response_lch_root]
# max_value_lch_root=np.max(max_values_lch_root)
# print ('Max value of lch_similarity with simulate_root=True between nouns is: '+str(max_value_lch_root))
#
# ##verb
# response_lch_root_verb =[m.getMaxLch_root(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_lch_root_verb)
#
# max_values_lch_root_verb = [x[0] for x in response_lch_root_verb]
# max_value_lch_root_verb=np.max(max_values_lch_root_verb)
# print ('Max value of lch_similarity with simulate_root=True between verbs is: '+str(max_value_lch_root_verb))
#
# ###Wup: simulate_root=True
# response_wup_root =[m.getMaxWup_root(wn.synsets(x),wn.synsets(y)) for x in synarray_right for y in synarray_left]
# print('The maximum of each pair: ')
# print (response_wup_root)
#
# max_values_wup_root = [x[0] for x in response_wup_root]
# max_value_wup_root=np.max(max_values_wup_root)
# print ('Max value of wup_similarity simulate_root=True is: '+str(max_value_wup_root))
#
#
# ####WORDNEET RAAW
# #FEATURE 12 and 13: If chunk1 senses are more specific
# #for word in synarray_right:
#     #for x in wn.synsets(word):
#         #print (x, x.max_depth())
#
# synarray_right= [x.max_depth() for word in synarray_right for x in wn.synsets(word)]
# maximo_right= max(synarray_right)
# print (maximo_right)
#
# synarray_left= [x.max_depth() for word in synarray_left for x in wn.synsets(word)]
# maximo_left = max(synarray_left)
# print (maximo_left)
#
# if maximo_left > maximo_right:
#     print (True)
#     print ("Chunk left is more specific")
# else:
#     print (False)
#     print ("Chunk right is more specific")
#
# ###Feature 14:
# difference=abs(maximo_left-maximo_right)
# print ('The difference is: ' +str(difference))
#
# ##Feature 15 and 16: Minimum and maximum value of pairwise difference of WordNet depth
# print("Right: ")
# print (synarray_right)
# print ("Left: ")
# print (synarray_left)
#
# differences= [abs (x-y) for y in synarray_right for x in synarray_left]
# print (differences)
#
# min_diff=min (differences)
# max_diff=max (differences)
#
# print ("The minimum difference: " + str (min_diff))
# print ("The maximum difference: " + str (max_diff))


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