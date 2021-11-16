##Import
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import genesis
import pandas as pd
import numpy as np

brown_ic = wordnet_ic.ic('ic-brown.dat')
genesis_ic = wn.ic(genesis, False, 0.0)

"""
PYTHON MODULE TO COMPUTE WORDNET FEATURES 
"""

# ####ONTHOLOGY
# MAX PATH_SIMILARITY
def addColumnsPathSimilarity(df : pd.DataFrame):
    df['path_similarity'] = df.apply(lambda x: get_Pathsimilarity(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX LCH_SIMILARITY_NOUNS
def addColumnsLchSimilarityNouns(df : pd.DataFrame):
    df['lch_similarity_nouns'] = df.apply(lambda x: get_Lchsimilarity_nouns(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX LCH_SIMILARITY_VERBS
def addColumnsLchSimilarityVerbs(df : pd.DataFrame):
    df['lch_similarity_verbs'] = df.apply(lambda x: get_Lchsimilarity_verbs(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX JCN_SIMILARITY_BROWN_NOUNS
def addColumnsJcnSimilarityBrownNouns(df : pd.DataFrame):
    df['jcn_similarity_brown_nouns'] = df.apply(lambda x: get_Jcnsimilarity_brown_nouns(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX JCN_SIMILARITY_BROWN_VERBS
def addColumnsJcnSimilarityBrownVerbs(df : pd.DataFrame):
    df['jcn_similarity_brown_verbs'] = df.apply(lambda x: get_Jcnsimilarity_brown_verbs(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX JCN_SIMILARITY_GENESIS_NOUNS
def addColumnsJcnSimilarityGenesisNouns(df : pd.DataFrame):
    df['jcn_similarity_genesis_nouns'] = df.apply(lambda x: get_Jcnsimilarity_genesis_nouns(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX JCN_SIMILARITY_GENESIS_VERBS
def addColumnsJcnSimilarityGenesisVerbs(df : pd.DataFrame):
    df['jcn_similarity_genesis_verbs'] = df.apply(lambda x: get_Jcnsimilarity_genesis_verbs(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX WUP_SIMILARITY
def addColumnsWupSimilarity(df : pd.DataFrame):
    df['wup_similarity'] = df.apply(lambda x: get_Wupsimilarity(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX PATH_SIMILARITY_ROOT
def addColumnsPathSimilarityRoot(df : pd.DataFrame):
    df['path_similarity_root'] = df.apply(lambda x: get_PathsimilarityRoot(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX LCH_SIMILARITY_NOUNS_ROOT
def addColumnsLchSimilarityNounsRoot(df : pd.DataFrame):
    df['lch_similarity_nouns_root'] = df.apply(lambda x: get_Lchsimilarity_nouns_root(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX LCH_SIMILARITY_VERBS_ROOT
def addColumnsLchSimilarityVerbsRoot(df : pd.DataFrame):
    df['lch_similarity_verbs_root'] = df.apply(lambda x: get_Lchsimilarity_verbs_root(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAX WUP_SIMILARITY_ROOT
def addColumnsWupSimilarityRoot(df : pd.DataFrame):
    df['wup_similarity_root'] = df.apply(lambda x: get_Wupsimilarity_root(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# CHUNK 1 MAXIMUM DEPTH
def addColumnsChunk1Maximum(df : pd.DataFrame):
    df['chunck1_maximum'] = df.apply(lambda x: get_Chunk1Maximum(x['left_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# CHUNK 2 MAXIMUM DEPTH
def addColumnsChunk2Maximum(df : pd.DataFrame):
    df['chunck2_maximum'] = df.apply(lambda x: get_Chunk2Maximum(x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# CHUNK 1 MORE SPECIFIC THAN CHUNK 2
def addColumnsChunk1Specific(df : pd.DataFrame):
    df['chunck1>chunk2'] = df.apply(lambda x: get_Chunk1Specific(x['chunck1_maximum'], x['chunck2_maximum']), axis=1)

# CHUNK 2 MORE SPECIFIC THAN CHUNK 1
def addColumnsChunk2Specific(df : pd.DataFrame):
    df['chunck2>chunk1'] = df.apply(lambda x: get_Chunk2Specific(x['chunck1_maximum'], x['chunck2_maximum']), axis=1)

# DIFFERENCE DEPTH CHUNKS
def addColumnsDifference(df : pd.DataFrame):
    df['|chunck1-chunk2|'] = df.apply(lambda x: difference(x['chunck1_maximum'], x['chunck2_maximum']), axis=1)

# MINIMUM DIFFERENCE
def addColumnsMinimumDifference(df : pd.DataFrame):
    df['minimum_difference'] = df.apply(lambda x: get_MinimumDifference(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

# MAXIMUM DIFFERENCE
def addColumnsMaximumDifference(df : pd.DataFrame):
    df['maximum_difference'] = df.apply(lambda x: get_MaximumDifference(x['left_strip_tokenized_noPunct_lemmat_noStopWords'], x['right_strip_tokenized_noPunct_lemmat_noStopWords']), axis=1)

def get_Pathsimilarity(column1, column2):

    response =[getMaxPath(wn.synsets(x),wn.synsets(y)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Lchsimilarity_nouns(column1, column2):

    response =[getMaxLch(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Lchsimilarity_verbs(column1, column2):

    response =[getMaxLch(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Jcnsimilarity_brown_nouns(column1, column2):

    response =[getMaxJcnBrown(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Jcnsimilarity_brown_verbs(column1, column2):

    response =[getMaxJcnBrown(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Jcnsimilarity_genesis_nouns(column1, column2):

    response =[getMaxJcnGenesis(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Jcnsimilarity_genesis_verbs(column1, column2):

    response =[getMaxJcnGenesis(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Wupsimilarity(column1, column2):

    response =[getMaxWup(wn.synsets(x),wn.synsets(y)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_PathsimilarityRoot(column1, column2):

    response =[getMaxPathRoot(wn.synsets(x),wn.synsets(y)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Lchsimilarity_nouns_root(column1, column2):

    response =[getMaxLchRoot(wn.synsets(x, pos=wn.NOUN),wn.synsets(y, pos=wn.NOUN)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Lchsimilarity_verbs_root(column1, column2):

    response =[getMaxLchRoot(wn.synsets(x, pos=wn.VERB),wn.synsets(y, pos=wn.VERB)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Wupsimilarity_root(column1, column2):

    response =[getMaxWupRoot(wn.synsets(x),wn.synsets(y)) for x in column2 for y in column1]
    try:
        max_value = np.max(response)
    except ValueError:
        max_value=0

    return max_value

def get_Chunk1Maximum (column1):

    synarray_left = [x.max_depth() for word in column1 for x in wn.synsets(word)]
    try:
        maximo_left = max(synarray_left)
    except ValueError:
        maximo_left = 0

    return maximo_left


def get_Chunk2Maximum(column2):
    synarray_left = [x.max_depth() for word in column2 for x in wn.synsets(word)]
    try:
        maximo_left = max(synarray_left)
    except ValueError:
        maximo_left = 0

    return maximo_left

def get_Chunk1Specific(maximo_left, maximo_right):

    if maximo_left>maximo_right:
        return True
    else:
        return False

def get_Chunk2Specific(maximo_left, maximo_right):

    if maximo_right>maximo_left:
        return True
    else:
        return False

def difference (maximo_left, maximo_right):
    return abs(maximo_left-maximo_right)


def get_MinimumDifference(left, right):

    synarray_right= [x.max_depth() for word in right for x in wn.synsets(word)]
    synarray_left = [x.max_depth() for word in left for x in wn.synsets(word)]

    differences= [abs (x-y) for y in synarray_left for x in synarray_right]
    try:
        min_diff = min(differences)
    except ValueError:
        min_diff=0
    return min_diff

def get_MaximumDifference(left, right):

    synarray_right= [x.max_depth() for word in right for x in wn.synsets(word)]
    synarray_left = [x.max_depth() for word in left for x in wn.synsets(word)]

    differences= [abs (x-y) for y in synarray_left for x in synarray_right]

    try:
        max_diff = max(differences)
    except ValueError:
        max_diff=0
    return max_diff

def getMaxPath(synset1, synset2):

    sim = [wn.path_similarity(i, j, simulate_root=False) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxLch(synset1, synset2):

    sim = [wn.lch_similarity(i, j, simulate_root=False) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxJcnBrown(synset1, synset2):

    sim = [wn.jcn_similarity(i, j, brown_ic) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxJcnGenesis(synset1, synset2):

    sim = [wn.jcn_similarity(i, j, genesis_ic) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxWup(synset1, synset2):

    sim = [wn.wup_similarity(i, j, simulate_root=False) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxPathRoot(synset1, synset2):

    sim = [wn.path_similarity(i, j, simulate_root=True) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxLchRoot(synset1, synset2):

    sim = [wn.lch_similarity(i, j, simulate_root=True) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim

def getMaxWupRoot(synset1, synset2):

    sim = [wn.wup_similarity(i, j, simulate_root=True) for i in synset1 for j in synset2]
    sim = [x for x in sim if x is not None]

    try:
        max_sim = max(sim, default=0)
    except ValueError:
        max_sim = 0

    return max_sim
