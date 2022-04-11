import sys
import src.Embeddings_baselines.sts_utils as utils
import src.Preprocess.Utils as preprocess_utils

import os
import argparse
import numpy as np
import pandas as pd
import math
# Evaluate non trainable baselines on a given test set
# Download fasttext embeddings from https://fasttext.cc/docs/en/crawl-vectors.html
# fasttext.vec: wiki-news-300d-1M.vec.zip
# fasttext2.vec: rawl-300d-2M.vec.zip

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate simple STS baselines with word embeddings')
# parser.add_argument('test', help='test file')
parser.add_argument('--embeddings', choices= ['src/Embeddings_baselines/fasttext.vec', 'src/Embeddings_baselines/fasttext2.vec'], default='src/Embeddings_baselines/fasttext.vec', help='the word embeddings txt file')
parser.add_argument('--mode', choices=['centroid', 'align'], default='align', help='the scoring model')
parser.add_argument('--normalize', action='store_true', help='length normalize word embeddings')
parser.add_argument('--keep_stopwords', action='store_true', help='do not remove stopwords')
parser.add_argument('--encoding', default='utf-8', help='the character encoding for input (defaults to utf-8)')
args = parser.parse_args()

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

models=[
   'centroid',
   'align',
]

embeddings=[
   'src/Embeddings_baselines/fasttext.vec',
   'src/Embeddings_baselines/fasttext2.vec',
]

datasetsFolder = "dataset/bin"

if not os.path.exists(datasetsFolder):
    sys.exit("First preprocess the datasets... Exiting")

all_datasets = dict( {name : preprocess_utils.loadDatasetPickle( os.path.join(datasetsFolder, name+".pickle")) for name in all_names })

all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"] = pd.concat( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_h_n"]))
all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"] = pd.concat( (all_datasets["PhrasIS_train_i_p"], all_datasets["PhrasIS_train_i_n"]))
all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p"] = pd.concat( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_i_p"]))
all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = pd.concat( (all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"], all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"]))

all_datasets["PhrasIS_test_h_p+PhrasIS_test_h_n"] = pd.concat( (all_datasets["PhrasIS_test_h_p"], all_datasets["PhrasIS_test_h_n"]))
all_datasets["PhrasIS_test_i_p+PhrasIS_test_i_n"] = pd.concat( (all_datasets["PhrasIS_test_i_p"], all_datasets["PhrasIS_test_i_n"]))
all_datasets["PhrasIS_test_h_p+PhrasIS_test_i_p"] = pd.concat( (all_datasets["PhrasIS_test_h_p"], all_datasets["PhrasIS_test_i_p"]))
all_datasets["PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n"] = pd.concat( (all_datasets["PhrasIS_test_h_p+PhrasIS_test_h_n"], all_datasets["PhrasIS_test_i_p+PhrasIS_test_i_n"]))

test_datasets = list()
test_datasets.append("PhrasIS_test_h_p")
test_datasets.append("PhrasIS_test_i_p")
test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_h_n")
test_datasets.append("PhrasIS_test_i_p+PhrasIS_test_i_n")
test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_i_p")
test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n")

results=[]
for embedding in embeddings:
    # Read embeddings
    f = open(embedding, encoding=args.encoding, errors='surrogateescape')
    words, emb = utils.read_embeddings(f)

    # Build word to index map
    word2ind = {word: i for i, word in enumerate(words)}

    # Length normalize embeddings
    if args.normalize:
        emb = utils.length_normalize_embeddings(emb)

    for evaluation_dataset in test_datasets:
        for model in models:
        
            dataset = all_datasets[evaluation_dataset]
    
            src = dataset['left']
            trg = dataset['right']
            ref = dataset['STS']

            # Tokenize
            src = [utils.tokenize(sent) for sent in src]
            trg = [utils.tokenize(sent) for sent in trg]
    
            # Recase
            src = [utils.recase(sent, word2ind) for sent in src]
            trg = [utils.recase(sent, word2ind) for sent in trg]
    
            # Strip punctuation
            src = [utils.strip_punctuation(sent) for sent in src]
            trg = [utils.strip_punctuation(sent) for sent in trg]

            # Remove stopwords
            if not args.keep_stopwords:
    	        src= [utils.remove_stopwords(sent) for sent in src]
    	        trg=[utils.remove_stopwords(sent) for sent in trg]
	
            # Compute similarities
            sys = np.zeros(ref.shape)
            
            for i in range (ref.shape[0]):
                if model == 'centroid':
                    sys[i]=utils.centroid_cosine (src[i], trg[i], emb, word2ind)
                if model == 'align':
                    sys [i]= utils.align_score(src[i], trg[i], emb, word2ind)
                    
            result=utils.pearson(sys, ref)
            results.append([embedding, evaluation_dataset,model, result])

table_results=pd.DataFrame (results, columns= ["Embedding", "CV Set", "Model Name", "Result"])
print ("Table of results")
print (table_results)
