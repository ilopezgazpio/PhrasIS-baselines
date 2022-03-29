# (c) Mikel Artetxe

import src.Embeddings_baselines.sts_utils as utils

import os
import sys
import argparse
import numpy as np
import sklearn.ensemble
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.svm
import pandas as pd
import src.Preprocess.Utils as preprocess_utils


def embedding_features(src, trg, emb, word2ind):
    features = []

    # BOW cosine
    features.append(utils.bow_cosine(src, trg))

    # Align
    features.append(utils.align_score(src, trg, emb, word2ind))

    # Align for non-overlapping tokens
    a = [word for word in src if word in word2ind and not word in trg]
    b = [word for word in trg if word in word2ind and not word in src]
    features.append(utils.align_score(a, b, emb, word2ind))

    return features


def length_features(src, trg):
    return [len(src), len(trg), abs(len(src) - len(trg))]


def overlap_features(src, trg, ngrams=(1,2,3)):
    features = []
    for n in ngrams:
        src_ngrams = utils.ngrams(src, n)
        trg_ngrams = utils.ngrams(trg, n)
        features.append(utils.overlap(src_ngrams, trg_ngrams))
        features.append(utils.overlap(trg_ngrams, src_ngrams))
    return features


def oov_features(src, trg, vocab):
    features = []

    # OOVs
    src_oov = [x for x in src if x not in vocab]
    trg_oov = [x for x in trg if x not in vocab]
    features.append(len(src_oov) / max(1, len(src)))
    features.append(len(trg_oov) / max(1, len(trg)))

    # Overlapping OOVs
    features.append(utils.overlap(src_oov, trg_oov))
    features.append(utils.overlap(trg_oov, src_oov))

    return features


def mt_features(src, trg):
    features = []
    features.append(utils.gleu(src, trg))  # GLEU is symmetric
    features.append(utils.chrf(src, trg))
    features.append(utils.chrf(trg, src))
    return features


def extract_instance_features(src, trg, emb, word2ind):
    # Tokenize
    src = utils.tokenize(src)
    trg = utils.tokenize(trg)

    # POS tagging
    src_pos = utils.pos_tag(src)
    trg_pos = utils.pos_tag(trg)

    # Recase
    src = utils.recase(src, word2ind)
    trg = utils.recase(trg, word2ind)

    # Nouns
    src_nouns = utils.nouns(src)
    trg_nouns = utils.nouns(trg)

    # Verbs
    src_verbs = utils.verbs(src)
    trg_verbs = utils.verbs(trg)

    # Adjectives
    src_adjectives = utils.adjectives(src)
    trg_adjectives = utils.adjectives(trg)

    # Adverbs
    src_adverbs = utils.adverbs(src)
    trg_adverbs = utils.adverbs(trg)

    # Stemming
    src_stemmed = utils.stem(src)
    trg_stemmed = utils.stem(trg)

    # Strip punctuation
    src = utils.strip_punctuation(src)
    trg = utils.strip_punctuation(trg)
    src_stemmed = utils.strip_punctuation(src_stemmed)
    trg_stemmed = utils.strip_punctuation(trg_stemmed)

    # Remove stopwords
    src_filt = utils.remove_stopwords(src)
    trg_filt = utils.remove_stopwords(trg)

    # Extract features
    features = []
    features.extend(length_features(src, trg))
    features.extend(overlap_features(src, trg))
    features.extend(oov_features(src, trg, word2ind))
    features.extend(embedding_features(src, trg, emb, word2ind))
    features.extend(mt_features(src, trg))

    features.extend(length_features(src_filt, trg_filt))
    features.extend(overlap_features(src_filt, trg_filt))
    features.extend(oov_features(src_filt, trg_filt, word2ind))
    features.extend(embedding_features(src_filt, trg_filt, emb, word2ind))
    features.extend(mt_features(src_filt, trg_filt))

    features.extend(length_features(src_nouns, trg_nouns))
    features.extend(overlap_features(src_nouns, trg_nouns))
    features.extend(oov_features(src_nouns, trg_nouns, word2ind))
    features.extend(embedding_features(src_nouns, trg_nouns, emb, word2ind))

    features.extend(length_features(src_verbs, trg_verbs))
    features.extend(overlap_features(src_verbs, trg_verbs))
    features.extend(oov_features(src_verbs, trg_verbs, word2ind))
    features.extend(embedding_features(src_verbs, trg_verbs, emb, word2ind))

    features.extend(length_features(src_adjectives, trg_adjectives))
    features.extend(overlap_features(src_adjectives, trg_adjectives))
    features.extend(oov_features(src_adjectives, trg_adjectives, word2ind))
    features.extend(embedding_features(src_adjectives, trg_adjectives, emb, word2ind))

    features.extend(length_features(src_adverbs, trg_adverbs))
    features.extend(overlap_features(src_adverbs, trg_adverbs))
    features.extend(oov_features(src_adverbs, trg_adverbs, word2ind))
    features.extend(embedding_features(src_adverbs, trg_adverbs, emb, word2ind))

    features.extend(length_features(src_stemmed, trg_stemmed))
    features.extend(overlap_features(src_stemmed, trg_stemmed))
    features.extend(mt_features(src_stemmed, trg_stemmed))

    features.extend(overlap_features(src_pos, trg_pos, (1, 2, 3, 4, 5)))

    return features


def extract_features(src, trg, emb, word2ind):
    return np.array([extract_instance_features(src[i], trg[i], emb, word2ind) for i in range(len(src))])
    

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a simple STS ML system')
    # parser.add_argument('train', help='training data')
    # parser.add_argument('test', help='test data')
    parser.add_argument('--embeddings', default='src/Embeddings_baselines/fasttext.vec', help='the word embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input (defaults to utf-8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    # Read embeddings
    f = open(args.embeddings, encoding=args.encoding, errors='surrogateescape')
    words, emb = utils.read_embeddings(f)

    # Build word to index map
    word2ind = {word: i for i, word in enumerate(words)}

    # Read data
    #train_src, train_trg, train_ref = utils.read_data(open(args.train, encoding=args.encoding, errors='surrogateescape'))
    #test_src, test_trg, test_ref = utils.read_data(open(args.test, encoding=args.encoding, errors='surrogateescape'))


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

    all_datasets = dict( {name : preprocess_utils.loadDatasetPickle( os.path.join(datasetsFolder, name+".pickle")) for name in all_names })

    all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"] = pd.concat( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_h_n"]))
    all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"] = pd.concat( (all_datasets["PhrasIS_train_i_p"], all_datasets["PhrasIS_train_i_n"]))
    all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p"] = pd.concat( (all_datasets["PhrasIS_train_h_p"], all_datasets["PhrasIS_train_i_p"]))
    all_datasets["PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n"] = pd.concat( (all_datasets["PhrasIS_train_h_p+PhrasIS_train_h_n"], all_datasets["PhrasIS_train_i_p+PhrasIS_train_i_n"]))

    all_datasets["PhrasIS_test_h_p+PhrasIS_test_h_n"] = pd.concat( (all_datasets["PhrasIS_test_h_p"], all_datasets["PhrasIS_test_h_n"]))
    all_datasets["PhrasIS_test_i_p+PhrasIS_test_i_n"] = pd.concat( (all_datasets["PhrasIS_test_i_p"], all_datasets["PhrasIS_test_i_n"]))
    all_datasets["PhrasIS_test_h_p+PhrasIS_test_i_p"] = pd.concat( (all_datasets["PhrasIS_test_h_p"], all_datasets["PhrasIS_test_i_p"]))
    all_datasets["PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n"] = pd.concat( (all_datasets["PhrasIS_test_h_p+PhrasIS_test_h_n"], all_datasets["PhrasIS_test_i_p+PhrasIS_test_i_n"]))

    train_datasets = list()
    train_datasets.append("PhrasIS_train_h_p")
    train_datasets.append("PhrasIS_train_i_p")
    train_datasets.append("PhrasIS_train_h_p+PhrasIS_train_h_n")
    train_datasets.append("PhrasIS_train_i_p+PhrasIS_train_i_n")
    train_datasets.append("PhrasIS_train_h_p+PhrasIS_train_i_p")
    train_datasets.append("PhrasIS_train_h_p+PhrasIS_train_i_p+PhrasIS_train_h_n+PhrasIS_train_i_n")

    test_datasets = list()
    test_datasets.append("PhrasIS_test_h_p")
    test_datasets.append("PhrasIS_test_i_p")
    test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_h_n")
    test_datasets.append("PhrasIS_test_i_p+PhrasIS_test_i_n")
    test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_i_p")
    test_datasets.append("PhrasIS_test_h_p+PhrasIS_test_i_p+PhrasIS_test_h_n+PhrasIS_test_i_n")

    for train_eval,test_eval in zip(train_datasets, test_datasets):
        train_df = all_datasets[train_eval]
        test_df = all_datasets[test_eval]

        train_src = train_df['left']
        train_trg = train_df['right']
        train_ref = train_df['STS']

        test_src = test_df['left']
        test_trg = test_df['right']
        test_ref = test_df['STS']

        # Augment training set based on symmetry
        train_src, train_trg = pd.concat([train_src , train_trg]), pd.concat([train_trg , train_src])
        train_ref = np.concatenate((train_ref, train_ref))

        # Extract features
        train_features = extract_features(train_src.values, train_trg.values, emb, word2ind)
        test_features1 = extract_features(test_src.values, test_trg.values, emb, word2ind)
        test_features2 = extract_features(test_trg.values, test_src.values, emb, word2ind)

        # Define models
        model_rf = sklearn.ensemble.RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=0.001, max_features=0.2, random_state=args.seed)
        model_gb = sklearn.ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=8, loss='ls', min_samples_split=0.01, subsample=0.5, max_features=0.9, min_impurity_decrease=0.0, random_state=args.seed)
        model_kr = sklearn.kernel_ridge.KernelRidge(alpha=0.4, kernel='laplacian')
        model_et = sklearn.ensemble.ExtraTreesRegressor(n_estimators=1000, max_depth=15, min_samples_split=0.0005, random_state=args.seed)
        models = [model_rf, model_gb, model_kr, model_et]

        for m in models:
            # Training
            m.fit(train_features, train_ref)

            # Evaluation
            r1 = m.predict(test_features1)
            r2 = m.predict(test_features2)
            test_sys = sum([r1 + r2])
            print(test_eval + " : " + str(utils.pearson(test_sys, test_ref)) + "Models : " + str(m))

if __name__ == '__main__':
    main()

