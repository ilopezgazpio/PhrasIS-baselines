# (c) Mikel Artetxe

import nltk
import nltk.corpus
import nltk.stem
import nltk.translate.gleu_score
import nltk.translate.chrf_score
import numpy as np
import string

# Load stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))

# Load stemmer
stemmer = nltk.stem.SnowballStemmer('english')


def read_embeddings(file, threshold=0, vocabulary=None):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim)) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ')
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' '))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix))


def length_normalize_embeddings(matrix):
    norms = np.sqrt(np.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    return matrix / norms[:, np.newaxis]


def read_data(file):
    src, trg, ref = [], [], []
    for line in file:
        cols = line.split('\t')
        src.append(cols[5])
        trg.append(cols[6])
        ref.append(float(cols[4]))
    return src, trg, np.array(ref)


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def truecase(sent, word2ind):
    sent = sent[:]
    original = sent[0]
    lower = original.lower()
    if lower in word2ind and (not original in word2ind or word2ind[lower] < word2ind[original]):
        sent[0] = lower
    return sent


def fixcase(sent, vocab):
    return [word.lower() if not word in vocab and word.lower() in vocab else word for word in sent]


def recase(sent, word2ind):
    ans = []
    for word in sent:
        if word in word2ind and word.lower() in word2ind:
            if word2ind[word.lower()] < word2ind[word]:
                ans.append(word.lower())
            else:
                ans.append(word)
        elif word.lower() in word2ind:
            ans.append(word.lower())
        else:
            ans.append(word)
    return ans


def pos_tag(sentence):
    return [tag for word, tag in nltk.pos_tag(sentence)]


def strip_punctuation(sentence):
    return [word for word in sentence if not all([c in string.punctuation for c in word])]


def remove_stopwords(sentence):
    return [word for word in sentence if word not in stopwords]


def remove_frequent(sentence, word2ind, threshold):
    return [word for word in sentence if word not in word2ind or word2ind[word] >= threshold]


def remove_oovs(sentence, vocab):
    return [word for word in sentence if word in vocab]


def nouns(sentence):
    return [word for word, tag in nltk.pos_tag(sentence) if tag[:2] == 'NN']


def verbs(sentence):
    return [word for word, tag in nltk.pos_tag(sentence) if tag[:2] == 'VB']


def adjectives(sentence):
    return [word for word, tag in nltk.pos_tag(sentence) if tag[:2] == 'JJ']


def adverbs(sentence):
    return [word for word, tag in nltk.pos_tag(sentence) if tag[:2] == 'RB']


def stem(sentence):
    return [stemmer.stem(word) for word in sentence]


def pearson(sys, ref):
    return np.corrcoef(sys, ref)[0, 1]


def cosine(a, b):
    return a.dot(b) / np.sqrt(a.dot(a)*b.dot(b))


def centroid(sent, emb, word2ind):
    return sum([emb[word2ind[word]] for word in sent]) / len(sent)


def centroid_cosine(src, trg, emb, word2ind, backoff=1.0):
    src = remove_oovs(src, word2ind)
    trg = remove_oovs(trg, word2ind)
    if len(src) == 0 or len(trg) == 0:
        return backoff
    else:
        return cosine(centroid(src, emb, word2ind), centroid(trg, emb, word2ind))


def align_score(src, trg, emb, word2ind, backoff=1.0):
    src = remove_oovs(src, word2ind)
    trg = remove_oovs(trg, word2ind)
    if len(src) == 0 or len(trg) == 0:
        return backoff
    else:
        ind_src = [word2ind[word] for word in src]
        ind_trg = [word2ind[word] for word in trg]
        sim = emb[ind_src,].dot(emb[ind_trg,].T)
        exp1 = emb[ind_src,].dot(emb[ind_src,].T)
        exp2 = emb[ind_trg,].dot(emb[ind_trg,].T)
        aux = np.sqrt(exp1.max(axis=0)).mean() * np.sqrt(exp2.max(axis=0)).mean()
        return (sim.max(axis=0).mean()/2 + sim.max(axis=1).mean()/2) / aux


def bow_cosine(src, trg, backoff=1.0):
    if len(src) == 0 or len(trg) == 0:
        return backoff
    else:
        vocab = set(src + trg)
        word2ind = {word: i for i, word in enumerate(vocab)}
        a = np.zeros(len(vocab))
        b = np.zeros(len(vocab))
        for word in src:
            a[word2ind[word]] += 1
        for word in trg:
            b[word2ind[word]] += 1
        return cosine(a, b)


def overlap(a, b):
    return 1.0 if len(a) == 0 else len([x for x in a if x in b]) / len(a)


def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


def gleu(sys, ref):
    return nltk.translate.gleu_score.sentence_gleu([ref], sys)


def chrf(sys, ref):
    return 1.0 if len(sys) == 0 or len(ref) == 0 else nltk.translate.chrf_score.sentence_chrf(ref, sys)

