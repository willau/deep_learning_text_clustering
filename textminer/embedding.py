# -*- coding: utf-8 -*-

import numpy as np
from string import punctuation
from operator import itemgetter
from collections import Counter
from tqdm import tqdm, tqdm_notebook
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

PUNCT = punctuation + "\t\n"
EPSILON = 1e-6
STR_TYPES = (str, np.str, np.str0, np.str_, np.string_)
INT_TYPES = (int, np.int, np.int_, np.int8, np.int16, np.int32, np.int64)


def get_sorted_voc(corpus):
    if not isinstance(corpus, STR_TYPES):
        corpus = " ".join(corpus)
    voc_raw = text_to_word_sequence(corpus, filters=PUNCT)
    voc_cnt = tuple(Counter(voc_raw).items())
    voc_sort = sorted(voc_cnt, key=itemgetter(1), reverse=True)
    voc = [tup[0] for tup in voc_sort]
    return voc


class Embeddor:

    def __init__(self, notebook_display=False):
        self.dim = 0
        self.n_global = 0
        self.emb_dim = 0
        self.emb = list()
        self.word_to_idx = dict()
        self.idx_to_word = dict()
        self.pad_dim = 0
        self.bar = tqdm
        if notebook_display:
            self.bar = tqdm_notebook

    def load_emb(self, src, n_emb):
        # read embedding file
        emb = list()
        with open(src, 'rb') as f:
            self.n_global, self.dim = map(int, f.readline().split())
            emb.append(np.zeros(shape=(self.dim,), dtype='float32'))
            for idx in self.bar(range(n_emb)):
                raw = f.readline().decode('utf-8').rstrip(' \n')
                line = raw.split(' ')
                word = line[0]
                vect = np.asarray(line[1:], dtype='float32')
                emb.append(vect)
                self.word_to_idx[word] = idx + 1

        # index to word dict
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.idx_to_word[0] = "_PAD"

        # embeddings
        self.emb_dim = np.size(emb[0])
        emb = np.vstack(emb)
        emb_norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        self.emb = emb / (emb_norm + EPSILON)
        return self.emb

    def get_emb(self, idx):
        if isinstance(idx, INT_TYPES):
            return self.emb[idx]
        elif isinstance(idx, STR_TYPES):
            if idx in self.word_to_idx:
                idx = self.word_to_idx[idx]
                return self.emb[idx]
            else:
                return None
        else:
            return None

    def get_idx(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else:
            return 0

    def to_seq(self, sent, padding=True):
        def convert_to_seq(words):
            seq_word = text_to_word_sequence(words, filters=PUNCT)
            return [self.get_idx(word) for word in seq_word]

        if isinstance(sent, STR_TYPES):
            sent = [sent]

        seq_idxs = list()
        for s in sent:
            seq_idxs.append(convert_to_seq(s))

        if padding:
            seq_idxs = pad_sequences(seq_idxs, padding='post')
            self.pad_dim = np.shape(seq_idxs)[1]

        return seq_idxs

    def to_words(self, seq, join=True):
        def convert_to_words(sequences):
            return [self.idx_to_word[i] for i in sequences]

        if np.size(seq):
            seq = [seq]

        sent = list()
        for s in seq:
            words = convert_to_words(s)
            if join:
                sent.append(" ".join(words))
            else:
                sent.append(words)

        return sent

    def most_similar(self, word, n_top=10):
        emb = self.get_emb(word)
        if emb is None:
            return [(word, 1.0)]
        else:
            emb = emb / np.linalg.norm(emb)
            cos_sim = np.dot(self.emb, emb)
            idxs = np.argsort(cos_sim)[::-1][:n_top]
            return [(self.idx_to_word[idx], cos_sim[idx]) for idx in idxs]
