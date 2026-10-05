"""Sparse-matrix BM25 (Okapi, Lucene-style IDF) with configurable tokenization."""
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp
import Stemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stem = Stemmer.Stemmer("english")
_SPLIT = {"word": re.compile(r"[a-z0-9]+"), "hyphen": re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")}


def make_tokenizer(stem=True, stop=True, pattern="word"):
    rx = _SPLIT[pattern]

    def tok(text):
        t = rx.findall(text.lower())
        if pattern == "hyphen":  # keep the compound and its parts
            t = t + [p for w in t if "-" in w for p in w.split("-")]
        if stop:
            t = [w for w in t if w not in ENGLISH_STOP_WORDS]
        return _stem.stemWords(t) if stem else t
    return tok


class BM25:
    def __init__(self, texts, tokenizer, k1=0.9, b=0.4):
        self.tok, self.k1, self.b = tokenizer, k1, b
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            for w, c in Counter(tokenizer(t)).items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i), cols.append(j), vals.append(c)
        self.tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.vocab)), dtype=np.float32)
        self.dl = np.asarray(self.tf.sum(1)).ravel()
        df = np.bincount(self.tf.indices, minlength=len(self.vocab))
        n = len(texts)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.set_params(k1, b)

    def set_params(self, k1, b):
        self.k1, self.b = k1, b
        tf = self.tf.tocoo()
        norm = k1 * (1 - b + b * self.dl / self.dl.mean())
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # vocab x docs

    def query_matrix(self, queries):
        rows, cols, vals = [], [], []
        for i, q in enumerate(queries):
            for w, c in Counter(self.tok(q)).items():
                if w in self.vocab:
                    rows.append(i), cols.append(self.vocab[w]), vals.append(c)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)), dtype=np.float32)

    def score(self, queries):
        return np.asarray((self.query_matrix(queries) @ self.W).todense())
