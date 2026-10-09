"""More weight-free ideas: sentence-level BM25 (max over sentences), title field, and a learning-to-rank
model (logistic regression on pairwise-free pointwise features) trained on training claims, reranking top-50."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import numpy as np
from sklearn.linear_model import LogisticRegression

from bm25 import BM25, make_tokenizer
from common import SEED, doc_text, evaluate, fmt, load_claims, load_corpus, per_claim, qrels, topk_from_scores

docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val = load_claims("train"), load_claims("val")
qv, qt = qrels(val), qrels(tr)
tok = make_tokenizer()
T = [doc_text(d) for d in docs]
sent_doc = np.array([i for i, d in enumerate(docs) for _ in d["abstract"]])
sents = [s for d in docs for s in d["abstract"]]
bm_doc = BM25(T, tok, 1.2, 0.3)
bm_sent = BM25(sents, tok, 1.2, 0.3)
bm_title = BM25([d["title"] for d in docs], tok, 1.2, 0.3)
exp = [[] for _ in docs]
for c in tr:
    for e in c["evidence"]:
        exp[pos[int(e["doc_id"])]].append(c["claim"])


def max_sent(S):
    out = np.full((S.shape[0], len(docs)), 0.0, dtype=np.float32)
    for i in range(S.shape[0]):
        np.maximum.at(out[i], sent_doc, S[i])
    return out


def feats(Q, train_mode):
    """train_mode: the doc-expansion field must not contain the query's own claim -> leave-one-out via rebuild"""
    F = {"doc": bm_doc.score(Q), "sent": max_sent(bm_sent.score(Q)), "title": bm_title.score(Q)}
    return F


def boot(a, b, n=5000):
    pa, pb = per_claim(a, qv), per_claim(b, qv)
    d = np.array([pb[q] - pa[q] for q in qv])
    m = d[np.random.default_rng(SEED).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), (m <= 0).mean()


Qv = [c["claim"] for c in val]
Fv = feats(Qv, False)
run = lambda M: {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(M, ids))}
base = run(Fv["doc"])
print("doc BM25        ", fmt(evaluate(base, qv)))
print("max-sentence    ", fmt(evaluate(run(Fv["sent"]), qv)))
print("title only      ", fmt(evaluate(run(Fv["title"]), qv)))
z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True).clip(1e-9)
for a in [0.25, 0.5, 1.0]:
    r = run(z(Fv["doc"]) + a * z(Fv["sent"]))
    print(f"doc + {a}*sent   ", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(base, r))
for a in [0.1, 0.25, 0.5]:
    r = run(z(Fv["doc"]) + a * z(Fv["title"]))
    print(f"doc + {a}*title  ", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(base, r))

# ---- LTR: features over the top-50 of doc BM25, trained on training claims with evidence
def ltr_feats(F, i, cand):
    d, s, t = F["doc"][i], F["sent"][i], F["title"][i]
    top = d[cand[0]]
    return np.stack([d[cand], d[cand] - top, s[cand], s[cand] / (d[cand] + 1e-9), t[cand],
                     np.log1p(np.arange(len(cand))), np.array([len(docs[j]["abstract"]) for j in cand])], 1)


trc = [c for c in tr if c["evidence"]]
Ft = feats([c["claim"] for c in trc], True)
X, y = [], []
for i, c in enumerate(trc):
    cand = np.argsort(-Ft["doc"][i])[:50]
    gold = {pos[d] for d in qt[c["id"]]}
    X.append(ltr_feats(Ft, i, cand))
    y += [j in gold for j in cand]
X = np.concatenate(X)
mu, sd = X.mean(0), X.std(0) + 1e-9
clf = LogisticRegression(max_iter=5000, C=1.0).fit((X - mu) / sd, y)
print("LTR coefficients (doc, doc-top, sent, sent/doc, title, logrank, nsent):", np.round(clf.coef_[0], 2))
rl = {}
for i, c in enumerate(val):
    cand = np.argsort(-Fv["doc"][i])[:50]
    p = clf.decision_function((ltr_feats(Fv, i, cand) - mu) / sd)
    o = np.argsort(-p, kind="stable")
    rest = [ids[j] for j in np.argsort(-Fv["doc"][i])[50:100]]
    rl[c["id"]] = [ids[cand[k]] for k in o] + rest
print("LTR rerank top50", fmt(evaluate(rl, qv)), "Δ=%.4f p=%.3f" % boot(base, rl))
