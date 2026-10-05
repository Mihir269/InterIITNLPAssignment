"""Exploration that needs no pretrained weights: doc expansion with training claims,
RM3 feedback, LSA 'dense' retrieval, and BM25+LSA fusion. All tuned on validation only."""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from bm25 import BM25, make_tokenizer
from common import SEED, doc_text, evaluate, fmt, load_claims, load_corpus, per_claim, qrels, topk_from_scores

docs, doc_ids = load_corpus()
pos = {d: i for i, d in enumerate(doc_ids)}
tr, val = load_claims("train"), load_claims("val")
qv = qrels(val)
Q = [c["claim"] for c in val]
tok = make_tokenizer()


def run_of(S):
    return {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(S, doc_ids))}


def boot(a, b, n=5000):
    """paired bootstrap: P(run b is not better than a) on per-claim nDCG@10"""
    pa, pb = per_claim(a, qv), per_claim(b, qv)
    diff = np.array([pb[q] - pa[q] for q in qv])
    rng = np.random.default_rng(SEED)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return diff.mean(), (means <= 0).mean()


base_texts = [doc_text(d) for d in docs]
bm = BM25(base_texts, tok, 1.2, 0.3)
S_bm = bm.score(Q)
r_bm = run_of(S_bm)
print("BM25            ", fmt(evaluate(r_bm, qv)))

# ---- 1. document expansion: append training claims to their gold abstracts
exp = [[] for _ in docs]
for c in tr:
    for e in c["evidence"]:
        exp[pos[int(e["doc_id"])]].append(c["claim"])
print("docs expanded:", sum(1 for x in exp if x))
for rep in [1, 2]:
    bm_x = BM25([t + (" " + " ".join(x)) * rep for t, x in zip(base_texts, exp)], tok, 1.2, 0.3)
    r = run_of(bm_x.score(Q))
    print(f"BM25+docexp x{rep}  ", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(r_bm, r))
# separate field: score the training-claim field and interpolate
bm_f = BM25([" ".join(x) if x else "" for x in exp], tok, 1.2, 0.75)
S_f = bm_f.score(Q)
for w in [0.1, 0.2, 0.3, 0.5]:
    r = run_of(S_bm + w * S_f)
    print(f"BM25 + {w}*claimfield", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(r_bm, r))

# ---- 2. RM3 pseudo-relevance feedback
def rm3(S, fb_docs=5, fb_terms=10, w_orig=0.7):
    W = bm.tf.multiply(1 / bm.tf.sum(1)).tocsr()  # p(t|d)
    out = np.zeros_like(S)
    qm = bm.query_matrix(Q).toarray()
    qm = qm / qm.sum(1, keepdims=True).clip(1e-9)
    for i, row in enumerate(S):
        top = np.argsort(-row)[:fb_docs]
        sc = np.exp(row[top] - row[top].max())
        fb = np.asarray(W[top].multiply(sc[:, None]).sum(0)).ravel()
        keep = np.argsort(-fb)[:fb_terms]
        v = np.zeros_like(fb); v[keep] = fb[keep] / fb[keep].sum()
        newq = w_orig * qm[i] + (1 - w_orig) * v
        out[i] = np.asarray(bm.W.T @ newq).ravel()
    return out
for fd, ft, wo in [(3, 10, 0.7), (5, 10, 0.7), (5, 20, 0.8), (10, 20, 0.8), (3, 5, 0.9)]:
    r = run_of(rm3(S_bm, fd, ft, wo))
    print(f"RM3 fd={fd} ft={ft} w={wo}", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(r_bm, r))

# ---- 3. LSA as a no-download 'dense' retriever, and fusion with BM25
tf = TfidfVectorizer(tokenizer=tok, lowercase=False, sublinear_tf=True, min_df=2, token_pattern=None)
X = tf.fit_transform(base_texts + [c["claim"] for c in tr])
for dim in [256]:
    svd = TruncatedSVD(dim, random_state=SEED).fit(X)
    D = svd.transform(tf.transform(base_texts)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    q = svd.transform(tf.transform(Q)); q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    S_lsa = q @ D.T
    r_lsa = run_of(S_lsa)
    print(f"LSA-{dim}         ", fmt(evaluate(r_lsa, qv)))


def zn(S):
    return (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True).clip(1e-9)


def rrf(*Ss, k=60):
    out = np.zeros_like(Ss[0])
    for S in Ss:
        rk = np.argsort(np.argsort(-S, 1), 1)
        out += 1 / (k + rk + 1)
    return out
for a in [0.1, 0.2, 0.3, 0.5]:
    r = run_of(zn(S_bm) + a * zn(S_lsa))
    print(f"BM25+{a}*LSA (z)  ", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(r_bm, r))
r = run_of(rrf(S_bm, S_lsa))
print("RRF(BM25,LSA)    ", fmt(evaluate(r, qv)), "Δ=%.4f p=%.3f" % boot(r_bm, r))
