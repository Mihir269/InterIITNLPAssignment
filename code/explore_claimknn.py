"""Dense claim-to-claim kNN: score(doc) = max cosine(claim, training claims whose gold evidence is doc).
A dense analogue of the BM25 document expansion. Fused into the hybrid; bootstrap vs submitted hybrid."""
import itertools
import os

import numpy as np
import torch

from bm25 import BM25, make_tokenizer
from common import CACHE, SEED, doc_text, evaluate, fmt, load_claims, load_corpus, per_claim, qrels, topk_from_scores
from models import BiEncoder

torch.set_num_threads(2)
docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
qv = qrels(val)
enc = BiEncoder("BAAI/bge-base-en-v1.5")
# training claims are encoded as *queries* too: claim-vs-claim symmetric similarity
E = {sp: enc.encode_queries([c["claim"] for c in cl]) for sp, cl in [("train", tr), ("val", val), ("eval", ev)]}
cite = [(i, pos[int(e["doc_id"])]) for i, c in enumerate(tr) for e in c["evidence"]]


def knn_scores(Q):
    sim = Q @ E["train"].T
    out = np.full((len(Q), len(docs)), -1.0, dtype=np.float32)
    for i, j in cite:
        out[:, j] = np.maximum(out[:, j], sim[:, i])
    return out


K = {sp: knn_scores(E[sp]) for sp in ["val", "eval"]}
np.save(os.path.join(CACHE, "claimknn_val.npy"), K["val"]), np.save(os.path.join(CACHE, "claimknn_eval.npy"), K["eval"])
tok = make_tokenizer()
T = [doc_text(d) for d in docs]
exp = [[] for _ in docs]
for c in tr:
    for e in c["evidence"]:
        exp[pos[int(e["doc_id"])]].append(c["claim"])
Q = [c["claim"] for c in val]
S = {"bm25": BM25(T, tok, 1.2, 0.3).score(Q), "bm25x": BM25([t + " " + " ".join(x) for t, x in zip(T, exp)], tok, 1.2, 0.3).score(Q),
     "general": np.load(os.path.join(CACHE, "general_val.npy")), "scientific": np.load(os.path.join(CACHE, "scientific_val.npy")),
     "knn": K["val"]}


def mm(M):
    t = -np.sort(-M, 1)[:, :100]
    return (M - t[:, -1:]) / (t[:, :1] - t[:, -1:]).clip(1e-9)


N = {k: mm(v) for k, v in S.items()}
N["knn"] = np.clip(S["knn"], 0, None)  # raw cosine (docs never cited stay at 0)
run = lambda M: {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(M, ids))}


def boot(a, b, n=5000):
    pa, pb = per_claim(a, qv), per_claim(b, qv)
    d = np.array([pb[q] - pa[q] for q in qv])
    m = d[np.random.default_rng(SEED).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), (m <= 0).mean()


ref = run(N["bm25x"] + 0.5 * N["general"] + 0.25 * N["scientific"])
print("submitted hybrid ", fmt(evaluate(ref, qv)))
print("claim-kNN alone  ", fmt(evaluate(run(S["knn"]), qv)))
res = []
for lex, wg, ws, wk, th in itertools.product(["bm25", "bm25x"], [0.25, 0.5, 1.0], [0, 0.25, 0.5], [0.25, 0.5, 1.0, 2.0], [0.0, 0.8, 0.85, 0.9]):
    kn = np.where(S["knn"] >= th, N["knn"], 0)
    r = run(N[lex] + wg * N["general"] + ws * N["scientific"] + wk * kn)
    res.append((evaluate(r, qv)["ndcg10"], lex, wg, ws, wk, th, r))
res.sort(key=lambda x: -x[0])
for nd, lex, wg, ws, wk, th, r in res[:5]:
    print(f"{lex}+{wg}gen+{ws}sci+{wk}knn(th={th}) nDCG@10={nd:.4f} Δ=%+.4f p=%.3f" % boot(ref, r))
bp = next(x for x in res if x[1] == "bm25")
print(f"best without BM25 expansion: {bp[0]:.4f} ({bp[1:6]})  vs 0.9146 before   Δ vs submitted=%+.4f" % boot(ref, bp[6])[0])
