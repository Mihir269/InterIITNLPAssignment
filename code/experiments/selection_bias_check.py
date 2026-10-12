"""How optimistic are the validation numbers? Re-select retrieval settings with 5-fold CV inside validation
(choose on 4/5 of the claims, score the held-out 1/5) and compare with the in-sample choice reported everywhere."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import itertools
import json
import os

import numpy as np

from bm25 import BM25, make_tokenizer
from common import CACHE, doc_text, load_claims, load_corpus, per_claim, qrels, topk_from_scores

docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val = load_claims("train"), load_claims("val")
qv = qrels(val)
tok = make_tokenizer()
T = [doc_text(d) for d in docs]
exp = [[] for _ in docs]
for c in tr:
    for e in c["evidence"]:
        exp[pos[int(e["doc_id"])]].append(c["claim"])
Q = [c["claim"] for c in val]
S = {"bm25": BM25(T, tok, 1.2, 0.3).score(Q), "bm25x": BM25([t + " " + " ".join(x) for t, x in zip(T, exp)], tok, 1.2, 0.3).score(Q),
     "general": np.load(os.path.join(CACHE, "general_val.npy")), "scientific": np.load(os.path.join(CACHE, "scientific_val.npy"))}


def mm(M):
    t = -np.sort(-M, 1)[:, :100]
    return (M - t[:, -1:]) / (t[:, :1] - t[:, -1:]).clip(1e-9)


N = {k: mm(v) for k, v in S.items()}
grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
configs = [(lex, wg, ws) for lex in ["bm25", "bm25x"] for wg, ws in itertools.product(grid, [0.0] + grid)]
qids = [c["id"] for c in val]
pcs = {}  # per-claim nDCG for every config
for cfg in configs:
    lex, wg, ws = cfg
    M = N[lex] + wg * N["general"] + ws * N["scientific"]
    run = {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(M, ids))}
    pcs[cfg] = per_claim(run, qv)
ev_ids = [q for q in qids if q in qv]
mean = lambda cfg, sub: np.mean([pcs[cfg][q] for q in sub])
best_in = max(configs, key=lambda c: mean(c, ev_ids))
print(f"hybrid: {len(configs)} configs; in-sample best {best_in} -> {mean(best_in, ev_ids):.4f}")
res = []
for seed in range(20):  # repeat the 5-fold split 20 times to average out the split
    folds = np.array_split(np.random.default_rng(seed).permutation(ev_ids), 5)
    out = []
    for k in range(5):
        te = list(folds[k]); trn = [q for j in range(5) if j != k for q in folds[j]]
        b = max(configs, key=lambda c: mean(c, trn))
        out += [pcs[b][q] for q in te]
    res.append(np.mean(out))
print(f"hybrid: 5-fold CV (20 repeats) {np.mean(res):.4f} ± {np.std(res):.4f}  -> optimism {mean(best_in, ev_ids) - np.mean(res):+.4f}")

# rerank depth k: in-sample choice vs CV
HY = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))["val"]
CE = json.load(open(os.path.join(CACHE, "ce_.._cache_medcpt-ce-ft_expanded_val30.json")))
pk = {}
for k in [10, 20, 30]:
    run = {}
    for c in val:
        dl = HY[str(c["id"])][0]
        o = np.argsort(-np.array(CE[str(c["id"])][:k]), kind="stable")
        run[c["id"]] = [dl[i] for i in o] + dl[k:]
    pk[k] = per_claim(run, qv)
kin = max(pk, key=lambda k: np.mean([pk[k][q] for q in ev_ids]))
res = []
for seed in range(20):
    folds = np.array_split(np.random.default_rng(seed).permutation(ev_ids), 5)
    out = []
    for f in range(5):
        te = list(folds[f]); trn = [q for j in range(5) if j != f for q in folds[j]]
        b = max(pk, key=lambda k: np.mean([pk[k][q] for q in trn]))
        out += [pk[b][q] for q in te]
    res.append(np.mean(out))
print(f"rerank: in-sample k={kin} -> {np.mean([pk[kin][q] for q in ev_ids]):.4f}; 5-fold CV {np.mean(res):.4f} ± {np.std(res):.4f}")
