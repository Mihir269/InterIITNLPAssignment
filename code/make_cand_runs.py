"""Candidate runs for Task 5: each val/eval claim's hybrid top-30 re-ordered by the t3_rerank cross-encoder (the submitted
best retriever), then hybrid ranks 31-100; training claims keep their plain BM25 runs. Writes cache/rerank_cand_runs.json."""
import json

import numpy as np

from common import CACHE, evaluate, fmt, load_claims, qrels

H = json.load(open(f"{CACHE}/hybrid_runs.json"))
out = {"train": H["train"]}
for sp in ["val", "eval"]:
    S = json.load(open(f"{CACHE}/ce_.._cache_medcpt-ce-ft_expanded_{sp}30.json"))
    out[sp] = {}
    for cid, sc in S.items():
        dl, hs = H[sp][cid]
        o = np.argsort(-np.array(sc), kind="stable")
        out[sp][cid] = [[dl[i] for i in o] + dl[30:], [float(sc[i]) for i in o] + hs[30:]]
val = load_claims("val")
print("val", fmt(evaluate({int(k): v[0] for k, v in out["val"].items()}, qrels(val))))
json.dump(out, open(f"{CACHE}/rerank_cand_runs.json", "w"))
