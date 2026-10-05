"""Model-free retrieval cache for Task 5 experiments: plain BM25 for train claims (no leakage),
BM25 over abstracts expanded with training claims for val/eval. Writes cache/bm25x_runs.json."""
import json
import os

from bm25 import BM25, make_tokenizer
from common import CACHE, doc_text, evaluate, fmt, load_claims, load_corpus, qrels, topk_from_scores

docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
tok = make_tokenizer()
T = [doc_text(d) for d in docs]
exp = [[] for _ in docs]
for c in tr:
    for e in c["evidence"]:
        exp[pos[int(e["doc_id"])]].append(c["claim"])
plain, bmx = BM25(T, tok, 1.2, 0.3), BM25([t + " " + " ".join(x) for t, x in zip(T, exp)], tok, 1.2, 0.3)
out = {}
for sp, cl, bm in [("train", tr, plain), ("val", val, bmx), ("eval", ev, bmx)]:
    out[sp] = {c["id"]: r for c, r in zip(cl, topk_from_scores(bm.score([c["claim"] for c in cl]), ids))}
print("val", fmt(evaluate({k: v[0] for k, v in out["val"].items()}, qrels(val))))
json.dump(out, open(os.path.join(CACHE, "bm25x_runs.json"), "w"))
