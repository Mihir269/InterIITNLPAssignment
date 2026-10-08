"""Write cache/hybrid_runs.json for Task 5: t3_hybrid rankings (val/eval) + plain BM25 rankings for training claims."""
import json

import numpy as np

from common import CACHE, evaluate, fmt, load_claims, load_corpus, qrels, topk_from_scores

docs, ids = load_corpus()
val, ev = load_claims("val"), load_claims("eval")
out = {sp: {c["id"]: r for c, r in zip(cl, topk_from_scores(np.load(f"{CACHE}/hybrid_{sp}.npy"), ids))}
       for sp, cl in [("val", val), ("eval", ev)]}
out["train"] = json.load(open(f"{CACHE}/bm25x_runs.json"))["train"]  # plain BM25: no leakage of the claim itself
print(fmt(evaluate({k: v[0] for k, v in out["val"].items()}, qrels(val))))
json.dump(out, open(f"{CACHE}/hybrid_runs.json", "w"))
