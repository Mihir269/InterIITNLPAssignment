"""Task 3 rerank (final): MedCPT cross-encoder over the hybrid's top-k, document text = training claims citing
the doc + title + abstract (scores from ce_scores.py). k chosen on val; ranks k+1..100 keep hybrid order."""
import json
import os

import numpy as np

from common import CACHE, PRED, SEED, evaluate, fmt, load_claims, per_claim, qrels, write_trec

MODEL = "ncbi/MedCPT-Cross-Encoder"
val, ev = load_claims("val"), load_claims("eval")
qv = qrels(val)
H = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))
f = lambda sp: os.path.join(CACHE, f"ce_{MODEL.replace('/', '_')}_expanded_{sp}30.json")
CE = {sp: json.load(open(f(sp))) for sp in ["val", "eval"]}


def assemble(sp, cid, k):
    dl = H[sp][str(cid)][0]
    sc = np.array(CE[sp][str(cid)][:k])
    o = np.argsort(-sc, kind="stable")
    top, scores = [dl[i] for i in o], [float(sc[i]) for i in o]
    rest = dl[k:100]
    return top + rest, scores + [scores[-1] - 1 - i * 1e-3 for i in range(len(rest))]


base = {c["id"]: H["val"][str(c["id"])][0] for c in val}
res = {k: evaluate({c["id"]: assemble("val", c["id"], k)[0] for c in val}, qv) for k in [10, 20, 30]}
for k, m in res.items():
    print(f"k={k}", fmt(m))
k = max(res, key=lambda k: res[k]["ndcg10"])
rr = {c["id"]: assemble("val", c["id"], k)[0] for c in val}
pa, pb = per_claim(base, qv), per_claim(rr, qv)
d = np.array([pb[q] - pa[q] for q in qv])
p = (d[np.random.default_rng(SEED).integers(0, len(d), (5000, len(d)))].mean(1) <= 0).mean()
print(f"chosen k={k}: Δ vs hybrid {d.mean():+.4f} (bootstrap p={p:.3f}); claims improved {int((d > 0).sum())}, hurt {int((d < 0).sum())}")
write_trec(os.path.join(PRED, "t3_rerank.trec"), [c["id"] for c in ev], [assemble("eval", c["id"], k) for c in ev], "rerank")
lat = json.load(open(f("eval").replace(".json", "_latency.json")))["ms_per_query"] * k / 30
st = json.load(open(os.path.join(CACHE, "t3_stats.json")))
st.update(reranker=MODEL + " (doc text = citing training claims | title + abstract)", k=k,
          rerank={str(kk): v for kk, v in res.items()}, rerank_latency_ms=lat, rerank_delta_vs_hybrid=float(d.mean()), rerank_p=float(p),
          rerank_v1=dict(model="BAAI/bge-reranker-base", k=10, val_ndcg10=0.8701, latency_ms=2529.9))
json.dump(st, open(os.path.join(CACHE, "t3_stats.json"), "w"), indent=1)
print(f"latency ≈ {lat:.0f} ms/query (CPU, cross-encoder only)")
