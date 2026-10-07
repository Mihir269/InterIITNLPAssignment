"""Why did the cross-encoder hurt, and what fixes it? On validation, rerank the hybrid's top-K with several
cross-encoders, with plain vs training-claim-expanded document text, and with score fusion
(minmax(first stage) + w * minmax(cross-encoder)) instead of pure reranker order. Scores are cached."""
import json
import os
import sys
import time

import numpy as np

from common import CACHE, SEED, doc_text, evaluate, fmt, load_claims, load_corpus, per_claim, qrels
from models import CrossEncoder

K = 30
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "cross-encoder/ms-marco-MiniLM-L-6-v2", "ncbi/MedCPT-Cross-Encoder", "BAAI/bge-reranker-base"]
docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val = load_claims("train"), load_claims("val")
qv = qrels(val)
H = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))["val"]
exp = {d: [] for d in ids.tolist()}
for c in tr:
    for e in c["evidence"]:
        exp[int(e["doc_id"])].append(c["claim"])
text = {"plain": lambda d: doc_text(docs[pos[d]]),
        "expanded": lambda d: (" ".join(exp[d]) + " | " if exp[d] else "") + doc_text(docs[pos[d]])}
base = {c["id"]: H[str(c["id"])][0] for c in val}
mb = evaluate(base, qv)
print("hybrid first stage", fmt(mb))


def boot(a, b, n=5000):
    pa, pb = per_claim(a, qv), per_claim(b, qv)
    d = np.array([pb[q] - pa[q] for q in qv])
    m = d[np.random.default_rng(SEED).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), (m <= 0).mean()


def mm(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / max(x.max() - x.min(), 1e-9)


summary = {}
for name in MODELS:
    ce = None
    for variant in ["plain", "expanded"]:
        path = os.path.join(CACHE, f"ce_{name.replace('/', '_')}_{variant}_val{K}.json")
        if os.path.exists(path):
            S = json.load(open(path))
        else:
            ce = ce or CrossEncoder(name)
            S, t0 = {}, time.time()
            for c in val:
                dl = base[c["id"]][:K]
                S[str(c["id"])] = ce.score([c["claim"]] * K, [text[variant](d) for d in dl]).tolist()
            print(f"  {name} {variant}: {(time.time() - t0) / len(val) * 1000:.0f} ms/query for k={K}")
            json.dump(S, open(path, "w"))
        for k in [10, 20, 30]:
            for w in [None, 0.25, 0.5, 1.0, 2.0]:
                run = {}
                for c in val:
                    dl, fs = base[c["id"]], H[str(c["id"])][1]
                    ce_s = np.array(S[str(c["id"])][:k])
                    sc = ce_s if w is None else mm(fs[:k]) + w * mm(ce_s)
                    o = np.argsort(-sc, kind="stable")
                    run[c["id"]] = [dl[i] for i in o] + dl[k:]
                m = evaluate(run, qv)
                d, p = boot(base, run)
                summary[f"{name}|{variant}|k={k}|w={w}"] = dict(ndcg10=m["ndcg10"], delta=d, p=p)
        best = max((v["ndcg10"], k) for k, v in summary.items() if k.startswith(f"{name}|{variant}|"))
        pure = max((v["ndcg10"], k) for k, v in summary.items() if k.startswith(f"{name}|{variant}|") and k.endswith("w=None"))
        print(f"{name:40s} {variant:8s} pure-rerank best {pure[0]:.4f} ({pure[1].split('|')[2]})   fused best {best[0]:.4f} ({best[1]})")
json.dump(summary, open(os.path.join(CACHE, "explore_rerank.json"), "w"), indent=1)
top = sorted(summary.items(), key=lambda x: -x[1]["ndcg10"])[:8]
for k, v in top:
    print(f"{v['ndcg10']:.4f}  Δ={v['delta']:+.4f} p={v['p']:.3f}  {k}")
