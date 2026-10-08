"""Score each claim's hybrid top-K with a cross-encoder over (training claims citing the doc | title + abstract)
and cache {split: {claim_id: [scores aligned with hybrid ranks 1..K]}} in cache/ce_<model>_expanded_<split>{K}.json.
Used by Task 3 (t3_rerank.trec) and Task 5 (evidence selection)."""
import argparse
import json
import os
import time

from common import CACHE, doc_text, load_claims, load_corpus
from models import CrossEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ncbi/MedCPT-Cross-Encoder")
ap.add_argument("--k", type=int, default=30)
ap.add_argument("--splits", default="val,eval")
args = ap.parse_args()
docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
exp = {d: [] for d in ids.tolist()}
for c in load_claims("train"):
    for e in c["evidence"]:
        exp[int(e["doc_id"])].append(c["claim"])
text = lambda d: (" ".join(exp[d]) + " | " if exp[d] else "") + doc_text(docs[pos[d]])
H = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))
ce = CrossEncoder(args.model)
for sp in args.splits.split(","):
    path = os.path.join(CACHE, f"ce_{args.model.replace('/', '_')}_expanded_{sp}{args.k}.json")
    if os.path.exists(path):
        print("exists", path)
        continue
    S, lat = {}, []
    for c in load_claims(sp):
        dl = H[sp][str(c["id"])][0][:args.k]
        t = time.time()
        S[str(c["id"])] = ce.score([c["claim"]] * len(dl), [text(d) for d in dl]).tolist()
        lat.append(time.time() - t)
    json.dump(S, open(path, "w"))
    json.dump({"ms_per_query": 1000 * sum(lat) / len(lat), "k": args.k}, open(path.replace(".json", "_latency.json"), "w"))
    print(sp, f"{1000 * sum(lat) / len(lat):.0f} ms/query")
