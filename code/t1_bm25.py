"""Task 1: BM25 grid search on validation, write t1_bm25.trec (eval) and val_bm25.trec."""
import itertools
import json
import os
import time

from bm25 import BM25, make_tokenizer
from common import (CACHE, PRED, doc_text, evaluate, fmt, load_claims, load_corpus, qrels,
                    topk_from_scores, write_trec)

docs, doc_ids = load_corpus()
val, ev = load_claims("val"), load_claims("eval")
qv = qrels(val)

results = []
for stem, stop, pattern, title in itertools.product([True, False], [True, False], ["word", "hyphen"], [True, False]):
    texts = [doc_text(d) if title else " ".join(d["abstract"]) for d in docs]
    bm = BM25(texts, make_tokenizer(stem, stop, pattern))
    for k1, b in itertools.product([0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0], [0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]):
        bm.set_params(k1, b)
        S = bm.score([c["claim"] for c in val])
        run = {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(S, doc_ids))}
        m = evaluate(run, qv)
        results.append(dict(stem=stem, stop=stop, pattern=pattern, title=title, k1=k1, b=b, **m))
results.sort(key=lambda r: -r["ndcg10"])
for r in results[:10]:
    print(r)
default = [r for r in results if r["stem"] and r["stop"] and r["pattern"] == "word" and r["title"]
           and r["k1"] == 0.9 and r["b"] == 0.4][0]
print("Pyserini-default k1=0.9 b=0.4:", default)
for key in ["stem", "stop", "pattern", "title"]:  # marginal effect of each option
    for v in sorted({r[key] for r in results}, key=str):
        print(f"  best with {key}={v}: {max(r['ndcg10'] for r in results if r[key] == v):.4f}")

best = results[0]
texts = [doc_text(d) if best["title"] else " ".join(d["abstract"]) for d in docs]
bm = BM25(texts, make_tokenizer(best["stem"], best["stop"], best["pattern"]), best["k1"], best["b"])
t0 = time.time()
Se = bm.score([c["claim"] for c in ev])
lat = (time.time() - t0) / len(ev) * 1000
write_trec(os.path.join(PRED, "t1_bm25.trec"), [c["id"] for c in ev], topk_from_scores(Se, doc_ids), "bm25")
Sv = bm.score([c["claim"] for c in val])
rv = topk_from_scores(Sv, doc_ids)
write_trec(os.path.join(PRED, "val_bm25.trec"), [c["id"] for c in val], rv, "bm25")
print("BEST", fmt(evaluate({c["id"]: d for c, (d, _) in zip(val, rv)}, qv)), f"latency {lat:.2f} ms/query")
json.dump({"best": best, "default": default, "latency_ms": lat, "top10": results[:10]},
          open(os.path.join(CACHE, "t1_bm25.json"), "w"), indent=1)
