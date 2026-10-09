"""Task 3: hybrid of BM25 + dense (RRF / normalised score interpolation), then cross-encoder reranking.
Needs cache/ from t1 (recomputed here) and t2. Writes t3_hybrid.trec and t3_rerank.trec."""
import argparse
import itertools
import json
import os
import time

import numpy as np

from bm25 import BM25, make_tokenizer
from common import (CACHE, PRED, doc_text, evaluate, fmt, load_claims, load_corpus, qrels, topk_from_scores,
                    write_trec)
from models import CrossEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
ap.add_argument("--ks", default="10,20,30,50")
ap.add_argument("--docexp", type=int, default=1, help="1: also allow BM25 over abstracts expanded with train claims")
args = ap.parse_args()

docs, doc_ids = load_corpus()
pos = {d: i for i, d in enumerate(doc_ids)}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
qv = qrels(val)
best1 = json.load(open(os.path.join(CACHE, "t1_bm25.json")))["best"]
tok = make_tokenizer(best1["stem"], best1["stop"], best1["pattern"])
texts = [doc_text(d) for d in docs]
S = {"val": {}, "eval": {}}
bm = BM25(texts, tok, best1["k1"], best1["b"])
for sp, cl in [("val", val), ("eval", ev)]:
    S[sp]["bm25"] = bm.score([c["claim"] for c in cl])
if args.docexp:
    exp = [[] for _ in docs]
    for c in tr:
        for e in c["evidence"]:
            exp[pos[int(e["doc_id"])]].append(c["claim"])
    bmx = BM25([t + " " + " ".join(x) for t, x in zip(texts, exp)], tok, best1["k1"], best1["b"])
    for sp, cl in [("val", val), ("eval", ev)]:
        S[sp]["bm25x"] = bmx.score([c["claim"] for c in cl])
for kind in ["general", "scientific"]:
    for sp in ["val", "eval"]:
        S[sp][kind] = np.load(os.path.join(CACHE, f"{kind}_{sp}.npy"))


def zn(M):
    return (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True).clip(1e-9)


def mm(M, top=100):  # min-max over each claim's top-100
    t = -np.sort(-M, 1)[:, :top]
    return (M - t[:, -1:]) / (t[:, :1] - t[:, -1:]).clip(1e-9)


def rrf(Ms, ws, k):
    out = 0
    for M, w in zip(Ms, ws):
        out = out + w / (k + np.argsort(np.argsort(-M, 1), 1) + 1)
    return out


def fuse(sp, cfg):
    Ms = [S[sp][c] for c in cfg["comps"]]
    if cfg["method"] == "rrf":
        return rrf(Ms, cfg["w"], cfg["k"])
    norm = zn if cfg["method"] == "z" else mm
    return sum(w * norm(M) for M, w in zip(Ms, cfg["w"]))


def val_run(M):
    return {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(M, doc_ids))}


single = {c: evaluate(val_run(S["val"][c]), qv) for c in S["val"]}
for c, m in single.items():
    print(f"single {c:11s}", fmt(m))
lex = ["bm25"] + (["bm25x"] if args.docexp else [])
cands = []
grid = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
for l in lex:
    for comps in [[l, "general"], [l, "scientific"], [l, "general", "scientific"]]:
        for method in ["z", "minmax", "rrf"]:
            for ws in itertools.product(grid, repeat=len(comps) - 1):
                for k in ([10, 30, 60] if method == "rrf" else [None]):
                    cfg = dict(comps=comps, method=method, w=[1.0, *ws], k=k)
                    cands.append((evaluate(val_run(fuse("val", cfg)), qv), cfg))
cands.sort(key=lambda x: -x[0]["ndcg10"])
for m, cfg in cands[:8]:
    print("hybrid", fmt(m), cfg)
best_plain = next((m, c) for m, c in cands if "bm25x" not in c["comps"])
print("best hybrid without doc expansion:", fmt(best_plain[0]), best_plain[1])
hm, hcfg = cands[0]
H = {sp: fuse(sp, hcfg) for sp in ["val", "eval"]}
write_trec(os.path.join(PRED, "t3_hybrid.trec"), [c["id"] for c in ev], topk_from_scores(H["eval"], doc_ids), "hybrid")

# ---- choose the first stage to rerank: best val nDCG among single systems and the hybrid
first = max([(m["ndcg10"], c, S["val"][c], S["eval"][c]) for c, m in single.items()] +
            [(hm["ndcg10"], "hybrid", H["val"], H["eval"])], key=lambda x: x[0])
print("rerank first stage:", first[1], f"{first[0]:.4f}")
ce = CrossEncoder(args.reranker)
ks = [int(k) for k in args.ks.split(",")]
dtext = {d: doc_text(docs[pos[d]]) for d in doc_ids}


def rerank(cl, M, kmax):
    """returns per-claim first-stage top-100 docs and cross-encoder scores for the top kmax, plus latency"""
    base = topk_from_scores(M, doc_ids)
    out, lat = [], []
    for c, (dl, _) in zip(cl, base):
        t = time.time()
        sc = ce.score([c["claim"]] * kmax, [dtext[d] for d in dl[:kmax]])
        lat.append(time.time() - t)
        out.append((dl, sc))
    return out, float(np.mean(lat) * 1000)


def assemble(item, k):
    dl, sc = item
    o = np.argsort(-sc[:k], kind="stable")
    top = [dl[i] for i in o]
    scores = [float(sc[i]) for i in o]
    lo = min(scores)
    rest = dl[k:]
    return top + rest, scores + [lo - 1 - i * 1e-3 for i in range(len(rest))]


rv, _ = rerank(val, first[2], max(ks))
res = {}
for k in ks:
    res[k] = evaluate({c["id"]: assemble(it, k)[0] for c, it in zip(val, rv)}, qv)
    print(f"rerank k={k:3d}", fmt(res[k]))
kbest = max(ks, key=lambda k: res[k]["ndcg10"])
re_, _ = rerank(ev, first[3], kbest)
_, lat = rerank(ev[:50], first[3], kbest)  # latency at the chosen k
write_trec(os.path.join(PRED, "t3_rerank.trec"), [c["id"] for c in ev], [assemble(it, kbest) for it in re_], "rerank")
np.save(os.path.join(CACHE, "hybrid_val.npy"), H["val"]), np.save(os.path.join(CACHE, "hybrid_eval.npy"), H["eval"])
# cache the reranked val/eval runs (with scores) for Task 5
json.dump({"val": {c["id"]: assemble(it, kbest) for c, it in zip(val, rv)},
           "eval": {c["id"]: assemble(it, kbest) for c, it in zip(ev, re_)}},
          open(os.path.join(CACHE, "rerank_runs.json"), "w"))
json.dump(dict(single=single, hybrid=dict(cfg=hcfg, val=hm), hybrid_no_docexp=dict(cfg=best_plain[1], val=best_plain[0]),
               first_stage=first[1], reranker=args.reranker, reranker_params_m=ce.n_params / 1e6,
               rerank={str(k): v for k, v in res.items()}, k=kbest, rerank_latency_ms=lat),
          open(os.path.join(CACHE, "t3_stats.json"), "w"), indent=1)
print(f"chosen k={kbest}  rerank latency {lat:.1f} ms/query (cross-encoder only)")
