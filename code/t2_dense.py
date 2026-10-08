"""Task 2: dense retrieval with a general and a scientific bi-encoder.
Writes t2_general.trec, t2_scientific.trec (eval), val_general.trec, val_scientific.trec (extra, not submitted)
and caches embeddings/score matrices in cache/ for Task 3."""
import argparse
import json
import os
import resource
import time

import numpy as np
import torch

from common import CACHE, PRED, SEED, evaluate, fmt, load_claims, load_corpus, qrels, topk_from_scores, write_trec
from models import BiEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--general", default="BAAI/bge-base-en-v1.5")
ap.add_argument("--scientific", default="ncbi/MedCPT")
ap.add_argument("--only", choices=["general", "scientific"])
args = ap.parse_args()
torch.manual_seed(SEED)

docs, doc_ids = load_corpus()
val, ev = load_claims("val"), load_claims("eval")
qv = qrels(val)
os.makedirs(CACHE, exist_ok=True)
stats_path = os.path.join(CACHE, "t2_stats.json")
stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


for kind in ["general", "scientific"]:
    if args.only and kind != args.only:
        continue
    name = getattr(args, kind)
    rss0 = rss_mb()
    enc = BiEncoder(name)
    model_mb = enc.n_params * 4 / 2 ** 20
    t0 = time.time()
    D = enc.encode_docs(docs)
    index_time = time.time() - t0
    # latency: one query at a time, encode + exact inner-product search + top-100
    lat = []
    for c in ev[:100]:
        t = time.time()
        q = enc.encode_queries([c["claim"]])
        s = q @ D.T
        np.argpartition(-s[0], 100)[:100]
        lat.append(time.time() - t)
    Qv, Qe = enc.encode_queries([c["claim"] for c in val]), enc.encode_queries([c["claim"] for c in ev])
    Sv, Se = Qv @ D.T, Qe @ D.T
    rv, re_ = topk_from_scores(Sv, doc_ids), topk_from_scores(Se, doc_ids)
    m = evaluate({c["id"]: d for c, (d, _) in zip(val, rv)}, qv)
    print(f"{kind:10s} {name}: {fmt(m)}  index {index_time:.1f}s  latency {np.mean(lat) * 1000:.1f}ms")
    write_trec(os.path.join(PRED, f"t2_{kind}.trec"), [c["id"] for c in ev], re_, kind)
    write_trec(os.path.join(PRED if kind == "general" else CACHE, f"val_{kind}.trec"), [c["id"] for c in val], rv, kind)
    np.save(os.path.join(CACHE, f"{kind}_val.npy"), Sv.astype(np.float32))
    np.save(os.path.join(CACHE, f"{kind}_eval.npy"), Se.astype(np.float32))
    stats[kind] = dict(model=name, params_m=enc.n_params / 1e6, val=m, index_time_sec=index_time,
                       query_latency_ms=float(np.mean(lat) * 1000), model_mb=model_mb,
                       index_mb=D.nbytes / 2 ** 20, peak_rss_delta_mb=rss_mb() - rss0,
                       memory_mb=model_mb + D.nbytes / 2 ** 20, device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    json.dump(stats, open(stats_path, "w"), indent=1)
    del enc
