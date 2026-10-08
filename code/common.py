"""Shared helpers: data loading, metrics, TREC run writing."""
import json
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PRED = os.environ.get("PRED_DIR", os.path.join(ROOT, "predictions"))
CACHE = os.environ.get("CACHE_DIR", os.path.join(ROOT, "cache"))
SEED = 42
TOP_K = 100


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_corpus():
    docs = read_jsonl(os.path.join(DATA, "corpus.jsonl"))
    return docs, np.array([int(d["doc_id"]) for d in docs])


def load_claims(split):
    return read_jsonl(os.path.join(DATA, f"{split}_claims.jsonl"))


def doc_text(d):
    return d["title"] + " " + " ".join(d["abstract"])


def qrels(claims):
    """{claim_id: {doc_id: label}} for claims with evidence."""
    out = {}
    for c in claims:
        if c.get("evidence"):
            out[c["id"]] = {int(e["doc_id"]): e["label"] for e in c["evidence"]}
    return out


# ------------------------------------------------------------------ metrics
def ndcg_at(ranked, rel, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg


def per_claim(run, qr, k=10):
    return {q: ndcg_at(run[q], qr[q], k) for q in qr}


def evaluate(run, qr):
    """run: {claim_id: [doc_id ranked]}; metrics over claims with gold evidence."""
    nd, rc, mrr = [], [], []
    for q, rel in qr.items():
        r = run[q]
        nd.append(ndcg_at(r, rel))
        rc.append(len(set(r[:100]) & set(rel)) / len(rel))
        mrr.append(next((1.0 / (i + 1) for i, d in enumerate(r) if d in rel), 0.0))
    return {"ndcg10": float(np.mean(nd)), "recall100": float(np.mean(rc)), "mrr": float(np.mean(mrr))}


def fmt(m):
    return f"nDCG@10={m['ndcg10']:.4f}  R@100={m['recall100']:.4f}  MRR={m['mrr']:.4f}"


# ------------------------------------------------------------------ runs
def topk_from_scores(scores, doc_ids, k=TOP_K):
    """scores: (n_claims, n_docs) array -> list of (doc_ids, scores) top-k, stable order."""
    out = []
    for row in scores:
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.lexsort((idx, -row[idx]))]
        out.append((doc_ids[idx].tolist(), row[idx].tolist()))
    return out


def write_trec(path, claim_ids, ranked, run_name):
    """ranked: list of (doc_ids, scores), aligned with claim_ids. Enforces format rules."""
    with open(path, "w") as f:
        for cid, (docs, scores) in zip(claim_ids, ranked):
            assert len(docs) == TOP_K and len(set(docs)) == TOP_K, cid
            prev = float("inf")
            for r, (d, s) in enumerate(zip(docs, scores), 1):
                s = min(float(s), prev)  # scores must not increase with rank
                prev = s
                f.write(f"{cid} Q0 {d} {r} {s:.6f} {run_name}\n")


def read_trec(path):
    run = {}
    for line in open(path):
        q, _, d, r, s, _ = line.split()
        run.setdefault(int(q), []).append((int(r), int(d), float(s)))
    return {q: [d for _, d, _ in sorted(v)] for q, v in run.items()}


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
