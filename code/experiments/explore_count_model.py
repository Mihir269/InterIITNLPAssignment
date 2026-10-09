"""Learned evidence selection ("how many abstracts to return"), step 1: nested 5-fold CV on validation.
Per candidate (t3_rerank top-3): logistic regression on candidate features (cross-encoder ensemble score, rank, gap to the
claim's best, NLI ensemble confidence and margin) + claim features (best/second score, training twin, claim length).
Selected = candidates with p >= threshold (threshold tuned on the training folds); label from the NLI ensemble as before.
Compared with the submitted threshold rule under the same folds (5-fold CV F1 0.763)."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import itertools
import json
import os
import pickle
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import CACHE, load_claims

tr, val = load_claims("train"), load_claims("val")
H = json.load(open(os.path.join(CACHE, "rerank_cand_runs.json")))["val"]
HY = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))["val"]
raw = json.load(open(os.path.join(CACHE, "ce_ensemble_expanded_val30.json")))
CE = {cid: dict(zip(HY[cid][0][:30], sc)) for cid, sc in raw.items()}
NL = [pickle.load(open(os.path.join(CACHE, f), "rb"))[0] for f in ["nli_probs_nli_rerank_cand_runs.pkl", "nli_probs_nli-ft_rerank_cand_runs.pkl"]]
gold = {c["id"]: {(int(e["doc_id"]), e["label"]) for e in c["evidence"]} for c in val}
gdoc = {k: {d for d, _ in v} for k, v in gold.items()}
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
twin = {c["id"]: max(len(words(c["claim"]) & words(t["claim"])) / len(words(c["claim"]) | words(t["claim"])) for t in tr) for c in val}

rows = []  # (claim_id, doc, features, is_gold, label)
for c in val:
    dl = H[str(c["id"])][0][:3]
    sc = np.array([CE[str(c["id"])][d] for d in dl])
    srt = np.sort(sc)[::-1]
    for r, d in enumerate(dl):
        pe, pc = np.mean([P[(c["id"], d)] for P in NL], 0)
        f = [sc[r], r, srt[0] - sc[r], max(pe, pc), abs(pe - pc), srt[0], srt[0] - srt[1], twin[c["id"]],
             len(c["claim"].split()), sc[r] * max(pe, pc)]
        rows.append((c["id"], d, f, d in gdoc[c["id"]], "SUPPORT" if pe >= pc else "CONTRADICT"))
Xall = np.array([r[2] for r in rows])
yall = np.array([r[3] for r in rows])
cid_of = np.array([r[0] for r in rows])


def f1(pred, ids):
    tp = sum(len(pred.get(i, set()) & gold[i]) for i in ids)
    a, b = sum(len(pred.get(i, set())) for i in ids), sum(len(gold[i]) for i in ids)
    P, R = tp / max(a, 1), tp / max(b, 1)
    return 2 * P * R / max(P + R, 1e-9)


def select(p, idx, th, maxk):
    pred = {}
    for i in idx:
        cid, d, _, _, lab = rows[i]
        pred.setdefault(cid, [])
        pred[cid].append((p[i], d, lab))
    return {cid: {(d, l) for pp, d, l in sorted(v, reverse=True)[:maxk] if pp >= th} for cid, v in pred.items()}


folds = np.array_split(np.random.default_rng(42).permutation(len(val)), 5)
for C in [0.1, 1.0]:
    allp = {}
    for k in range(5):
        te_ids = {val[i]["id"] for i in folds[k]}
        trm, tem = ~np.isin(cid_of, list(te_ids)), np.isin(cid_of, list(te_ids))
        sc = StandardScaler().fit(Xall[trm])
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(Xall[trm]), yall[trm])
        p = m.predict_proba(sc.transform(Xall))[:, 1]
        tr_ids = [c["id"] for c in val if c["id"] not in te_ids]
        th, mk = max(itertools.product(np.arange(0.2, 0.96, 0.025), [1, 2, 3]),
                     key=lambda g: f1(select(p, np.where(trm)[0], *g), tr_ids))
        allp.update(select(p, np.where(tem)[0], th, mk))
    print(f"learned selection (logreg C={C}): nested 5-fold CV F1 {f1(allp, [c['id'] for c in val]):.4f}")
print("submitted threshold rule, same folds:  5-fold CV F1 0.7631")
