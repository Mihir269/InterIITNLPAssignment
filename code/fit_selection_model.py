"""Learned evidence selection, step 2: train on training claims (unbiased zero-shot features from build_train_features.py),
evaluate once on validation (held out), apply to eval. Candidates on val/eval = t3_rerank top-3 (as submitted); features
use only zero-shot models on both sides. Label = NLI ensemble (zero-shot + fine-tuned) as in t5_verify_v2.
Selection threshold / max-k: (a) tuned on training claims -> clean held-out val score; (b) tuned on val -> 5-fold CV."""
import argparse
import csv
import itertools
import json
import os
import pickle
import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import CACHE, PRED, load_claims

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true", help="write predictions (to PRED_DIR) with the best variant")
args = ap.parse_args()
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
W = {c["id"]: words(c["claim"]) for c in tr}


def twin(c):
    cw = words(c["claim"])
    return max(len(cw & W[t["id"]]) / len(cw | W[t["id"]]) for t in tr if t["id"] != c["id"])


def feats(sc, nli, rank_sc, claim, tw):
    """sc: zero-shot CE scores of the 3 candidates; nli: [(p_ent, p_con)]; rank by rank_sc."""
    srt = np.sort(sc)[::-1]
    order = list(np.argsort(-np.array(rank_sc), kind="stable"))
    out = []
    for i in range(len(sc)):
        pe, pc = nli[i]
        out.append([sc[i], order.index(i), srt[0] - sc[i], max(pe, pc), abs(pe - pc), srt[0], srt[0] - srt[1], tw,
                    len(claim.split()), sc[i] * max(pe, pc)])
    return out


# ---- training rows
T = json.load(open(os.path.join(CACHE, "train_sel_features.json")))
Xt, yt = [], []
for c in tr:
    f = T[str(c["id"])]
    g = {int(e["doc_id"]) for e in c["evidence"]}
    for row, d in zip(feats(f["ce"], f["nli"], f["ce"], c["claim"], twin(c)), f["cand"]):
        Xt.append(row), yt.append(d in g)
Xt, yt = np.array(Xt), np.array(yt)
print(f"training rows {len(yt)} ({yt.mean():.0%} gold); gold reachable in train top-3: "
      f"{sum(len({int(e['doc_id']) for e in c['evidence']} & set(T[str(c['id'])]['cand'])) for c in tr)}/"
      f"{sum(len(c['evidence']) for c in tr)}")

# ---- val / eval rows
R = json.load(open(os.path.join(CACHE, "rerank_cand_runs.json")))
HY = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))
ZS = {sp: {cid: dict(zip(HY[sp][cid][0][:30], s)) for cid, s in
           json.load(open(os.path.join(CACHE, f"ce_ncbi_MedCPT-Cross-Encoder_expanded_{sp}30.json"))).items()} for sp in ["val", "eval"]}
NZ = pickle.load(open(os.path.join(CACHE, "nli_probs_nli_rerank_cand_runs.pkl"), "rb"))
NF = pickle.load(open(os.path.join(CACHE, "nli_probs_nli-ft_rerank_cand_runs.pkl"), "rb"))
SP = {"val": 0, "eval": 1}


def rows_for(claims, sp):
    out = []
    for c in claims:
        dl = R[sp][str(c["id"])][0][:3]
        sc = [ZS[sp][str(c["id"])][d] for d in dl]
        nz = [NZ[SP[sp]][(c["id"], d)] for d in dl]
        for row, d, pz in zip(feats(sc, nz, sc, c["claim"], twin(c)), dl, nz):
            pe, pc = np.mean([pz, NF[SP[sp]][(c["id"], d)]], 0)
            out.append((c["id"], d, row, "SUPPORT" if pe >= pc else "CONTRADICT"))
    return out


V, E = rows_for(val, "val"), rows_for(ev, "eval")
Xv, Xe = np.array([r[2] for r in V]), np.array([r[2] for r in E])
gold = {c["id"]: {(int(e["doc_id"]), e["label"]) for e in c["evidence"]} for c in val}


def select(rows, p, th, mk):
    by = {}
    for (cid, d, _, lab), pp in zip(rows, p):
        by.setdefault(cid, []).append((pp, d, lab))
    return {cid: {(d, l) for pp, d, l in sorted(v, reverse=True)[:mk] if pp >= th} for cid, v in by.items()}


def f1(pred, ids):
    tp = sum(len(pred.get(i, set()) & gold[i]) for i in ids)
    a, b = sum(len(pred.get(i, set())) for i in ids), sum(len(gold[i]) for i in ids)
    P, R_ = tp / max(a, 1), tp / max(b, 1)
    return 2 * P * R_ / max(P + R_, 1e-9), P, R_


def train_f1(p, th, mk):
    """F1 on training claims with the same selection (labels: gold label if doc is gold — label quality is not under test)."""
    by, i = {}, 0
    for c in tr:
        for d in T[str(c["id"])]["cand"]:
            by.setdefault(c["id"], []).append((p[i], d))
            i += 1
    tp = a = b = 0
    for c in tr:
        g = {int(e["doc_id"]) for e in c["evidence"]}
        sel = {d for pp, d in sorted(by[c["id"]], reverse=True)[:mk] if pp >= th}
        tp += len(sel & g); a += len(sel); b += len(g)
    P, R_ = tp / max(a, 1), tp / max(b, 1)
    return 2 * P * R_ / max(P + R_, 1e-9)


G = list(itertools.product(np.arange(0.1, 0.96, 0.025), [1, 2, 3]))
folds = np.array_split(np.random.default_rng(42).permutation(len(val)), 5)
ids = [c["id"] for c in val]
results = {}
for name, model in [("logreg", LogisticRegression(C=1.0, max_iter=3000)),
                    ("gboost", GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42))]:
    sc = StandardScaler().fit(Xt)
    model.fit(sc.transform(Xt), yt)
    pt, pv, pe = (model.predict_proba(sc.transform(X))[:, 1] for X in (Xt, Xv, Xe))
    # (a) threshold tuned on training claims -> clean held-out score on val
    ga = max(G, key=lambda g: train_f1(pt, *g))
    fa = f1(select(V, pv, *ga), ids)
    # (b) threshold tuned on val -> 5-fold CV estimate
    cvp = {}
    for k in range(5):
        te = {val[i]["id"] for i in folds[k]}
        trn = [i for i in ids if i not in te]
        g = max(G, key=lambda g: f1(select(V, pv, *g), trn)[0])
        cvp.update({cid: s for cid, s in select(V, pv, *g).items() if cid in te})
    gb = max(G, key=lambda g: f1(select(V, pv, *g), ids)[0])
    fb = f1(cvp, ids)
    results[name] = dict(a=(ga, fa), b=(gb, fb), pv=pv, pe=pe)
    print(f"{name}: (a) train-tuned th={ga[0]:.3f} k≤{ga[1]}: held-out val F1 {fa[0]:.4f} (P {fa[1]:.3f} R {fa[2]:.3f})"
          f"   (b) val-tuned: 5-fold CV F1 {fb[0]:.4f}")
print("reference: submitted threshold rule, 5-fold CV F1 0.7631 (in-sample 0.7631)")
json.dump({k: dict(a_th=v["a"][0], a_val=v["a"][1], b_th=v["b"][0], b_cv=v["b"][1]) for k, v in results.items()},
          open(os.path.join(CACHE, "selection_model_stats.json"), "w"), indent=1, default=float)
