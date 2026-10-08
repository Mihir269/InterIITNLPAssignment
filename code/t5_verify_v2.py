"""Task 5 (v2): cross-encoder-gated verification.
Evidence selection: among the hybrid's top-3 abstracts, keep those whose MedCPT cross-encoder score (claim vs
training-claim-expanded abstract) is >= s and within `rel` of the claim's best score (at most n per claim).
Label: SUPPORT/CONTRADICT from the mean of zero-shot and fine-tuned NLI probabilities (max over sentences).
s, n, rel tuned on validation F1; a 5-fold cross-validated estimate on validation is reported alongside.
Inputs (all produced earlier): cache/hybrid_runs.json, cache/ce_ncbi_MedCPT-Cross-Encoder_expanded_{val,eval}30.json,
NLI probability caches from `t5_verify.py --scorer nli` and `--scorer nli-ft` (depth 3)."""
import argparse
import csv
import itertools
import json
import os
import pickle
import re

import numpy as np

from common import CACHE, PRED, load_claims

ap = argparse.ArgumentParser()
ap.add_argument("--nli", default="nli_probs_71cec93cb3.pkl,nli_probs_70ef28d906.pkl", help="comma list of NLI caches to average")
ap.add_argument("--ce", default="ce_ncbi_MedCPT-Cross-Encoder_expanded_{}30.json")
ap.add_argument("--run", default="hybrid_runs.json", help="candidate run file in cache/ (top-3 per claim are verified)")
ap.add_argument("--tag", default="")
args = ap.parse_args()

val, ev = load_claims("val"), load_claims("eval")
H = json.load(open(os.path.join(CACHE, args.run)))
HY = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))  # cross-encoder scores are aligned with hybrid ranks 1..30
CE = {}
for sp in ["val", "eval"]:
    raw = json.load(open(os.path.join(CACHE, args.ce.format(sp))))
    CE[sp] = {cid: dict(zip(HY[sp][cid][0][:len(sc)], sc)) for cid, sc in raw.items()}
NL = [pickle.load(open(os.path.join(CACHE, f), "rb")) for f in args.nli.split(",")]
NLI = {"val": [p[0] for p in NL], "eval": [p[1] for p in NL]}
gold = {c["id"]: {(int(e["doc_id"]), e["label"]) for e in c["evidence"]} for c in val}


def predict(c, sp, s, n, rel):
    dl = H[sp][str(c["id"])][0][:3]
    sc = np.array([CE[sp][str(c["id"])][d] for d in dl])
    rows = []
    for i in np.argsort(-sc)[:n]:
        if sc[i] >= s and sc[i] >= sc.max() - rel:
            pe, pc = np.mean([P[(c["id"], dl[i])] for P in NLI[sp]], 0)
            rows.append((dl[i], "SUPPORT" if pe >= pc else "CONTRADICT"))
    return rows


def prf(claims, sp, g):
    pr = {(c["id"], d, l) for c in claims for d, l in predict(c, sp, *g)}
    gd = {(c["id"],) + x for c in claims for x in gold[c["id"]]}
    tp = len(pr & gd)
    P, R = tp / max(len(pr), 1), tp / max(len(gd), 1)
    return dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9))


allsc = np.array([CE["val"][str(c["id"])][d] for c in val for d in H["val"][str(c["id"])][0][:3]])
GRID = list(itertools.product([float(x) for x in np.quantile(allsc, np.arange(0.3, 0.95, 0.025))], [1, 2, 3], [0.5, 1, 2, 100]))
best = max(GRID, key=lambda g: prf(val, "val", g)["f1"])
mv = prf(val, "val", best)
folds = np.array_split(np.random.default_rng(42).permutation(len(val)), 5)
cvp, cvg = set(), set()
for k in range(5):
    te = [val[i] for i in folds[k]]
    trn = [val[i] for j in range(5) if j != k for i in folds[j]]
    g = max(GRID, key=lambda g: prf(trn, "val", g)["f1"])
    cvp |= {(c["id"], d, l) for c in te for d, l in predict(c, "val", *g)}
    cvg |= {(c["id"],) + x for c in te for x in gold[c["id"]]}
tp = len(cvp & cvg)
cvP, cvR = tp / len(cvp), tp / len(cvg)
cv = 2 * cvP * cvR / (cvP + cvR)
print(f"best s={best[0]:.3f} n={best[1]} rel={best[2]}  val", {k: round(v, 4) for k, v in mv.items()}, f" 5-fold CV F1 {cv:.4f}")

# ---- error tagging (same rules as t5_verify.py)
NUM = re.compile(r"\d|percent|%|fold|times|half|double")
NEGW = re.compile(r"\b(not|no|never|without|lack|lacks|fail|fails|cannot|unable|absence|neither|nor)\b", re.I)


def tag(c, rows):
    g = {d: l for d, l in gold[c["id"]]}
    got = dict(rows)
    if g and not set(g) & set(H["val"][str(c["id"])][0][:3]):
        return "retrieval_miss"
    if len(g) > 1 and set(got) & set(g) and set(g) - set(got):
        return "needs_multiple_docs"
    if any(d in g and g[d] != l for d, l in got.items()):
        return "numerical" if NUM.search(c["claim"]) else ("negation" if NEGW.search(c["claim"]) else "other")
    if not g:
        return "entity_mismatch" if got else "other"
    if got and not set(got) & set(g):
        return "entity_mismatch"
    return "numerical" if NUM.search(c["claim"]) else ("negation" if NEGW.search(c["claim"]) else "other")


pv = {c["id"]: predict(c, "val", *best) for c in val}
pe = {c["id"]: predict(c, "eval", *best) for c in ev}
errors = {c["id"]: tag(c, pv[c["id"]]) for c in val if set(pv[c["id"]]) != gold[c["id"]]}
cnt = {k: sum(v == k for v in errors.values()) for k in sorted(set(errors.values()))}
print(f"validation errors: {len(errors)}/{len(val)}", cnt)


def write_csv(path, pred):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "doc_id", "label"])
        for cid in sorted(pred):
            for d, l in pred[cid]:
                w.writerow([cid, d, l])


write_csv(os.path.join(PRED, f"t5_verification{args.tag}.csv"), pe)
write_csv(os.path.join(PRED, f"t5_val_verification{args.tag}.csv"), pv)
with open(os.path.join(PRED, f"t5_val_errors{args.tag}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_id", "category"])
    for cid in sorted(errors):
        w.writerow([cid, errors[cid]])
json.dump(dict(scorer="ce-gated", ce=args.ce, run=args.run, model="ncbi/MedCPT-Cross-Encoder (evidence selection) + mean of zero-shot and fine-tuned "
                                    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (label)",
               s=best[0], n=best[1], rel=best[2], val=mv, cv_f1=cv, errors=cnt, n_errors=len(errors)),
          open(os.path.join(CACHE, f"t5_stats_v2{args.tag}.json"), "w"), indent=1)
