"""Can the error-analysis findings be used? 5-fold CV on validation for the submitted verifier (t5_verify_v2 logic) with
(a) separate thresholds for claims with / without a near-duplicate training claim, (b) separate thresholds for clinical vs
molecular topics, (c) a bias term on the SUPPORT/CONTRADICT decision. Both group features are known at test time."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import itertools
import json
import os
import pickle
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from common import CACHE, SEED, load_claims

tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
H = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))["val"]
CE = json.load(open(os.path.join(CACHE, "ce_ensemble_expanded_val30.json")))
NL = [pickle.load(open(os.path.join(CACHE, f), "rb"))[0] for f in ["nli_probs_nli_hybrid_runs.pkl", "nli_probs_nli-ft_hybrid_runs.pkl"]]
gold = {c["id"]: {(int(e["doc_id"]), e["label"]) for e in c["evidence"]} for c in val}
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
twin = {c["id"]: max(len(words(c["claim"]) & words(t["claim"])) / len(words(c["claim"]) | words(t["claim"])) for t in tr) >= 0.6 for c in val}
allc = tr + val + ev
E = np.load(os.path.join(CACHE, "claim_emb_bge_base.npy"))
km = KMeans(8, random_state=SEED, n_init=20).fit(E)
tf = TfidfVectorizer(stop_words="english", min_df=3, sublinear_tf=True)
X = tf.fit_transform([c["claim"] for c in allc]).toarray()
t = np.array(tf.get_feature_names_out())
clin = [k for k in range(8) if t[np.argsort(-(X[km.labels_ == k].mean(0) - X.mean(0)))[0]] in ("incidence", "risk")]
lab = dict(zip([c["id"] for c in allc], km.labels_))
topic = {c["id"]: lab[c["id"]] in clin for c in val}


def predict(c, s, n, rel, bias=0.0):
    dl, sc = H[str(c["id"])][0][:3], np.array(CE[str(c["id"])][:3])
    rows = []
    for i in np.argsort(-sc)[:n]:
        if sc[i] >= s and sc[i] >= sc.max() - rel:
            pe, pc = np.mean([P[(c["id"], dl[i])] for P in NL], 0)
            rows.append((dl[i], "SUPPORT" if pe >= pc + bias else "CONTRADICT"))
    return set(rows)


def f1(pairs):
    tp = sum(len(p & gold[i]) for i, p in pairs)
    a, b = sum(len(p) for _, p in pairs), sum(len(gold[i]) for i, _ in pairs)
    P, R = tp / max(a, 1), tp / max(b, 1)
    return 2 * P * R / max(P + R, 1e-9)


allsc = np.concatenate([CE[str(c["id"])][:3] for c in val])
G = list(itertools.product([float(x) for x in np.quantile(allsc, np.arange(0.3, 0.95, 0.05))], [1, 2, 3], [0.5, 1, 2, 100]))
folds = np.array_split(np.random.default_rng(42).permutation(len(val)), 5)


def cv(group=None, biases=(0.0,)):
    """group: dict claim_id -> bool or None. Tune (s, n, rel[, bias]) per group on 4 folds, score the 5th."""
    out = []
    for k in range(5):
        te = [val[i] for i in folds[k]]
        trn = [val[i] for j in range(5) if j != k for i in folds[j]]
        for gv in ([None] if group is None else [True, False]):
            sub_tr = [c for c in trn if group is None or group[c["id"]] == gv]
            sub_te = [c for c in te if group is None or group[c["id"]] == gv]
            best = max(((g, b) for g in G for b in biases), key=lambda gb: f1([(c["id"], predict(c, *gb[0], gb[1])) for c in sub_tr]))
            out += [(c["id"], predict(c, *best[0], best[1])) for c in sub_te]
    return f1(out)


print(f"submitted rule (one global threshold)      5-fold CV F1 {cv():.3f}")
print(f"(a) separate thresholds by training twin   5-fold CV F1 {cv(twin):.3f}")
print(f"(b) separate thresholds clinical/molecular 5-fold CV F1 {cv(topic):.3f}")
print(f"(c) SUPPORT/CONTRADICT bias term           5-fold CV F1 {cv(biases=(-0.2, -0.1, 0.0, 0.1, 0.2)):.3f}")
