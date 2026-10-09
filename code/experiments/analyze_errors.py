"""Error analysis of the submitted Task 5 verifier on validation claims.
Claim-level confusion (gold vs predicted verdict), examples per cell, error rate by claim property, and by topic
(TF-IDF + k-means over all claims, clusters named by their top terms). Writes docs/ERROR_ANALYSIS.md (sections 7-8 of that file were added by hand afterwards:
significance tests and the follow-up experiments; re-running this script regenerates sections 1-6 only)."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import collections
import csv
import json
import os
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from common import CACHE, ROOT, SEED, load_claims, load_corpus

docs, ids = load_corpus()
D = {int(d["doc_id"]): d for d in docs}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
H = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))["val"]
gold = {c["id"]: {(int(e["doc_id"]), e["label"]) for e in c["evidence"]} for c in val}
pred = collections.defaultdict(set)
for r in csv.DictReader(open(os.path.join(ROOT, "predictions", "t5_val_verification.csv"))):
    pred[int(r["claim_id"])].add((int(r["doc_id"]), r["label"]))
cat = {int(r["claim_id"]): r["category"] for r in csv.DictReader(open(os.path.join(ROOT, "predictions", "t5_val_errors.csv")))}


def verdict(rows):
    labs = {l for _, l in rows}
    return "NEI" if not labs else (labs.pop() if len(labs) == 1 else "MIXED")


def outcome(c):
    g, p = gold[c["id"]], pred[c["id"]]
    if g == p:
        return "correct"
    gd, pd = {d for d, _ in g}, {d for d, _ in p}
    if not g:
        return "false alarm (NEI claim, predicted evidence)"
    if not p:
        return "miss (has evidence, predicted NEI)"
    if gd == pd:
        return "right abstract, wrong label"
    if gd & pd:
        return "partially right (missing / extra abstract)"
    return "wrong abstract"


out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))

# ---------------------------------------------------------------- 1. confusion
labels = ["SUPPORT", "CONTRADICT", "NEI"]
conf = collections.Counter((verdict(gold[c["id"]]), verdict(pred[c["id"]])) for c in val)
P("## 1. Claim-level confusion (rows = gold verdict, columns = predicted verdict)\n")
P("| gold \\ predicted | SUPPORT | CONTRADICT | NEI | total | verdict accuracy |")
P("|---|---|---|---|---|---|")
for g in labels:
    row = [conf[(g, p)] for p in labels]
    P(f"| **{g}** | " + " | ".join(map(str, row)) + f" | {sum(row)} | {row[labels.index(g)] / max(sum(row), 1):.0%} |")
P("\nVerdict = the label of the predicted rows (NEI if none). A verdict can be right while the abstract is wrong; section 2 "
  "counts exact matches of (abstract, label) pairs, which is what the grader scores.\n")
oc = collections.Counter(outcome(c) for c in val)
P("## 2. Exact-match outcomes\n")
P("| outcome | claims |")
P("|---|---|")
for k, v in oc.most_common():
    P(f"| {k} | {v} |")
tp = sum(len(gold[c['id']] & pred[c['id']]) for c in val)
npred = sum(len(pred[c["id"]]) for c in val)
ngold = sum(len(gold[c["id"]]) for c in val)
P(f"\nTriples: {tp} true positives, {npred - tp} false positives, {ngold - tp} false negatives "
  f"(precision {tp / npred:.3f}, recall {tp / ngold:.3f}).\n")


# ---------------------------------------------------------------- 3. examples
def title(d):
    return D[d]["title"][:110]


P("## 3. The individual errors\n")
for name, test in [("Gold SUPPORT → predicted CONTRADICT (label flipped)", lambda g, p: g == "SUPPORT" and p == "CONTRADICT"),
                   ("Gold CONTRADICT → predicted SUPPORT (label flipped)", lambda g, p: g == "CONTRADICT" and p == "SUPPORT"),
                   ("Gold NEI → predicted evidence (false alarm)", lambda g, p: g == "NEI" and p != "NEI"),
                   ("Gold evidence → predicted NEI (miss)", lambda g, p: g != "NEI" and p == "NEI")]:
    rows = [c for c in val if test(verdict(gold[c["id"]]), verdict(pred[c["id"]]))]
    P(f"### {name} — {len(rows)} claims\n")
    P("| id | claim | abstract involved |")
    P("|---|---|---|")
    for c in rows:
        d = next(iter({d for d, _ in pred[c['id']]} or {d for d, _ in gold[c['id']]}))
        P(f"| {c['id']} | {c['claim']} | {title(d)} |")
    P("")

# ---------------------------------------------------------------- 4. by claim property
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
cited = {int(e["doc_id"]) for c in tr for e in c["evidence"]}
NEG = re.compile(r"\b(not|no|never|without|lack|lacks|fail|fails|cannot|unable|absence|neither|nor|does not|do not)\b", re.I)
NUM = re.compile(r"\d|percent|%|fold|times|half|double")
DIR = re.compile(r"\b(increase[sd]?|decrease[sd]?|reduce[sd]?|higher|lower|more|less|promote[sd]?|inhibit[sd]?|impair[sd]?|enhance[sd]?)\b", re.I)


def twin(c):
    cw = words(c["claim"])
    return max(len(cw & words(t["claim"])) / len(cw | words(t["claim"])) for t in tr)


props = {c["id"]: {
    "has negation word": bool(NEG.search(c["claim"])),
    "has a number / quantity": bool(NUM.search(c["claim"])),
    "has a direction word (increase/reduce…)": bool(DIR.search(c["claim"])),
    "long claim (> 15 words)": len(c["claim"].split()) > 15,
    "near-duplicate training claim (Jaccard ≥ 0.6)": twin(c) >= 0.6,
    "gold abstract also cited by a training claim": bool({d for d, _ in gold[c["id"]]} & cited),
    "more than one gold abstract": len(gold[c["id"]]) > 1,
    "NEI claim": not gold[c["id"]],
} for c in val}
err = {c["id"]: gold[c["id"]] != pred[c["id"]] for c in val}
base = np.mean(list(err.values()))
P(f"## 4. Error rate by claim property (overall: {base:.0%} of {len(val)} claims)\n")
P("| property | claims with it | error rate with | error rate without |")
P("|---|---|---|---|")
for k in props[val[0]["id"]]:
    w = [err[i] for i in err if props[i][k]]
    wo = [err[i] for i in err if not props[i][k]]
    P(f"| {k} | {len(w)} | {np.mean(w):.0%} | {np.mean(wo):.0%} |")
P("")

# ---------------------------------------------------------------- 5. by topic
allc = tr + val + ev
text = [c["claim"] for c in allc]
from models import BiEncoder  # noqa: E402
emb_path = os.path.join(CACHE, "claim_emb_bge_base.npy")
if os.path.exists(emb_path):
    E = np.load(emb_path)
else:
    E = BiEncoder("BAAI/bge-base-en-v1.5").encode_queries(text)
    np.save(emb_path, E)
K = 8
km = KMeans(K, random_state=SEED, n_init=20).fit(E)
tf = TfidfVectorizer(stop_words="english", min_df=3, sublinear_tf=True)
X = tf.fit_transform(text).toarray()
terms = np.array(tf.get_feature_names_out())
overall = X.mean(0)
names = {k: ", ".join(terms[np.argsort(-(X[km.labels_ == k].mean(0) - overall))[:5]]) for k in range(K)}
lab = dict(zip([c["id"] for c in allc], km.labels_))
P(f"## 5. Error rate by topic (k-means on bge-base embeddings of all {len(allc)} claims, k={K}; topic named by its most distinctive terms)\n")
P("| topic (distinctive terms) | val claims | errors | error rate | NEI share | main error outcome |")
P("|---|---|---|---|---|---|")
rows = []
for k in range(K):
    cl = [c for c in val if lab[c["id"]] == k]
    if not cl:
        continue
    e = [c for c in cl if err[c["id"]]]
    main = collections.Counter(outcome(c) for c in e).most_common(1)
    rows.append((len(e) / len(cl), k, cl, e, main))
for rate, k, cl, e, main in sorted(rows, reverse=True):
    nei = np.mean([not gold[c["id"]] for c in cl])
    P(f"| {names[k]} | {len(cl)} | {len(e)} | {rate:.0%} | {nei:.0%} | {main[0][0] + ' (' + str(main[0][1]) + ')' if main else '–'} |")
P(f"\nWith 10–30 claims per topic, one claim moves a topic's error rate by 3–10 points; treat differences under ~20 "
  f"points as noise.\n")

# ---------------------------------------------------------------- 6. error categories submitted
P("## 6. Submitted error categories (t5_val_errors.csv)\n")
P("| category | claims |")
P("|---|---|")
for k, v in collections.Counter(cat.values()).most_common():
    P(f"| {k} | {v} |")
open(os.path.join(ROOT, "docs", "ERROR_ANALYSIS.md"), "w").write(
    "# Task 5 error analysis (validation claims, submitted verifier)\n\nGenerated by `code/analyze_errors.py`.\n\n" + "\n".join(out) + "\n")
print("\n".join(out))
