"""Task 5: claim verification on top of the best retriever, plus rule-based error tagging on validation.

scorer:
  nli      zero-shot NLI over each abstract sentence (premise) vs claim (hypothesis); max over sentences
  nli-ft   same model fine-tuned on training rationales (entail/contradict) + mined neutral sentences
  lexical  no pretrained weights: logistic regression on lexical (claim, sentence) features from training claims
Decision: keep the top-N retrieved docs whose max(p_ent, p_con) >= tau and margin over the other >= delta,
at most 3 rows per claim. N, tau, delta are tuned on validation F1.
--pair_prior: if a training claim is a near-duplicate (word Jaccard >= j) of the test claim, predict its gold docs
with flipped labels (or NEI if it had none). Measured separately so its effect is visible."""
import argparse
import csv
import itertools
import json
import os
import random
import re

import numpy as np

from common import CACHE, PRED, SEED, load_claims, load_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.join(CACHE, "rerank_runs.json"), help="json {val:{cid:[docs,scores]},eval:...} or a .trec prefix")
ap.add_argument("--scorer", default="nli", choices=["nli", "nli-ft", "lexical"])
ap.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
ap.add_argument("--depth", type=int, default=5, help="docs per claim sent to the stance scorer")
ap.add_argument("--pair_prior", type=float, default=0.0, help="Jaccard threshold; 0 disables")
ap.add_argument("--tag", default="")
args = ap.parse_args()
random.seed(SEED), np.random.seed(SEED)

docs, doc_ids = load_corpus()
D = {int(d["doc_id"]): d for d in docs}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
runs = json.load(open(args.run))
runs = {sp: {int(k): v for k, v in r.items()} for sp, r in runs.items()}
FLIP = {"SUPPORT": "CONTRADICT", "CONTRADICT": "SUPPORT"}
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))


# ------------------------------------------------------------------ stance scorers -> {(cid, did): (p_ent, p_con)}
def pairs_for(claims, sp):
    return [(c, d) for c in claims for d in runs[sp][c["id"]][0][:args.depth]]


def nli_scores(model, claims, sp):
    P = pairs_for(claims, sp)
    prem, hyp, idx = [], [], []
    for k, (c, d) in enumerate(P):
        sents = D[d]["abstract"]
        for s in sents:
            prem.append(s), hyp.append(c["claim"]), idx.append(k)
        prem.append(D[d]["title"] + " " + " ".join(sents)), hyp.append(c["claim"]), idx.append(k)  # whole abstract
    p = model.predict(prem, hyp)
    out = {}
    for k, (c, d) in enumerate(P):
        rows = p[np.array(idx) == k]
        out[(c["id"], d)] = (float(rows[:, 0].max()), float(rows[:, 2].max()))
    return out


def finetune_nli(name):
    """fine-tune on training claims: rationale sentences -> entail/contradict, other sentences of retrieved docs -> neutral"""
    import torch
    from models import DEV, NLI
    m = NLI(name)
    ent, neu, con = m.idx
    ex = []
    for c in tr:
        gold = {int(e["doc_id"]): e for e in c["evidence"]}
        for d, e in gold.items():
            rat = " ".join(D[d]["abstract"][i] for i in e["sentences"])
            ex.append((rat, c["claim"], ent if e["label"] == "SUPPORT" else con))
            others = [s for i, s in enumerate(D[d]["abstract"]) if i not in e["sentences"]]
            if others and neu is not None:
                ex.append((random.choice(others), c["claim"], neu))
        for d in runs.get("train", {}).get(c["id"], [[]])[0][:3]:
            if d not in gold and neu is not None:
                ex.append((random.choice(D[d]["abstract"]), c["claim"], neu))
    print(f"NLI fine-tuning examples: {len(ex)}")
    opt = torch.optim.AdamW(m.m.parameters(), lr=1e-5, weight_decay=0.01)
    m.m.train()
    for ep in range(2):
        random.shuffle(ex)
        for i in range(0, len(ex), 16):
            b = ex[i:i + 16]
            enc = m.tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
            loss = torch.nn.functional.cross_entropy(m.m(**enc).logits, torch.tensor([x[2] for x in b], device=DEV))
            loss.backward(), opt.step(), opt.zero_grad()
    m.m.eval()
    out = os.path.join(CACHE, "nli-ft")
    m.m.save_pretrained(out), m.tok.save_pretrained(out)
    return m


def lexical_model():
    """logistic regression over hand-built features of (claim, best-matching sentence); trained on training claims."""
    from sklearn.linear_model import LogisticRegression
    NEG = {"not", "no", "never", "without", "lack", "lacks", "fail", "fails", "cannot", "unable", "absence", "neither", "nor"}
    DIR_UP = {"increase", "increases", "increased", "higher", "promote", "promotes", "enhance", "enhances", "raise", "raises", "more", "induce", "induces", "activates", "upregulates"}
    DIR_DN = {"decrease", "decreases", "decreased", "lower", "reduce", "reduces", "reduced", "inhibit", "inhibits", "suppress", "suppresses", "less", "prevents", "downregulates", "impairs"}

    def feats(claim, d, rank, score):
        cw = words(claim)
        best = max(D[d]["abstract"], key=lambda s: len(cw & words(s)) / (len(cw) + 1))
        sw = words(best)
        ov = len(cw & sw) / (len(cw) + 1)
        tov = len(cw & words(D[d]["title"])) / (len(cw) + 1)
        f = [ov, tov, 1 / (rank + 1), score, len(cw & NEG) > 0, len(sw & NEG) > 0, (len(cw & NEG) > 0) != (len(sw & NEG) > 0),
             len(cw & DIR_UP) > 0 and len(sw & DIR_DN) > 0, len(cw & DIR_DN) > 0 and len(sw & DIR_UP) > 0,
             bool(re.search(r"\d", claim))]
        return np.array(f, dtype=float)

    X, y = [], []
    for c in tr:
        gold = {int(e["doc_id"]): e["label"] for e in c["evidence"]}
        if c["id"] not in runs.get("train", {}):
            continue
        dl, sc = runs["train"][c["id"]]
        for r, d in enumerate(dl[:args.depth]):
            X.append(feats(c["claim"], d, r, sc[r] - sc[0])), y.append({"SUPPORT": 0, "CONTRADICT": 2}.get(gold.get(d), 1))
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(np.array(X), y)

    def score(claims, sp):
        out = {}
        for c in claims:
            dl, sc = runs[sp][c["id"]]
            for r, d in enumerate(dl[:args.depth]):
                p = clf.predict_proba(feats(c["claim"], d, r, sc[r] - sc[0])[None])[0]
                out[(c["id"], d)] = (float(p[0]), float(p[2]))
        return out
    return score


# ------------------------------------------------------------------ decision + metrics
_NT = {}


def nearest_train(c):
    if c["id"] not in _NT:
        cw = words(c["claim"])
        best = max(tr, key=lambda t: len(cw & words(t["claim"])) / len(cw | words(t["claim"])))
        _NT[c["id"]] = best, len(cw & words(best["claim"])) / len(cw | words(best["claim"]))
    return _NT[c["id"]]


def decide(claims, sp, P, n, tau, delta):
    pred = {}
    for c in claims:
        rows = []
        if args.pair_prior:
            t, j = nearest_train(c)
            if j >= args.pair_prior:
                pred[c["id"]] = [(int(e["doc_id"]), FLIP[e["label"]]) for e in t["evidence"]][:3]
                continue
        for d in runs[sp][c["id"]][0][:n]:
            pe, pc = P[(c["id"], d)]
            if max(pe, pc) >= tau and abs(pe - pc) >= delta:
                rows.append((max(pe, pc), d, "SUPPORT" if pe >= pc else "CONTRADICT"))
        pred[c["id"]] = [(d, l) for _, d, l in sorted(rows, reverse=True)[:3]]
    return pred


def prf(pred, claims):
    g = {(c["id"], int(e["doc_id"]), e["label"]) for c in claims for e in c["evidence"]}
    p = {(cid, d, l) for cid, rows in pred.items() for d, l in rows}
    tp = len(g & p)
    P, R = tp / max(len(p), 1), tp / max(len(g), 1)
    return dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), n_pred=len(p), n_gold=len(g))


if args.scorer == "lexical":
    sc = lexical_model()
    Pv, Pe = sc(val, "val"), sc(ev, "eval")
    model_name = "logreg-lexical-features"
else:
    import hashlib
    import pickle
    key = hashlib.md5(f"{args.scorer}|{args.nli_model}|{args.depth}|{os.path.abspath(args.run)}".encode()).hexdigest()[:10]
    cpath = os.path.join(CACHE, f"nli_probs_{key}.pkl")
    if os.path.exists(cpath):
        Pv, Pe = pickle.load(open(cpath, "rb"))
    else:
        from models import NLI
        nli = finetune_nli(args.nli_model) if args.scorer == "nli-ft" else NLI(args.nli_model)
        Pv, Pe = nli_scores(nli, val, "val"), nli_scores(nli, ev, "eval")
        pickle.dump((Pv, Pe), open(cpath, "wb"))
    # stable, machine-independent copy used by t5_verify_v2.py: nli_probs_<scorer>_<run file name>.pkl
    stable = os.path.join(CACHE, f"nli_probs_{args.scorer}_{os.path.splitext(os.path.basename(args.run))[0]}.pkl")
    if not os.path.exists(stable):
        pickle.dump((Pv, Pe), open(stable, "wb"))
    model_name = args.nli_model

grid = []
for n, tau, delta in itertools.product(range(1, args.depth + 1), np.arange(0.1, 0.96, 0.05), [0.0, 0.1, 0.2, 0.4]):
    grid.append((prf(decide(val, "val", Pv, n, tau, delta), val)["f1"], n, round(float(tau), 2), delta))
grid.sort(reverse=True)
f1, n, tau, delta = grid[0]
pv = decide(val, "val", Pv, n, tau, delta)
pe = decide(ev, "eval", Pe, n, tau, delta)
mv = prf(pv, val)
print(f"[{args.scorer}{' +pair' if args.pair_prior else ''}] best n={n} tau={tau} delta={delta}  val", mv)


# ------------------------------------------------------------------ error tagging (validation, rule-based)
NUM = re.compile(r"\d|percent|%|fold|times|half|double")
NEGW = re.compile(r"\b(not|no|never|without|lack|lacks|fail|fails|cannot|unable|absence|neither|nor)\b", re.I)


def tag(c, rows):
    gold = {int(e["doc_id"]): e["label"] for e in c["evidence"]}
    got = dict(rows)
    retrieved = set(runs["val"][c["id"]][0][:n])
    if gold and not set(gold) & retrieved:
        return "retrieval_miss"
    if gold and len(gold) > 1 and set(got) & set(gold) and set(gold) - set(got):
        return "needs_multiple_docs"
    if any(d in gold and gold[d] != l for d, l in got.items()):  # right doc, wrong polarity
        return "numerical" if NUM.search(c["claim"]) else ("negation" if NEGW.search(c["claim"]) else "other")
    if not gold:  # NEI claim but we predicted something
        return "entity_mismatch" if any(d not in gold for d in got) else "other"
    if got and not set(got) & set(gold):  # confident on the wrong abstract
        return "entity_mismatch"
    if NUM.search(c["claim"]):
        return "numerical"
    return "negation" if NEGW.search(c["claim"]) else "other"


errors = {}
for c in val:
    gold = {(int(e["doc_id"]), e["label"]) for e in c["evidence"]}
    if set(pv[c["id"]]) != gold:
        errors[c["id"]] = tag(c, pv[c["id"]])
cnt = {k: sum(v == k for v in errors.values()) for k in sorted(set(errors.values()))}
print(f"validation errors: {len(errors)}/{len(val)} claims", cnt)


def write_csv(path, pred):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "doc_id", "label"])
        for cid in sorted(pred):
            for d, l in pred[cid]:
                w.writerow([cid, d, l])


suffix = args.tag
write_csv(os.path.join(PRED, f"t5_verification{suffix}.csv"), pe)
write_csv(os.path.join(PRED, f"t5_val_verification{suffix}.csv"), pv)
with open(os.path.join(PRED, f"t5_val_errors{suffix}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_id", "category"])
    for cid in sorted(errors):
        w.writerow([cid, errors[cid]])
json.dump(dict(scorer=args.scorer, model=model_name, pair_prior=args.pair_prior, n=n, tau=tau, delta=delta, val=mv,
               errors=cnt, n_errors=len(errors), top_grid=grid[:5]),
          open(os.path.join(CACHE, f"t5_stats{suffix or '_' + args.scorer}.json"), "w"), indent=1)
