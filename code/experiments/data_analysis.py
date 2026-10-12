"""Final data analysis: dataset, retrieval, verification, eval predictions. Prints the numbers used in
docs/DATA_ANALYSIS.md, saves them to docs/figures/analysis_numbers.json and draws docs/figures/*.png."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import collections
import csv
import json
import os
import pickle
import re

import numpy as np
import Stemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from common import CACHE, ROOT, evaluate, load_claims, load_corpus, qrels, read_trec

FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
N = {}  # every number reported
stem = Stemmer.Stemmer("english")
docs, ids = load_corpus()
D = {int(d["doc_id"]): d for d in docs}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
content = lambda s: set(stem.stemWords([w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in ENGLISH_STOP_WORDS]))


def pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


# ======================================================================= 1. dataset
ds = {}
for name, cl in [("train", tr), ("val", val), ("eval", ev)]:
    r = {"claims": len(cl), "claim_words_mean": round(np.mean([len(c["claim"].split()) for c in cl]), 1),
         "claim_words_p10_p90": [int(np.percentile([len(c["claim"].split()) for c in cl], q)) for q in (10, 90)]}
    if name != "eval":
        labs = [{e["label"] for e in c["evidence"]} for c in cl]
        r.update(support=sum(l == {"SUPPORT"} for l in labs), contradict=sum(l == {"CONTRADICT"} for l in labs),
                 nei=sum(not l for l in labs), mixed=sum(len(l) > 1 for l in labs),
                 docs_per_claim=dict(collections.Counter(len(c["evidence"]) for c in cl if c["evidence"])),
                 rationale_sents_per_doc=dict(collections.Counter(len(e["sentences"]) for c in cl for e in c["evidence"])))
    ds[name] = r
alen = [sum(len(s.split()) for s in d["abstract"]) for d in docs]
nsent = [len(d["abstract"]) for d in docs]
ds["corpus"] = {"abstracts": len(docs), "words_mean": round(np.mean(alen)), "words_median": int(np.median(alen)),
                "words_p90": int(np.percentile(alen, 90)), "sentences_mean": round(np.mean(nsent), 1),
                "share_over_512_tokens_approx": pct(sum(w * 1.3 > 512 for w in alen), len(alen))}
lab = [c for c in tr + val]
cited = collections.Counter(int(e["doc_id"]) for c in lab for e in c["evidence"])
ds["corpus"]["cited_by_any_labelled_claim"] = len(cited)
ds["corpus"]["cited_by_2plus_claims"] = sum(v > 1 for v in cited.values())
# rationale sentence position (relative, 0 = first sentence, 1 = last)
pos = [i / max(len(D[int(e["doc_id"])]["abstract"]) - 1, 1) for c in lab for e in c["evidence"] for i in e["sentences"]]
ds["rationale_position"] = {"n": len(pos), "share_last_third": pct(sum(p > 2 / 3 for p in pos), len(pos)),
                            "share_first_third": pct(sum(p < 1 / 3 for p in pos), len(pos)),
                            "share_last_sentence": pct(sum(p == 1 for p in pos), len(pos)),
                            "hist10": np.histogram(pos, bins=10, range=(0, 1))[0].tolist()}
# lexical overlap: share of claim content words present in gold abstract / in rationale sentences
ov_abs, ov_rat, ov_lab = [], [], []
for c in lab:
    cw = content(c["claim"])
    for e in c["evidence"]:
        d = D[int(e["doc_id"])]
        ov_abs.append(len(cw & content(d["title"] + " " + " ".join(d["abstract"]))) / max(len(cw), 1))
        ov_rat.append(len(cw & content(" ".join(d["abstract"][i] for i in e["sentences"]))) / max(len(cw), 1))
        ov_lab.append(e["label"])
ds["overlap"] = {"abstract_mean": round(np.mean(ov_abs), 3), "rationale_mean": round(np.mean(ov_rat), 3),
                 "abstract_share_below_half": pct(sum(o < 0.5 for o in ov_abs), len(ov_abs)),
                 "abstract_mean_support": round(np.mean([o for o, l in zip(ov_abs, ov_lab) if l == "SUPPORT"]), 3),
                 "abstract_mean_contradict": round(np.mean([o for o, l in zip(ov_abs, ov_lab) if l == "CONTRADICT"]), 3),
                 "hist10": np.histogram(ov_abs, bins=10, range=(0, 1))[0].tolist()}
# negation / numbers by label
NEG = re.compile(r"\b(not|no|never|without|lack|lacks|fail|fails|cannot|unable|absence|neither|nor)\b", re.I)
NUM = re.compile(r"\d")
bylab = collections.defaultdict(list)
for c in lab:
    l = next(iter({e["label"] for e in c["evidence"]}), "NEI")
    bylab[l].append(c["claim"])
ds["cues_by_label"] = {l: {"n": len(v), "negation_pct": pct(sum(bool(NEG.search(x)) for x in v), len(v)),
                           "number_pct": pct(sum(bool(NUM.search(x)) for x in v), len(v))} for l, v in bylab.items()}
# near-duplicate training claim ("twin")
W = {c["id"]: words(c["claim"]) for c in tr}


def twin(c, exclude_self=False):
    cw = words(c["claim"])
    return max(len(cw & W[t["id"]]) / len(cw | W[t["id"]]) for t in tr if not (exclude_self and t["id"] == c["id"]))


tw = {"val": [twin(c) for c in val], "eval": [twin(c) for c in ev], "train": [twin(c, True) for c in tr]}
ds["twins"] = {k: {"share_ge_0.6": pct(sum(x >= 0.6 for x in v), len(v)), "share_ge_0.8": pct(sum(x >= 0.8 for x in v), len(v)),
                   "mean_best_jaccard": round(np.mean(v), 3)} for k, v in tw.items()}
gval = {int(e["doc_id"]) for c in val for e in c["evidence"]}
gtr = {int(e["doc_id"]) for c in tr for e in c["evidence"]}
ds["val_gold_cited_in_train"] = f"{len(gval & gtr)}/{len(gval)}"
N["dataset"] = ds
print(json.dumps(ds, indent=1))
json.dump(N, open(os.path.join(FIG, "analysis_numbers.json"), "w"), indent=1)

# ======================================================================= 2. retrieval (validation)
qv = qrels(val)
HY = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))
RR = json.load(open(os.path.join(CACHE, "rerank_cand_runs.json")))
_, doc_ids = load_corpus()


def run_from_npy(name):
    S = np.load(os.path.join(CACHE, f"{name}_val.npy"))
    return {c["id"]: [int(doc_ids[i]) for i in np.argsort(-S[k])[:100]] for k, c in enumerate(val)}


systems = {
    "BM25": read_trec(os.path.join(ROOT, "predictions", "val_bm25.trec")),
    "bge-base": read_trec(os.path.join(ROOT, "predictions", "val_general.trec")),
    "MedCPT": read_trec(os.path.join(CACHE, "val_scientific.trec")),
    "Hybrid": {c["id"]: HY["val"][str(c["id"])][0] for c in val},
    "Rerank": {c["id"]: RR["val"][str(c["id"])][0] for c in val},
    "FT random": run_from_npy("ft_random"),
    "FT hard": run_from_npy("ft_hard"),
}
rt = {}
ranks = {}
for name, run in systems.items():
    m = evaluate(run, qv)
    rk = [next((i + 1 for i, d in enumerate(run[q]) if d in rel), 101) for q, rel in qv.items()]  # first gold rank
    ranks[name] = rk
    rec = lambda k: float(np.mean([len(set(run[q][:k]) & set(rel)) / len(rel) for q, rel in qv.items()]))
    rt[name] = {"ndcg10": round(m["ndcg10"], 4), "mrr": round(m["mrr"], 4), "R@1": round(rec(1), 3), "R@3": round(rec(3), 3),
                "R@10": round(rec(10), 3), "R@100": round(rec(100), 3),
                "first_gold_rank": {"1": sum(r == 1 for r in rk), "2-3": sum(2 <= r <= 3 for r in rk), "4-10": sum(4 <= r <= 10 for r in rk),
                                    "11-100": sum(11 <= r <= 100 for r in rk), ">100": sum(r > 100 for r in rk)},
                "recall_curve": [round(rec(k), 4) for k in range(1, 101)]}
twin_ids = {c["id"] for c, t in zip(val, tw["val"]) if t >= 0.6}
cited_ids = {c["id"] for c in val if {int(e["doc_id"]) for e in c["evidence"]} & gtr}
for name, run in systems.items():
    from common import per_claim
    pc = per_claim(run, qv)
    rt[name]["ndcg10_gold_cited_in_train"] = round(np.mean([pc[q] for q in qv if q in cited_ids]), 4)
    rt[name]["ndcg10_gold_not_cited"] = round(np.mean([pc[q] for q in qv if q not in cited_ids]), 4)
rt["_n"] = {"claims_with_evidence": len(qv), "gold_cited_in_train": len(cited_ids & set(qv)), "not_cited": len(set(qv) - cited_ids)}
hard = [q for q in qv if ranks["Rerank"][list(qv).index(q)] > 3]
rt["rerank_gold_not_in_top3"] = {str(q): {"claim": next(c["claim"] for c in val if c["id"] == q),
                                          "first_gold_rank": ranks["Rerank"][list(qv).index(q)]} for q in hard}
N["retrieval"] = rt
for name in systems:
    r = rt[name]
    print(f"{name:10s} nDCG {r['ndcg10']:.3f} MRR {r['mrr']:.3f} R@1 {r['R@1']:.3f} R@3 {r['R@3']:.3f} R@10 {r['R@10']:.3f} R@100 {r['R@100']:.3f} "
          f"| cited {r['ndcg10_gold_cited_in_train']:.3f} not {r['ndcg10_gold_not_cited']:.3f} | {r['first_gold_rank']}")
print(rt["_n"])
print("rerank gold not in top 3:", json.dumps(rt["rerank_gold_not_in_top3"], indent=0)[:1500])
json.dump(N, open(os.path.join(FIG, "analysis_numbers.json"), "w"), indent=1)

# ======================================================================= 3. verification (validation, submitted system)
st = json.load(open(os.path.join(CACHE, "t5_stats_v2.json")))
s0, n0, rel0 = st["s"], st["n"], st["rel"]
CEraw = {sp: json.load(open(os.path.join(CACHE, f"ce_ensemble_expanded_{sp}30.json"))) for sp in ["val", "eval"]}
CE = {sp: {cid: dict(zip(HY[sp][cid][0][:30], sc)) for cid, sc in CEraw[sp].items()} for sp in ["val", "eval"]}
NLs = [pickle.load(open(os.path.join(CACHE, f), "rb")) for f in ["nli_probs_nli_rerank_cand_runs.pkl", "nli_probs_nli-ft_rerank_cand_runs.pkl"]]
NL = {"val": [p[0] for p in NLs], "eval": [p[1] for p in NLs]}
gold = {c["id"]: {(int(e["doc_id"]), e["label"]) for e in c["evidence"]} for c in val}
gdoc = {k: {d for d, _ in v} for k, v in gold.items()}


def predict(c, sp, s=s0, n=n0, rel=rel0):
    dl = RR[sp][str(c["id"])][0][:3]
    sc = np.array([CE[sp][str(c["id"])][d] for d in dl])
    rows = []
    for i in np.argsort(-sc)[:n]:
        if sc[i] >= s and sc[i] >= sc.max() - rel:
            pe, pc = np.mean([P[(c["id"], dl[i])] for P in NL[sp]], 0)
            rows.append((dl[i], "SUPPORT" if pe >= pc else "CONTRADICT"))
    return set(rows)


def prf(pred):
    g = {(c["id"],) + x for c in val for x in gold[c["id"]]}
    p = {(cid,) + x for cid, rows in pred.items() for x in rows}
    tp = len(g & p)
    P, R = tp / max(len(p), 1), tp / max(len(g), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


vf = {}
sweep = []
for s in np.linspace(0, 6, 61):
    P, R, F = prf({c["id"]: predict(c, "val", s=s) for c in val})
    sweep.append((round(float(s), 2), round(P, 3), round(R, 3), round(F, 3)))
vf["threshold_sweep"] = sweep
vf["chosen"] = {"s": s0, "n": n0, "rel": rel0, "P_R_F1": [round(x, 4) for x in prf({c["id"]: predict(c, "val") for c in val})]}
pv = {c["id"]: predict(c, "val") for c in val}
for L in ["SUPPORT", "CONTRADICT"]:
    g = {(c["id"],) + x for c in val for x in gold[c["id"]] if x[1] == L}
    p = {(cid,) + x for cid, rows in pv.items() for x in rows if x[1] == L}
    tp = len(g & p)
    vf[f"per_label_{L}"] = {"gold": len(g), "pred": len(p), "P": round(tp / max(len(p), 1), 3), "R": round(tp / max(len(g), 1), 3)}
# label accuracy of the NLI ensemble when given the correct abstract
la = collections.Counter()
for c in val:
    for d, L in gold[c["id"]]:
        if (c["id"], d) in NL["val"][0]:
            pe, pc = np.mean([P[(c["id"], d)] for P in NL["val"]], 0)
            la[(L, "SUPPORT" if pe >= pc else "CONTRADICT")] += 1
vf["nli_label_given_gold_abstract"] = {f"{a}->{b}": v for (a, b), v in la.items()}
vf["nli_label_accuracy"] = round(sum(v for (a, b), v in la.items() if a == b) / sum(la.values()), 3)
# cross-encoder score distribution: gold vs non-gold candidates (rerank top 3)
cg, cn = [], []
for c in val:
    for d in RR["val"][str(c["id"])][0][:3]:
        (cg if d in gdoc[c["id"]] else cn).append(CE["val"][str(c["id"])][d])
vf["ce_scores"] = {"gold_mean": round(np.mean(cg), 2), "nongold_mean": round(np.mean(cn), 2), "n_gold": len(cg), "n_nongold": len(cn),
                   "gold_above_threshold_pct": pct(sum(x >= s0 for x in cg), len(cg)),
                   "nongold_above_threshold_pct": pct(sum(x >= s0 for x in cn), len(cn))}
vf["_ce_gold"], vf["_ce_nongold"] = [round(x, 3) for x in cg], [round(x, 3) for x in cn]
# accuracy by claim cue
for name, f in [("negation", lambda c: bool(NEG.search(c["claim"]))), ("number", lambda c: bool(NUM.search(c["claim"])))]:
    a = [pv[c["id"]] == gold[c["id"]] for c in val if f(c)]
    b = [pv[c["id"]] == gold[c["id"]] for c in val if not f(c)]
    vf[f"claim_accuracy_{name}"] = {"with": f"{sum(a)}/{len(a)}", "without": f"{sum(b)}/{len(b)}"}
N["verification"] = vf
print("chosen", vf["chosen"], {k: vf[k] for k in vf if k.startswith("per_label")})
print("NLI label given gold:", vf["nli_label_given_gold_abstract"], "acc", vf["nli_label_accuracy"])
print("CE:", {k: v for k, v in vf["ce_scores"].items()})
print({k: vf[k] for k in vf if k.startswith("claim_accuracy")})
best = max(sweep, key=lambda x: x[3]); print("sweep best", best, "| at s=0:", sweep[0], "| s=6:", sweep[-1])

# ======================================================================= 4. eval predictions vs validation
ep = collections.defaultdict(set)
for r in csv.DictReader(open(os.path.join(ROOT, "predictions", "t5_verification.csv"))):
    ep[int(r["claim_id"])].add((int(r["doc_id"]), r["label"]))
pvv = collections.defaultdict(set)
for r in csv.DictReader(open(os.path.join(ROOT, "predictions", "t5_val_verification.csv"))):
    pvv[int(r["claim_id"])].add((int(r["doc_id"]), r["label"]))
cmp_ = {}
for sp, cl, P in [("val", val, pvv), ("eval", ev, ep)]:
    top = [max(CE[sp][str(c["id"])][d] for d in RR[sp][str(c["id"])][0][:3]) for c in cl]
    top1_cited = [RR[sp][str(c["id"])][0][0] in gtr for c in cl]
    labs = collections.Counter(next(iter({l for _, l in P[c["id"]]})) if len({l for _, l in P[c["id"]]}) == 1 else ("NEI" if not P[c["id"]] else "MIXED") for c in cl)
    cmp_[sp] = {"claims": len(cl), "pred_NEI_pct": pct(labs["NEI"], len(cl)), "pred_SUPPORT_pct": pct(labs["SUPPORT"], len(cl)),
                "pred_CONTRADICT_pct": pct(labs["CONTRADICT"], len(cl)), "pred_MIXED": labs["MIXED"],
                "rows_per_claim": round(sum(len(P[c["id"]]) for c in cl) / len(cl), 2),
                "top_ce_mean": round(float(np.mean(top)), 2), "top_ce_above_threshold_pct": pct(sum(t >= s0 for t in top), len(cl)),
                "top1_abstract_cited_by_training_claim_pct": pct(sum(top1_cited), len(cl)),
                "twin_ge_0.6_pct": ds["twins"][sp]["share_ge_0.6"], "_top_ce": [round(t, 3) for t in top]}
cmp_["val"]["true_NEI_pct"] = pct(ds["val"]["nei"], ds["val"]["claims"])
N["eval_vs_val"] = cmp_
print({sp: {k: v for k, v in d.items() if not k.startswith("_")} for sp, d in cmp_.items()})
# agreement of top-1 abstract between systems on eval
ev_runs = {n: read_trec(os.path.join(ROOT, "predictions", f)) for n, f in
           [("BM25", "t1_bm25.trec"), ("bge-base", "t2_general.trec"), ("MedCPT", "t2_scientific.trec"), ("Hybrid", "t3_hybrid.trec"),
            ("Rerank", "t3_rerank.trec"), ("FT hard", "t4_hard.trec")]}
names = list(ev_runs)
agree = {a: {b: pct(sum(ev_runs[a][c["id"]][0] == ev_runs[b][c["id"]][0] for c in ev), len(ev)) for b in names} for a in names}
N["eval_top1_agreement"] = agree
print("eval top-1 agreement with Rerank:", {b: agree["Rerank"][b] for b in names})
json.dump(N, open(os.path.join(FIG, "analysis_numbers.json"), "w"), indent=1)
