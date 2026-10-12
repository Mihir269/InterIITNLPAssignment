"""Draw docs/figures/*.png from docs/figures/analysis_numbers.json (written by data_analysis.py).
Palette: reference categorical slots 1-4 (validated, light surface); thin marks, recessive grid, direct labels."""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # shared modules live in code/

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT

FIG = os.path.join(ROOT, "docs", "figures")
N = json.load(open(os.path.join(FIG, "analysis_numbers.json")))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def finish(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


ds, rt, vf, ev = N["dataset"], N["retrieval"], N["verification"], N["eval_vs_val"]

# 1. label composition (100% stacked, with a 2px surface gap between segments)
fig, ax = plt.subplots(figsize=(7, 2.3))
for row, sp in enumerate(["val", "train"]):
    d = ds[sp]
    left = 0
    for lab, key, col in [("SUPPORT", "support", S1), ("CONTRADICT", "contradict", S2), ("NOT ENOUGH INFO", "nei", S3)]:
        w = 100 * d[key] / d["claims"]
        ax.barh(row, w, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=2, label=lab if row == 0 else None)
        ax.text(left + w / 2, row, f"{w:.0f}%", ha="center", va="center", color=INK, fontsize=9)
        left += w
ax.set_yticks([0, 1], [f"Validation ({ds['val']['claims']})", f"Train ({ds['train']['claims']})"])
ax.set_xlim(0, 100); ax.set_xticks([])
ax.set_title("Label mix per split (share of claims)")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
finish(ax, "x"); save(fig, "01_label_mix.png")

# 2. where evidence sentences sit
h = np.array(ds["rationale_position"]["hist10"]); h = 100 * h / h.sum()
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(np.arange(10) * 10 + 5, h, width=8, color=S1)
for x, v in zip(np.arange(10) * 10 + 5, h):
    ax.text(x, v + 0.4, f"{v:.0f}", ha="center", color=INK2, fontsize=8)
ax.set_xticks([0, 20, 40, 60, 80, 100], ["first", "20%", "40%", "60%", "80%", "last"])
ax.set_xlabel("position of the evidence sentence within its abstract"); ax.set_ylabel("% of evidence sentences")
ax.set_title(f"Evidence sits at the end of abstracts ({ds['rationale_position']['n']} rationale sentences)")
finish(ax); save(fig, "02_evidence_position.png")

# 3. lexical overlap claim vs gold abstract
h = np.array(ds["overlap"]["hist10"]); h = 100 * h / h.sum()
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(np.arange(10) * 10 + 5, h, width=8, color=S1)
for x, v in zip(np.arange(10) * 10 + 5, h):
    ax.text(x, v + 0.4, f"{v:.0f}", ha="center", color=INK2, fontsize=8)
ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
ax.set_xlabel("share of the claim's content words that appear in its gold abstract (stemmed)")
ax.set_ylabel("% of (claim, abstract) pairs")
ax.set_title(f"Claim words found in the gold abstract (mean {100 * ds['overlap']['abstract_mean']:.0f}%)")
finish(ax); save(fig, "03_word_overlap.png")

# 4. retrieval ablation
order = ["BM25", "MedCPT", "bge-base", "FT random", "FT hard", "Hybrid", "Rerank"]
vals = [rt[o]["ndcg10"] for o in order]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hlines(range(len(order)), 0.8, vals, color=GRID, linewidth=1)
ax.scatter(vals, range(len(order)), s=70, color=S1, edgecolor=SURF, linewidth=2, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(order)), ["BM25 (T1)", "MedCPT (T2)", "bge-base (T2)", "Fine-tuned, random neg. (T4)",
                                  "Fine-tuned, hard neg. (T4)", "Hybrid (T3)", "Rerank (T3)"])
ax.set_xlim(0.8, 1.0); ax.set_xlabel("validation nDCG@10")
ax.set_title("Retrieval systems, validation nDCG@10")
finish(ax, "x"); save(fig, "04_retrieval_ndcg.png")

# 5. recall@k curves
fig, ax = plt.subplots(figsize=(7, 3.6))
ks = np.arange(1, 101)
for name, col in [("BM25", S1), ("bge-base", S2), ("Hybrid", S3), ("Rerank", S4)]:
    y = np.array(rt[name]["recall_curve"])
    ax.plot(ks, y, color=col, linewidth=2, label=name)
    ax.text(5.3, y[4] + {"Rerank": 0.004, "Hybrid": -0.012}.get(name, 0), f"{name}", color=INK, fontsize=8, va="center", ha="left",
            bbox=dict(boxstyle="square,pad=0.1", facecolor=SURF, edgecolor="none"))
ax.set_xscale("log"); ax.set_xlim(1, 100)
ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
ax.set_xlabel("k (number of abstracts kept, log scale)"); ax.set_ylabel("recall@k")
ax.set_title("Share of gold abstracts found in the top k (validation)")
ax.legend(loc="lower right"); finish(ax, "both"); save(fig, "05_recall_at_k.png")

# 6. cited vs not cited in training
names = ["BM25", "bge-base", "Hybrid", "Rerank"]
a = [rt[n]["ndcg10_gold_cited_in_train"] for n in names]; b = [rt[n]["ndcg10_gold_not_cited"] for n in names]
fig, ax = plt.subplots(figsize=(7, 3.4))
y = np.arange(len(names))[::-1]
for yi, va_, vb_ in zip(y, a, b):
    ax.hlines(yi, min(va_, vb_), max(va_, vb_), color=GRID, linewidth=2)
ax.scatter(a, y, s=70, color=S1, edgecolor=SURF, linewidth=2, zorder=3, label=f"gold abstract cited by a training claim ({rt['_n']['gold_cited_in_train']} claims)")
ax.scatter(b, y, s=70, color=S2, edgecolor=SURF, linewidth=2, zorder=3, marker="s", label=f"not cited ({rt['_n']['not_cited']} claims)")
for yi, va_, vb_ in zip(y, a, b):
    lo, hi = (va_, vb_) if va_ < vb_ else (vb_, va_)
    ax.text(lo - 0.004, yi, f"{lo:.3f}", ha="right", va="center", fontsize=8, color=INK)
    ax.text(hi + 0.004, yi, f"{hi:.3f}", ha="left", va="center", fontsize=8, color=INK)
ax.set_yticks(y, names); ax.set_xlim(0.82, 1.0); ax.set_ylim(-0.6, len(names) - 0.2); ax.set_xlabel("validation nDCG@10")
ax.set_title("Gains hold on claims the training-claim expansion cannot help")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=8); finish(ax, "x"); save(fig, "06_cited_vs_not.png")

# 7. threshold sweep
sw = np.array(vf["threshold_sweep"], dtype=float)
sw[sw[:, 2] == 0, 1] = np.nan  # no predictions -> precision undefined, not 0 or 1
sw = sw[sw[:, 0] <= 5]
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, (lab, col) in enumerate([("Precision", S1), ("Recall", S2), ("F1", S3)], start=1):
    ax.plot(sw[:, 0], sw[:, i], color=col, linewidth=2, label=lab)
s0 = vf["chosen"]["s"]
ax.axvline(s0, color=INK2, linewidth=1, linestyle="--")
ax.text(s0 + 0.08, 0.05, f"chosen threshold {s0:.2f}\nF1 {vf['chosen']['P_R_F1'][2]:.3f}", color=INK, fontsize=8)
ax.set_xlim(0, 5); ax.set_ylim(0, 1.02)
ax.set_xlabel("cross-encoder evidence threshold (z-scored ensemble score)"); ax.set_ylabel("validation score")
ax.set_title("Verification: precision / recall trade-off of the evidence threshold")
ax.legend(loc="lower left"); finish(ax, "both"); save(fig, "07_threshold_sweep.png")

# 8. cross-encoder score distribution, gold vs non-gold candidates
g, ng = np.array(vf["_ce_gold"]), np.array(vf["_ce_nongold"])
bins = np.linspace(min(g.min(), ng.min()), max(g.max(), ng.max()), 30)
fig, ax = plt.subplots(figsize=(7, 3.2))
for arr, col, lab in [(ng, S2, f"not gold ({len(ng)} candidates)"), (g, S1, f"gold ({len(g)} candidates)")]:
    hh, _ = np.histogram(arr, bins=bins)
    ax.step(bins[:-1], 100 * hh / hh.sum(), where="post", color=col, linewidth=2, label=lab)
ax.axvline(s0, color=INK2, linewidth=1, linestyle="--")
ax.text(s0 - 0.1, ax.get_ylim()[1] * 0.62, f"threshold: {vf['ce_scores']['gold_above_threshold_pct']:.0f}% of gold,\n"
        f"{vf['ce_scores']['nongold_above_threshold_pct']:.0f}% of non-gold pass", fontsize=8, color=INK, ha="right")
ax.set_xlabel("cross-encoder ensemble score (rerank top-3 candidates, validation)"); ax.set_ylabel("% of candidates")
ax.set_title("The cross-encoder separates evidence from non-evidence")
ax.legend(loc="upper left", fontsize=8); finish(ax, "both"); save(fig, "08_ce_scores.png")

# 9. eval vs validation
items = [("predicted NOT ENOUGH INFO", "pred_NEI_pct"), ("top candidate passes threshold", "top_ce_above_threshold_pct"),
         ("top abstract cited by a training claim", "top1_abstract_cited_by_training_claim_pct"),
         ("near-duplicate training claim", "twin_ge_0.6_pct")]
x = np.arange(len(items)); w = 0.36
fig, ax = plt.subplots(figsize=(7, 3.4))
va = [ev["val"][k] for _, k in items]; vb = [ev["eval"][k] for _, k in items]
ax.bar(x - w / 2 - 0.01, va, w, color=S1, label="validation (162)")
ax.bar(x + w / 2 + 0.01, vb, w, color=S2, label="eval (300)")
for xi, (p, q) in enumerate(zip(va, vb)):
    ax.text(xi - w / 2 - 0.01, p + 1, f"{p:.0f}", ha="center", fontsize=8, color=INK)
    ax.text(xi + w / 2 + 0.01, q + 1, f"{q:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, ["predicted\nNEI", "top candidate\npasses threshold", "top abstract cited\nby a training claim", "has a near-duplicate\ntraining claim"], fontsize=8)
ax.set_ylabel("% of claims"); ax.set_ylim(0, 80)
ax.set_title("Why eval gets more NOT ENOUGH INFO than validation")
ax.legend(loc="upper right"); finish(ax); save(fig, "09_eval_vs_val.png")
print(sorted(f for f in os.listdir(FIG) if f.endswith(".png")))
