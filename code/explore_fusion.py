"""Does adding the Task 4 fine-tuned bi-encoder to the hybrid help? min-max fusion grid on validation."""
import itertools
import json
import os

import numpy as np

from bm25 import BM25, make_tokenizer
from common import CACHE, SEED, doc_text, evaluate, fmt, load_claims, load_corpus, per_claim, qrels, topk_from_scores

docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val = load_claims("train"), load_claims("val")
qv = qrels(val)
tok = make_tokenizer()
T = [doc_text(d) for d in docs]
exp = [[] for _ in docs]
for c in tr:
    for e in c["evidence"]:
        exp[pos[int(e["doc_id"])]].append(c["claim"])
Q = [c["claim"] for c in val]
S = {"bm25": BM25(T, tok, 1.2, 0.3).score(Q), "bm25x": BM25([t + " " + " ".join(x) for t, x in zip(T, exp)], tok, 1.2, 0.3).score(Q)}
for k in ["general", "scientific", "ft_hard", "ft_random"]:
    S[k] = np.load(os.path.join(CACHE, f"{k}_val.npy"))


def mm(M):
    t = -np.sort(-M, 1)[:, :100]
    return (M - t[:, -1:]) / (t[:, :1] - t[:, -1:]).clip(1e-9)


N = {k: mm(v) for k, v in S.items()}
run = lambda M: {c["id"]: d for c, (d, _) in zip(val, topk_from_scores(M, ids))}


def boot(a, b, n=5000):
    pa, pb = per_claim(a, qv), per_claim(b, qv)
    d = np.array([pb[q] - pa[q] for q in qv])
    m = d[np.random.default_rng(SEED).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), (m <= 0).mean()


ref = run(N["bm25x"] + 0.5 * N["general"] + 0.25 * N["scientific"])
print("submitted hybrid       ", fmt(evaluate(ref, qv)))
for k in ["ft_hard", "ft_random"]:
    print(f"single {k:10s}      ", fmt(evaluate(run(S[k]), qv)))
grid = [0, 0.25, 0.5, 1.0]
res = []
for lex in ["bm25", "bm25x"]:
    for wg, ws, wf in itertools.product(grid, grid, grid):
        if wf == 0:
            continue
        r = run(N[lex] + wg * N["general"] + ws * N["scientific"] + wf * N["ft_hard"])
        res.append((evaluate(r, qv)["ndcg10"], lex, wg, ws, wf, r))
res.sort(key=lambda x: -x[0])
for nd, lex, wg, ws, wf, r in res[:5]:
    print(f"{lex}+{wg}*gen+{ws}*sci+{wf}*ft_hard  nDCG@10={nd:.4f}  Δ vs submitted=%+.4f p=%.3f" % boot(ref, r))
best_plain = next(x for x in res if x[1] == "bm25")
print(f"best without expansion: bm25+{best_plain[2]}*gen+{best_plain[3]}*sci+{best_plain[4]}*ft_hard nDCG@10={best_plain[0]:.4f}")
