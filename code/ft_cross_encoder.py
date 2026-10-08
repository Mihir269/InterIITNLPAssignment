"""Fine-tune the MedCPT cross-encoder on training claims (pointwise BCE).
Positives: gold abstracts. Negatives: 4 BM25 hard negatives (ranks 1-20, gold removed) + 1 random per evidence claim;
3 top BM25 abstracts per NEI training claim (teaches 'no evidence'). Document text = citing training claims (leave-one-out:
never the claim itself) | title + abstract, as at test time. Val check after each epoch (top-10 rerank nDCG@10); best kept.
Saves to cache/medcpt-ce-ft; then score with `ce_scores.py --model ../cache/medcpt-ce-ft`."""
import argparse
import json
import os
import random

import numpy as np
import torch

from common import CACHE, SEED, doc_text, evaluate, fmt, load_claims, load_corpus, qrels
from models import DEV, CrossEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="ncbi/MedCPT-Cross-Encoder")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=384)
args = ap.parse_args()
random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr, val = load_claims("train"), load_claims("val")
qv = qrels(val)
cites = {d: [] for d in ids.tolist()}
for c in tr:
    for e in c["evidence"]:
        cites[int(e["doc_id"])].append((c["id"], c["claim"]))


def text(d, exclude=None):
    ex = [t for cid, t in cites[d] if cid != exclude]
    return (" ".join(ex) + " | " if ex else "") + doc_text(docs[pos[d]])


H = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))
ex = []
for c in tr:
    gold = {int(e["doc_id"]) for e in c["evidence"]}
    cand = [d for d in H["train"][str(c["id"])][0][:20] if d not in gold]
    if gold:
        for d in gold:
            ex.append((c["claim"], text(d, c["id"]), 1.0))
        for d in random.sample(cand, min(4, len(cand))) + [int(random.choice(ids))]:
            if d not in gold:
                ex.append((c["claim"], text(d, c["id"]), 0.0))
    else:
        for d in cand[:3]:
            ex.append((c["claim"], text(d, c["id"]), 0.0))
print(f"{len(ex)} training pairs ({sum(e[2] for e in ex):.0f} positive)")

ce = CrossEncoder(args.base, max_len=args.max_len)
m = ce.m
opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
steps = args.epochs * ((len(ex) + args.bs - 1) // args.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))


def val_check(k=10):
    run = {}
    for c in val:
        dl = H["val"][str(c["id"])][0]
        sc = ce.score([c["claim"]] * k, [text(d) for d in dl[:k]])
        run[c["id"]] = [dl[i] for i in np.argsort(-sc, kind="stable")] + dl[k:]
    return evaluate(run, qv)


out = os.path.join(CACHE, "medcpt-ce-ft")
log = [dict(epoch=0, val=val_check())]
print("epoch 0 val top-10 rerank", fmt(log[-1]["val"]))
best = log[-1]["val"]["ndcg10"]
m.save_pretrained(out), ce.tok.save_pretrained(out)
for ep in range(1, args.epochs + 1):
    m.train()
    random.shuffle(ex)
    losses = []
    for i in range(0, len(ex), args.bs):
        b = ex[i:i + args.bs]
        enc = ce.tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation="only_second",
                     max_length=args.max_len, return_tensors="pt").to(DEV)
        logit = m(**enc).logits[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([x[2] for x in b], device=DEV))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        losses.append(loss.item())
    m.eval()
    log.append(dict(epoch=ep, loss=float(np.mean(losses)), val=val_check()))
    print(f"epoch {ep} loss {np.mean(losses):.4f} val top-10 rerank", fmt(log[-1]["val"]))
    if log[-1]["val"]["ndcg10"] > best:
        best = log[-1]["val"]["ndcg10"]
        m.save_pretrained(out), ce.tok.save_pretrained(out)
json.dump(dict(args=vars(args), log=log, best=best), open(os.path.join(CACHE, "ce_ft_stats.json"), "w"), indent=1)
print("best val top-10 rerank nDCG@10", round(best, 4), "(epoch 0 = MedCPT-Cross-Encoder zero-shot)")
