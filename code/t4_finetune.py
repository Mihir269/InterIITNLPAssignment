"""Task 4: fine-tune a bi-encoder on training claims with (a) random negatives, (b) hard negatives mined from
our own Task 1-2 systems (BM25 + base dense model). In-batch negatives + 1 explicit negative, InfoNCE loss.
Validation nDCG@10 after every epoch; the best epoch is kept (and the gap to train nDCG shows overfitting)."""
import argparse
import json
import os
import random

import numpy as np
import torch

from bm25 import BM25, make_tokenizer
from common import (CACHE, PRED, SEED, doc_text, evaluate, fmt, load_claims, load_corpus, qrels,
                    topk_from_scores, write_trec)
from models import DEV, BiEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--doc_len", type=int, default=256)
ap.add_argument("--hard_from", type=int, default=3, help="skip the top ranks when mining (false negatives)")
ap.add_argument("--hard_to", type=int, default=30)
ap.add_argument("--modes", default="random,hard")
args = ap.parse_args()

docs, doc_ids = load_corpus()
pos = {d: i for i, d in enumerate(doc_ids)}
tr, val, ev = load_claims("train"), load_claims("val"), load_claims("eval")
qv, qt = qrels(val), qrels(tr)
dtext = [doc_text(d) for d in docs]
pairs = [(c["claim"], pos[int(e["doc_id"])], c["id"]) for c in tr for e in c["evidence"]]
print(f"{len(pairs)} (claim, gold doc) training pairs from {len(qt)} claims")


def seed_all(s=SEED):
    random.seed(s), np.random.seed(s), torch.manual_seed(s)


def mine_hard():
    """union of BM25 and base-dense rankings for every training claim, ranks hard_from..hard_to, gold removed"""
    best1 = json.load(open(os.path.join(CACHE, "t1_bm25.json")))["best"]
    bm = BM25(dtext, make_tokenizer(best1["stem"], best1["stop"], best1["pattern"]), best1["k1"], best1["b"])
    Sb = bm.score([c["claim"] for c in tr])
    enc = BiEncoder(args.base)
    Sd = enc.encode_queries([c["claim"] for c in tr]) @ enc.encode_docs(docs).T
    hard = {}
    for i, c in enumerate(tr):
        gold = {pos[int(e["doc_id"])] for e in c["evidence"]}
        cand = []
        for S in (Sb, Sd):
            cand += [j for j in np.argsort(-S[i])[args.hard_from - 1:args.hard_to] if j not in gold]
        hard[c["id"]] = list(dict.fromkeys(cand))
    return hard


def eval_model(enc, claims, D=None):
    D = enc.encode_docs(docs) if D is None else D
    S = enc.encode_queries([c["claim"] for c in claims]) @ D.T
    return {c["id"]: d for c, (d, _) in zip(claims, topk_from_scores(S, doc_ids))}, D, S


def train(mode, hard):
    seed_all()
    enc = BiEncoder(args.base)
    enc.max_len = args.doc_len
    model = enc.q  # shared query/doc encoder
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(pairs) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    trsub = random.Random(SEED).sample([c for c in tr if c["evidence"]], 100)
    qsub = {c["id"]: qt[c["id"]] for c in trsub}
    log, best = [], (-1, None)
    run0, _, _ = eval_model(enc, val)
    log.append(dict(epoch=0, val=evaluate(run0, qv), train=evaluate(eval_model(enc, trsub)[0], qsub)))
    print(mode, "epoch 0", fmt(log[-1]["val"]))

    def emb(texts, max_len, prefix):
        b = enc.tok([prefix + t for t in texts], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
        return torch.nn.functional.normalize(enc._pool(model(**b), b["attention_mask"]), dim=-1)

    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pairs)
        losses = []
        for i in range(0, len(pairs), args.bs):
            b = pairs[i:i + args.bs]
            negs = []
            for q, p, cid in b:
                gold = {pos[int(e["doc_id"])] for c in tr if c["id"] == cid for e in c["evidence"]}
                if mode == "hard" and hard[cid]:
                    negs.append(random.choice(hard[cid][:10]))
                else:
                    n = random.randrange(len(docs))
                    while n in gold:
                        n = random.randrange(len(docs))
                    negs.append(n)
            q = emb([x[0] for x in b], 64, enc.qp)
            d = emb([dtext[x[1]] for x in b] + [dtext[n] for n in negs], args.doc_len, enc.dp)
            logits = q @ d.T / args.temp
            # mask in-batch "negatives" that are actually the same gold doc of another pair
            same = torch.tensor([[x[1] == y for y in [z[1] for z in b] + negs] for x in b], device=DEV)
            same[torch.arange(len(b)), torch.arange(len(b))] = False
            logits = logits.masked_fill(same, -1e4)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(b), device=DEV))
            loss.backward(), opt.step(), sched.step(), opt.zero_grad()
            losses.append(loss.item())
        model.eval()
        run, D, S = eval_model(enc, val)
        log.append(dict(epoch=ep, loss=float(np.mean(losses)), val=evaluate(run, qv), train=evaluate(eval_model(enc, trsub, D)[0], qsub)))
        print(f"{mode} epoch {ep} loss {np.mean(losses):.4f} train-nDCG {log[-1]['train']['ndcg10']:.4f} val", fmt(log[-1]["val"]))
        if log[-1]["val"]["ndcg10"] > best[0]:
            Se = enc.encode_queries([c["claim"] for c in ev]) @ D.T
            best = (log[-1]["val"]["ndcg10"], ep, S, Se)
    _, ep, Sv, Se = best
    write_trec(os.path.join(PRED, f"t4_{mode}.trec"), [c["id"] for c in ev], topk_from_scores(Se, doc_ids), f"ft_{mode}")
    np.save(os.path.join(CACHE, f"ft_{mode}_val.npy"), Sv), np.save(os.path.join(CACHE, f"ft_{mode}_eval.npy"), Se)
    return dict(best_epoch=ep, best_val_ndcg10=best[0], log=log)


hard = mine_hard() if "hard" in args.modes else None
out = {"base": args.base, "args": vars(args)}
for mode in args.modes.split(","):
    out[mode] = train(mode, hard)
json.dump(out, open(os.path.join(CACHE, "t4_stats.json"), "w"), indent=1)
