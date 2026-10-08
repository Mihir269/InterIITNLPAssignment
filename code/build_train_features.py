"""Unbiased features for training claims (for the learned evidence-selection model): zero-shot MedCPT cross-encoder over
leave-one-out expanded abstracts (citing training claims except the claim itself) on each claim's plain-BM25 top-20,
then zero-shot NLI (max over sentences + whole abstract) on the cross-encoder's top 3. Fine-tuned models are not used
because they were trained on these claims. Writes cache/train_sel_features.json."""
import json
import os
import time

import numpy as np

from common import CACHE, doc_text, load_claims, load_corpus
from models import NLI, CrossEncoder

K = 20
docs, ids = load_corpus()
pos = {d: i for i, d in enumerate(ids)}
tr = load_claims("train")
cites = {d: [] for d in ids.tolist()}
for c in tr:
    for e in c["evidence"]:
        cites[int(e["doc_id"])].append((c["id"], c["claim"]))
text = lambda d, own: (lambda ex: (" ".join(ex) + " | " if ex else "") + doc_text(docs[pos[d]]))([t for i, t in cites[d] if i != own])
B = json.load(open(os.path.join(CACHE, "hybrid_runs.json")))["train"]  # plain BM25 (no expansion) for training claims
ce = CrossEncoder("ncbi/MedCPT-Cross-Encoder")
out, t0 = {}, time.time()
for n, c in enumerate(tr):
    dl = B[str(c["id"])][0][:K]
    sc = ce.score([c["claim"]] * K, [text(d, c["id"]) for d in dl])
    o = np.argsort(-sc, kind="stable")
    out[c["id"]] = {"cand": [dl[i] for i in o[:3]], "ce": [float(sc[i]) for i in o[:3]], "ce_all": sc.tolist(), "bm25_docs": dl}
    if n % 100 == 0:
        print(f"CE {n}/{len(tr)} {time.time() - t0:.0f}s", flush=True)
del ce
nli = NLI("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
prem, hyp, key = [], [], []
for c in tr:
    for d in out[c["id"]]["cand"]:
        for s in docs[pos[d]]["abstract"] + [doc_text(docs[pos[d]])]:
            prem.append(s), hyp.append(c["claim"]), key.append((c["id"], d))
p = nli.predict(prem, hyp)
agg = {}
for k, row in zip(key, p):
    a = agg.setdefault(k, [0.0, 0.0])
    a[0], a[1] = max(a[0], float(row[0])), max(a[1], float(row[2]))
for c in tr:
    out[c["id"]]["nli"] = [agg[(c["id"], d)] for d in out[c["id"]]["cand"]]
json.dump(out, open(os.path.join(CACHE, "train_sel_features.json"), "w"))
print(f"done {time.time() - t0:.0f}s")
