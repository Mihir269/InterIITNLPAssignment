"""Thin wrappers around HuggingFace encoders (bi-encoder, cross-encoder, NLI). CPU/GPU agnostic."""
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())

# name -> (query encoder, doc encoder, pooling, query prefix, doc prefix, max_len)
BI_ENCODERS = {
    "BAAI/bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", None, "cls",
                              "Represent this sentence for searching relevant passages: ", "", 512),
    "BAAI/bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", None, "cls",
                               "Represent this sentence for searching relevant passages: ", "", 512),
    "intfloat/e5-base-v2": ("intfloat/e5-base-v2", None, "mean", "query: ", "passage: ", 512),
    "ncbi/MedCPT": ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", "cls", "", "", 512),
    "pritamdeka/S-PubMedBert-MS-MARCO": ("pritamdeka/S-PubMedBert-MS-MARCO", None, "mean", "", "", 350),
}


def param_count(*models):
    return sum(p.numel() for m in models for p in m.parameters())


class BiEncoder:
    def __init__(self, name):
        spec = BI_ENCODERS.get(name, (name, None, "mean", "", "", 512))  # local dirs: mean pooling
        qname, dname, self.pool, self.qp, self.dp, self.max_len = spec
        self.tok = AutoTokenizer.from_pretrained(qname)
        self.q = AutoModel.from_pretrained(qname, torch_dtype=torch.float32).to(DEV).eval()
        self.d = AutoModel.from_pretrained(dname, torch_dtype=torch.float32).to(DEV).eval() if dname else self.q
        self.dtok = AutoTokenizer.from_pretrained(dname) if dname else self.tok
        self.n_params = param_count(self.q) + (param_count(self.d) if dname else 0)

    def _pool(self, out, mask):
        if self.pool == "cls":
            return out.last_hidden_state[:, 0]
        m = mask.unsqueeze(-1).float()
        return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def _encode(self, model, tok, texts, max_len, bs=32):
        out = []
        order = np.argsort([-len(t) for t in texts])  # length-sorted batches are much faster
        for i in range(0, len(texts), bs):
            batch = [texts[j] for j in order[i:i + bs]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
            e = self._pool(model(**enc), enc["attention_mask"])
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
        emb = np.concatenate(out)
        res = np.empty_like(emb)
        res[order] = emb
        return res

    def encode_queries(self, texts, bs=64):
        return self._encode(self.q, self.tok, [self.qp + t for t in texts], 128, bs)

    def encode_docs(self, docs, bs=32):
        if self.d is not self.q:  # MedCPT article encoder takes (title, abstract) pairs
            return self._encode_pairs(docs, bs)
        return self._encode(self.d, self.dtok, [self.dp + d["title"] + " " + " ".join(d["abstract"]) for d in docs],
                            self.max_len, bs)

    @torch.no_grad()
    def _encode_pairs(self, docs, bs):
        out = []
        for i in range(0, len(docs), bs):
            b = docs[i:i + bs]
            enc = self.dtok([d["title"] for d in b], [" ".join(d["abstract"]) for d in b], padding=True,
                            truncation="only_second", max_length=self.max_len, return_tensors="pt").to(DEV)
            e = self.d(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out)


class CrossEncoder:
    def __init__(self, name, max_len=512):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
        self.max_len = max_len
        self.n_params = param_count(self.m)

    @torch.no_grad()
    def score(self, queries, passages, bs=32):
        out = []
        for i in range(0, len(queries), bs):
            enc = self.tok(queries[i:i + bs], passages[i:i + bs], padding=True, truncation="only_second",
                           max_length=self.max_len, return_tensors="pt").to(DEV)
            logits = self.m(**enc).logits
            out.append((logits[:, 0] if logits.shape[1] == 1 else logits.log_softmax(-1)[:, -1]).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


class NLI:
    """Returns probabilities for (entailment, neutral, contradiction) regardless of the model's label order."""

    def __init__(self, name, max_len=512):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
        self.max_len = max_len
        lab = {v.lower(): k for k, v in self.m.config.id2label.items()}
        find = lambda key: next(i for l, i in lab.items() if l.startswith(key))
        self.idx = [find("entail"), find("neutral") if any(l.startswith("neutral") for l in lab) else None,
                    find("contradict")]
        self.n_params = param_count(self.m)

    @torch.no_grad()
    def predict(self, premises, hypotheses, bs=16):
        out = []
        for i in range(0, len(premises), bs):
            enc = self.tok(premises[i:i + bs], hypotheses[i:i + bs], padding=True, truncation="only_first",
                           max_length=self.max_len, return_tensors="pt").to(DEV)
            p = self.m(**enc).logits.softmax(-1).float().cpu().numpy()
            e, n, c = self.idx
            out.append(np.stack([p[:, e], p[:, n] if n is not None else 1 - p[:, e] - p[:, c], p[:, c]], 1))
        return np.concatenate(out) if out else np.zeros((0, 3))


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0
