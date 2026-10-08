# Session handoff — Inter-IIT NLP selection bootcamp (SciFact)

Branch: `claude/tender-feynman-j3yelk` (all work pushed). Detailed numbers: `FINDINGS.md`. Reproduction commands:
`code/README.md`. Submission answers: `report.md`. This file explains what was done, in what order, and why.

---

## 1. The assignment in one paragraph

Given a scientific claim, find the abstracts in a 5,183-abstract corpus (SciFact) that SUPPORT or CONTRADICT it.
Five tasks: (1) BM25, (2) two pretrained dense retrievers (general + biomedical) with cost numbers, (3) a hybrid and a
cross-encoder rerank, (4) fine-tuning a bi-encoder with random vs hard negatives, (5) NLI-based verification plus
error tagging. Rules: open-source encoders ≤ 400M parameters, no generative LLMs, all tuning/selection on the
training claims and the validation split only; "best" = highest validation nDCG@10. Graded on the hidden labels of
300 eval claims (nDCG@10, R@100, MRR; triple-level F1 for verification), plus the error tags and a 5-question report.
It is a selection assignment with a prescribed pipeline, not an open leaderboard: every required file must follow its
task's recipe.

## 2. Final state (the one submitted version)

All 12 prediction files, `results.json`, `report.md` and `code/README.md` pass `check_format.py` ("All good").
`./make_submission.sh <rollno>` writes the roll number into `results.json` and builds `<rollno>_nlp_bootcamp.zip`
(code/, predictions/, results.json, report.md only), then runs the checker on the zip.

| Task | Submitted system | Validation |
|---|---|---|
| 1 | BM25 (own scipy implementation), k1=1.2 b=0.3, stemming, stopwords, title | nDCG@10 0.865 |
| 2 | BAAI/bge-base-en-v1.5 / ncbi/MedCPT (query + article encoders) | 0.898 / 0.826 |
| 3 hybrid | min-max fusion: BM25 over abstracts + citing training claims (1.0), bge-base (0.5), MedCPT (0.25) | 0.949 |
| 3 rerank | MedCPT-Cross-Encoder fine-tuned on training claims, top-30, doc text = citing training claims + abstract | 0.968 |
| 4 | bge-base fine-tuned, random / hard negatives (best epoch on val) | 0.906 / 0.920 |
| 5 | mean of two cross-encoders picks evidence among t3_rerank top-3; mean of zero-shot + fine-tuned NLI gives the label | F1 0.763 (5-fold CV 0.763) |

The user chose this version over a "by-the-book" one (plain-BM25 hybrid ≈0.915, zero-shot plain-text rerank ≈0.913,
NLI-only verifier F1 0.627) after we discussed that the training-claim expansion and the cross-encoder evidence step
go beyond the expected recipe. Both are allowed by the rules and disclosed in the report.

**Still open for the user:** real name in `results.json` (`YOUR_NAME`); roll number (`ROLLNO`, filled by the script).

## 3. Timeline and reasoning

### Phase 0 — Environment (blockers, and how they were cleared)
- Container: 4 vCPU, 15 GB RAM, **no GPU**. Everything ran on CPU, so all latencies in `results.json` are CPU numbers.
- HuggingFace was blocked by the environment's network policy. The user switched Network access to Custom and added
  `huggingface.co`, then `*.hf.co`, `*.xethub.hf.co`, `*.huggingface.co` (model files are served from Xet/CDN hosts).
- `git push` returned 403 until the user fixed the Claude GitHub App installation.
- GPU options (user's machine, Kaggle, Colab) were discussed; the user chose to stay on CPU.

### Phase 1 — Data exploration (before any model)
Two findings shaped the rest of the project:
- **Gold abstracts recur:** 62 of the 112 validation gold abstracts are gold for some training claim.
- **Claims come in negated pairs:** a training near-duplicate (word Jaccard > 0.7) exists for 69/300 eval claims;
  on validation the twin points to the same abstract with the label flipped 30/35 times, and NEI twins stay NEI 18/18.

### Phase 2 — Task 1 and weight-free ideas (while HuggingFace was blocked)
- BM25 grid (stemming × stopwords × tokenizer × title × k1 × b): 0.865; defaults give 0.863, so tuning is noise.
  Metric verified against `rank_bm25` and `pytrec_eval`.
- **Appending training claims to the abstracts they cite: 0.865 → 0.928 (p < 0.001)**, with no effect on claims
  whose gold abstract was never cited in training. This became the lexical component of the hybrid.
- Rejected: RM3, LSA, max-sentence BM25, title boost, learning-to-rank (all ≤ +0.003 or negative).
- A weight-free verification baseline (logistic regression on lexical features) gave F1 0.44.

### Phase 3 — Tasks 2–5, first complete submission
- Task 2: bge-base 0.898, MedCPT 0.826 (index 18.5 / 24 min, ~150 ms/query on CPU).
- Task 3: hybrid 0.949; the first reranker (bge-reranker-base) **hurt** (0.870): it prefers closely related studies
  over the actual evidence.
- Task 4: bge-small fine-tuned, random 0.882 / hard 0.899.
- Task 5: zero-shot NLI F1 0.608; fine-tuned NLI 0.627 (5-fold CV 0.608 vs 0.593).
- Q1 claim IDs chosen from per-claim nDCG: BM25 wins 673, 1261 (rare exact terms); bge-base wins 766, 1311
  (paraphrases).

### Phase 4 — Improvement search (the user asked for a heavy search)
Biggest wins, in order:
1. **Verification: let the cross-encoder decide *whether* an abstract is evidence and NLI decide only the label.**
   NLI separates gold from non-gold abstracts poorly (AUC 0.68); the MedCPT cross-encoder with expanded text does it
   well (AUC 0.96). F1 0.627 → 0.741; averaging the zero-shot and fine-tuned cross-encoders → 0.771 (CV 0.751).
   Errors fell from 73 to 43; NEI false alarms from 30 to 14.
2. **Reranker:** a biomedical cross-encoder that also sees the citing training claims (0.956), fine-tuned on training
   claims with BM25 hard negatives and leave-one-out expansion (**0.968**).
3. **Task 4 with bge-base:** 0.906 / 0.920. With random negatives, validation peaked at epoch 1 and fell to 0.891
   while training-claim nDCG kept rising — the overfitting evidence used for Q3.

Rejected (documented in FINDINGS.md): adding the fine-tuned bi-encoder or a dense claim-to-claim kNN to the hybrid
(≈ +0.003, noise, same signal as the expansion), a retrieval-score gate for NEI claims (overfits under CV), the
fine-tuned cross-encoder alone for verification (CV 0.737 vs 0.741).

**Selection discipline:** every change was compared with a paired bootstrap (retrieval) or 5-fold cross-validation
inside the validation split (verification) before adoption. Validation has only 106 claims with evidence; a single
tuned threshold is optimistic by ~0.015 F1, and nDCG differences under ~0.01 are not significant.

**Kept out of the submission:** the paired-claim prior (copy a training twin's abstracts with flipped labels:
F1 0.627 → 0.690 on top of NLI-only). It bypasses the retriever and NLI classifier the task asks for. Outputs are in
`exploration/`.

### Phase 5 — Error analysis and the last improvement
`error_analysis.md` (`code/analyze_errors.py`): claim-level confusion, every error listed, error rate by claim property
and by topic (k-means on bge-base claim embeddings). Significant weak spots: multi-document claims, claims with no
training twin, molecular/cell-biology topics. Group-specific thresholds overfit (`explore_groups.py`). The usable finding
was the candidate pool: verifying over the t3_rerank top 3 instead of the hybrid's (118 vs 112 of 124 gold abstracts
reachable) raised 5-fold CV F1 0.751 → 0.763 and cut error claims 43 → 40. Adopted. The fine-tuned NLI model is now
saved in `cache/nli-ft`.

## 4. Bugs and operational lessons

- **DeBERTa NLI checkpoint loads as fp16** (recent `transformers` keeps the stored dtype): fine-tuning gave NaN after one
  step. Fixed by force-loading every model in float32 (`models.py`). Zero-shot results were unchanged.
- **Unsorted NLI/cross-encoder batches** padded short sentences to full-abstract length. Fixed with length-sorted batches.
- **bge-base fine-tuning at batch 16 was killed (exit 137, likely out of memory)**; batch 8 peaks at ~8.2 GB.
- **`pkill -f <pattern>` killed its own shell** three times because the pattern was in the shell's own command line.
  Use PIDs or a bracketed pattern (`[t]4_finetune`).
- **Background tool tasks stop after 2 hours** and a container restart killed one run. Long jobs were launched
  detached (`setsid nohup … & disown`) with `code/run_bg.sh <logname> <script> <args>`, which writes
  `cache/<logname>.log` ending in `EXIT <code>`; short watchers poll for that line.

## 5. What is *not* in git and how to rebuild it

`data/` and `cache/` are git-ignored. `data/` comes from `unzip data-*.zip`. `cache/` holds the score matrices, NLI
probabilities, cross-encoder scores and fine-tuned models that produced the submitted files. To rebuild from scratch
on this CPU container (approximate times):

| Step | Command (from `code/`) | CPU time |
|---|---|---|
| BM25 | `python t1_bm25.py` | 1 min |
| Dense | `python t2_dense.py` | 45 min |
| Hybrid (+ first rerank) | `python t3_hybrid_rerank.py` | 1.5 h |
| Hybrid runs file | `python make_bm25_runs.py && python make_hybrid_runs.py` | 1 min |
| Task 4 | `python t4_finetune.py --base BAAI/bge-base-en-v1.5 --epochs 4 --doc_len 256 --bs 8` | 4–5 h |
| NLI zero-shot / fine-tuned | `python t5_verify.py --run ../cache/hybrid_runs.json --scorer nli --depth 3` (and `--scorer nli-ft … --tag _nlift`) | 45 min / 1.5 h |
| Cross-encoder scores | `python ce_scores.py --splits val,eval` | 1.2 h |
| Fine-tune cross-encoder + rescore | `python ft_cross_encoder.py && python ce_scores.py --model ../cache/medcpt-ce-ft --splits val,eval` | 3–4 h |
| Final rerank | `python make_t3_rerank.py ../cache/medcpt-ce-ft` | 1 min |
| Final verifier | `python make_cand_runs.py`, NLI zero-shot + fine-tuned on `rerank_cand_runs.json` (2.5 h), `python make_ce_ensemble.py`, `python t5_verify_v2.py --run rerank_cand_runs.json --nli <zs>,<ft> --ce "ce_ensemble_expanded_{}30.json"` | 2.5 h |
| results.json | `T5_STATS=t5_stats_v2.json python make_results.py` | instant |

On a GPU all of this takes well under an hour (`models.py` picks CUDA automatically).

## 6. File map

- `code/` — `common.py` (data, metrics, TREC writer), `bm25.py`, `models.py` (bi-encoder, cross-encoder, NLI
  wrappers), task scripts `t1`–`t5`, final-stage scripts (`ce_scores.py`, `ft_cross_encoder.py`, `make_t3_rerank.py`,
  `make_ce_ensemble.py`, `t5_verify_v2.py`, `make_results.py`), exploration scripts (`explore_*.py`), `run_bg.sh`.
- `predictions/` — the submitted files. `exploration/` — alternative outputs and logs (not submitted).
- `report.md` — Q1–Q5 (each ≤ 100 words). `results.json` — filled from saved stats by `make_results.py`.
- `FINDINGS.md` — all numbers, including rejected ideas. `make_submission.sh` — builds and checks the zip.

## 7. If work continues

The most promising next step (report Q5): 32 of the 40 remaining verification errors are about *how many* abstracts
to return (misses, missing second documents, NEI false alarms); only 6 have the wrong label. A small claim-level model
trained on training claims to predict 0/1/2 abstracts from cross-encoder and NLI features should help most. It needs
cross-encoder and NLI scores for the training claims first (≈2 h on CPU).
