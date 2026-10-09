# Code — reproducing every submitted file

## Setup

```bash
pip install -r code/requirements.txt          # Python 3.11; torch, transformers, scikit-learn, PyStemmer, ...
unzip assignment/data-*.zip                    # from the repo root: creates data/ (corpus + claims)
cd code                                        # every command below runs from code/
```

- Pretrained weights are downloaded from the HuggingFace Hub on first use (all models ≤ 400M parameters:
  bge-base 110M, MedCPT encoders 2×110M, MedCPT cross-encoder 110M, bge-reranker-base 278M, DeBERTa-v3-base NLI 184M).
- Seed 42 everywhere. Submitted files go to `predictions/`; intermediate state (score matrices, fine-tuned models,
  NLI probabilities, stats) goes to `cache/` (git-ignored). `PRED_DIR` / `CACHE_DIR` override both.
- Times below are for a 4-vCPU machine without GPU; a GPU is used automatically if present (minutes instead of hours).

## Pipeline, in order

| # | Command | Produces | CPU time |
|---|---|---|---|
| 1 | `python t1_bm25.py` | `t1_bm25.trec`, `val_bm25.trec` (grid search over k1, b, stemming, stop words, tokenizer, title) | 1 min |
| 2 | `python t2_dense.py` | `t2_general.trec` (bge-base), `t2_scientific.trec` (MedCPT), `val_general.trec`; timings and memory | 45 min |
| 3 | `python t3_hybrid_rerank.py` | `t3_hybrid.trec` (min-max fusion grid); also the first rerank attempt with bge-reranker-base, reported in Q2 (its `t3_rerank.trec` is replaced in step 8) | 1.5 h |
| 4 | `python make_bm25_runs.py && python make_hybrid_runs.py` | `cache/hybrid_runs.json` (hybrid rankings; plain BM25 rankings for training claims) | 1 min |
| 5 | `python ce_scores.py --splits val,eval` | zero-shot MedCPT cross-encoder scores for the hybrid top 30 | 1.2 h |
| 6 | `python ft_cross_encoder.py` | `cache/medcpt-ce-ft` (cross-encoder fine-tuned on training claims) | 3 h |
| 7 | `python ce_scores.py --model ../cache/medcpt-ce-ft --splits val,eval` | fine-tuned cross-encoder scores | 1.2 h |
| 8 | `python make_t3_rerank.py` | **`t3_rerank.trec`** (fine-tuned cross-encoder, k chosen on validation) | 1 min |
| 9 | `python t4_finetune.py --base BAAI/bge-base-en-v1.5 --epochs 4 --doc_len 256 --bs 8` | `t4_random.trec`, `t4_hard.trec` (best validation epoch of each) | 5 h |
| 10 | `python make_cand_runs.py` | `cache/rerank_cand_runs.json` (t3_rerank order = Task 5 candidates) | 1 min |
| 11 | `PRED_DIR=../cache python t5_verify.py --run ../cache/rerank_cand_runs.json --scorer nli --depth 3` | zero-shot NLI probabilities | 45 min |
| 12 | `PRED_DIR=../cache python t5_verify.py --run ../cache/rerank_cand_runs.json --scorer nli-ft --depth 3` | NLI fine-tuned on training rationales (`cache/nli-ft`) and its probabilities | 1.5 h |
| 13 | `python make_ce_ensemble.py` | average of zero-shot and fine-tuned cross-encoder scores | 1 min |
| 14 | `python t5_verify_v2.py` | **`t5_verification.csv`, `t5_val_verification.csv`, `t5_val_errors.csv`** | 1 min |
| 15 | `T5_STATS=t5_stats_v2.json python make_results.py` | `results.json`, filled from the stats saved by every step | instant |

Steps 8 and 14 run with their defaults set to the submitted configuration and were checked to reproduce the
submitted files byte for byte from the cached scores. Step 9 is independent of steps 3–8 and can run in parallel.

## Scripts

| File | Role |
|---|---|
| `common.py` | Data loading, metrics (nDCG@10, Recall@100, MRR), TREC writer, paths |
| `bm25.py` | Sparse-matrix BM25 (Lucene IDF) with configurable tokenizer (stemming, stop words) |
| `models.py` | Wrappers for bi-encoders, cross-encoders and NLI models (float32, length-sorted batching) |
| `t1_bm25.py` … `t5_verify_v2.py` | The pipeline steps above |
| `ce_scores.py`, `ft_cross_encoder.py`, `make_t3_rerank.py` | Cross-encoder scoring, fine-tuning, final rerank |
| `make_bm25_runs.py`, `make_hybrid_runs.py`, `make_cand_runs.py`, `make_ce_ensemble.py` | Intermediate run/score files |
| `make_results.py` | Writes `results.json` from cached stats |
| `build_train_features.py`, `fit_selection_model.py` | Report Q5 experiment: learned evidence selection trained on training claims (0.745 held-out vs 0.763 for the submitted rule; not used in the submission) |

## How the final system works (short)

- **Retrieval:** BM25 over each abstract plus the training claims that cite it, fused (min-max) with bge-base and MedCPT
  dense scores (weights 1.0 / 0.5 / 0.25) → hybrid top 30 re-ordered by the fine-tuned MedCPT cross-encoder.
- **Verification:** for each claim's top 3 reranked abstracts, the average of the zero-shot and fine-tuned cross-encoder
  scores decides whether it is evidence (threshold tuned on validation); the average of zero-shot and fine-tuned
  DeBERTa NLI probabilities (max over sentences) gives SUPPORT vs CONTRADICT; no rows = NOT ENOUGH INFO.

## Implementation notes

- All models are force-loaded in float32: the DeBERTa NLI checkpoint is stored in fp16 and produces NaN when fine-tuned
  in fp16.
- Inputs are length-sorted before batching; otherwise short sentences are padded to full-abstract length.
- bge-base fine-tuning needs batch 8 on a 15 GB machine (batch 16 runs out of memory).
- During cross-encoder fine-tuning, a training claim never sees its own text inside the abstract it is matched against
  (leave-one-out expansion).

Exploration scripts for ideas that were tried and not used (RM3, LSA, learning-to-rank, reranker comparison, error
analysis, …) are in the GitHub repository under `code/experiments/`; they are not needed to reproduce the submission.
