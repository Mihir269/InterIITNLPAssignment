# SciFact retrieval + verification — reproduction

Python 3.11. `pip install -r code/requirements.txt`. Unzip the data first: `unzip data-*.zip` (creates `data/`).
All scripts run from `code/`, use seed 42, write runs to `predictions/` and intermediate state to `cache/`
(`PRED_DIR` / `CACHE_DIR` env vars override both). Pretrained weights are pulled from the HuggingFace Hub.

| Task | Command | Outputs |
|---|---|---|
| 1 BM25 (grid search on val) | `python t1_bm25.py` | `t1_bm25.trec`, `val_bm25.trec`, `cache/t1_bm25.json` |
| 2 Dense (general + scientific) | `python t2_dense.py --general BAAI/bge-base-en-v1.5 --scientific ncbi/MedCPT` | `t2_general.trec`, `t2_scientific.trec`, `val_general.trec`, `cache/t2_stats.json` |
| 3 Hybrid + rerank | `python t3_hybrid_rerank.py --reranker BAAI/bge-reranker-base --ks 10,20,30,50,100` | `t3_hybrid.trec`, `t3_rerank.trec`, `cache/t3_stats.json`, `cache/rerank_runs.json` |
| 4 Fine-tuning | `python t4_finetune.py --base BAAI/bge-base-en-v1.5 --epochs 4` | `t4_random.trec`, `t4_hard.trec`, `cache/t4_stats.json` |
| 5 Verification + errors | `python t5_verify.py --scorer nli --nli_model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | `t5_verification.csv`, `t5_val_verification.csv`, `t5_val_errors.csv` |

Order matters: 1 → 2 → 3 → 4/5 (Task 3 reads the Task 1 settings and the Task 2 score matrices; Task 5 reads the reranked runs).

Task 5 variants: `--scorer nli-ft` fine-tunes the NLI model on training rationales; `--scorer lexical` needs no
pretrained weights (with `--run ../cache/bm25x_runs.json` from `python make_bm25_runs.py`); `--pair_prior 0.6`
adds the near-duplicate-training-claim prior. `--tag _x` writes alternative output files without overwriting.

Exploration (no pretrained weights): `python explore_lexical.py` (doc expansion with training claims, RM3, LSA,
fusion), `python explore_ltr.py` (sentence-level BM25, title field, learning-to-rank).

All parameter counts are ≤ 400M: bge-base 110M, MedCPT 2×110M, bge-reranker-base 278M, DeBERTa-v3-base NLI 184M.
