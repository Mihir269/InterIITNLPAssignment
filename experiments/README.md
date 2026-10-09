# Experiment outputs (not part of the submission)

Outputs of the alternative systems that were tried, so every number in `docs/FINDINGS.md` can be checked.
The submitted files are in `predictions/`; anything identical to them was removed from here.
All scores are on the **validation** split. "CV" = 5-fold cross-validation inside validation (tune on 4/5, score the
held-out 1/5), the honest estimate used to decide what to adopt.

## Task 3 – reranking

| Folder / file | What it is | Validation nDCG@10 |
|---|---|---|
| `task3_rerank/t3_first_rerank_bge_reranker.log`, `t3_first_rerank_stats.json` | Full hybrid grid search and the **first rerank attempt** with `BAAI/bge-reranker-base` | hybrid 0.949; rerank **0.870** (hurt) |

The submitted rerank (fine-tuned MedCPT cross-encoder, 0.968) is in `predictions/t3_rerank.trec`. The comparison of
three cross-encoders with plain vs training-claim-expanded text is produced by `code/experiments/explore_rerank.py`.

## Task 4 – bi-encoder fine-tuning

| Folder | What it is | Random negatives | Hard negatives |
|---|---|---|---|
| `task4_finetune/bge_small/` | First run with `bge-small-en-v1.5`: eval runs, training log, per-epoch stats | 0.882 | 0.899 |
| `task4_finetune/bge_base/training.log` | Training log of the **submitted** `bge-base` runs (per-epoch validation and training-claim nDCG; shows overfitting after epoch 1 with random negatives) | 0.906 | 0.920 |

## Task 5 – verification variants

Each folder has `t5_verification_<tag>.csv` (eval), `t5_val_verification_<tag>.csv` and `t5_val_errors_<tag>.csv`.

| Folder | Tags | System | Val F1 (CV) |
|---|---|---|---|
| `01_lexical_baseline/` | `lex0`, `lexhyb0` | No neural model: logistic regression on lexical features, over expanded-BM25 / hybrid candidates | 0.439 / 0.443 |
| | `lex0.6`, `lexhyb0.6` | Same + paired-claim prior | 0.536 / 0.536 |
| `02_zero_shot_nli_plus_pair_prior/` | `nli_pair0.5` … `nli_pair0.8` | Zero-shot NLI (hybrid candidates) + paired-claim prior at word-overlap thresholds 0.5–0.8 (zero-shot NLI alone: 0.608) | 0.633 / 0.650 / 0.669 / 0.667 |
| `03_finetuned_nli/` | `nlift` | NLI fine-tuned on training rationales (hybrid candidates) | 0.627 (0.608) |
| | `nlift_pair0.6`, `nlift_pair0.7` | Same + paired-claim prior | 0.667 / 0.690 (0.682) |
| `04_ce_gate_finetuned_ce_only/` | `ceft` | Fine-tuned cross-encoder decides *if* an abstract is evidence, NLI ensemble decides the label (hybrid candidates) | 0.759 (0.737) |
| `05_ce_gate_ensemble_hybrid_candidates/` | `ceens` | Average of both cross-encoders as the gate (hybrid candidates) — the submission before the last change | 0.771 (0.751) |
| `06_nli_only_rerank_candidates/` | `nli_cand`, `nlift_cand` | NLI alone, zero-shot / fine-tuned, over the t3_rerank top 3 | 0.617 / 0.683 (fine-tuned CV 0.672) |
| **Submitted** (`predictions/`) | — | Gate of `05` over the t3_rerank top 3 | **0.763 (0.763)** |

The **paired-claim prior** (copy a near-duplicate training claim's abstracts with flipped labels) is the only idea that
scored well but was deliberately left out: it bypasses the retriever and NLI classifier the task asks for. See
`docs/FINDINGS.md` and `docs/WALKTHROUGH.md` §13.
