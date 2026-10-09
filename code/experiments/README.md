# Exploration scripts (not needed to reproduce the submission)

Run from `code/`, e.g. `python experiments/explore_lexical.py`. They read the data and the `cache/` files written by the
pipeline (see `code/README.md`); most take seconds to minutes because they reuse cached scores.

| Script | Question it answers | Result (validation) |
|---|---|---|
| `explore_lexical.py` | Do document expansion with training claims, RM3, LSA, or BM25+LSA fusion help BM25? | Expansion 0.865 → **0.928** (p < 0.001), no change on claims whose gold abstract is uncited; RM3 +0.003 (p = 0.30); LSA 0.673, fusion ≤ 0.861 |
| `explore_ltr.py` | Do sentence-level BM25, a title field, or learning-to-rank on lexical features help? | Sentence −0.011, title −0.003, LTR +0.002 (p = 0.40) |
| `explore_rerank.py` | Which cross-encoder, with plain vs training-claim-expanded text, pure vs fused with first-stage scores? | MiniLM 0.890 → 0.940, MedCPT 0.913 → **0.956** with expansion; bge-reranker 0.870; fused best 0.962 |
| `explore_fusion.py` | Does adding the Task 4 fine-tuned bi-encoder to the hybrid help? | +0.003 (p = 0.30); 0.915 → 0.926 without expansion |
| `explore_claimknn.py` | Does dense claim-to-claim matching (new claim vs citing training claims) help? | +0.0025 (p = 0.12); 0.915 → 0.942 without expansion (same signal as expansion) |
| `analyze_errors.py` | Where does the verifier fail? Writes `docs/ERROR_ANALYSIS.md` sections 1–6 | 40/162 errors; 32 are about how many abstracts to return |
| `explore_groups.py` | Do separate thresholds per claim group (training twin, topic) or a label-bias term help? | CV 0.745 / 0.748 / 0.743 vs 0.751 — overfit |
| `explore_count_model.py` | Can a learned evidence-selection model trained on validation (nested CV) beat thresholds? | 0.745–0.747 vs 0.763 |

The training-claim version of the learned selection model (`build_train_features.py`, `fit_selection_model.py`) is in
`code/` because report Q5 cites it.
