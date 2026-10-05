# Findings so far (work in progress)

Status: Task 1 is complete and validated. Tasks 2–5 are fully coded and smoke-tested end-to-end (tiny random
models, outputs pass `check_format.py`), but **not yet run with real models**: this cloud session's network
policy blocks `huggingface.co`, so no pretrained weights can be downloaded here.

All numbers are on the validation split (162 claims; 106 have gold evidence, so one claim ≈ 0.009 nDCG@10).
Δ / p = paired bootstrap vs BM25 (5000 resamples).

## Data
- 5,183 abstracts (≈202 words, ≈9 sentences); train 647 / val 162 / eval 300 claims.
- 35% of claims are NEI. Almost all evidence claims have 1 gold abstract; no claim mixes SUPPORT and CONTRADICT.
- **Gold docs recur across splits**: 62/112 val gold docs are also gold for some training claim.
- **Claims come in near-duplicate (negated) pairs**: 69/300 eval claims have a training claim with >0.7 word Jaccard.
  On val (53 such claims): when the training twin has evidence it is the *same doc with the label flipped* 30/35 times;
  when the twin is NEI the val claim is NEI 18/18 times.

## Task 1 – BM25
| setting | nDCG@10 | R@100 | MRR |
|---|---|---|---|
| default k1=0.9 b=0.4 (stem+stop+title) | 0.8625 | 0.981 | 0.845 |
| tuned k1=1.2 b=0.3 | **0.8653** | 0.981 | 0.850 |
- Stemming matters (+0.024), title helps (+0.008); stopwords and k1/b barely matter — tuning gain is inside noise.
- Verified against `rank_bm25` (0.861) and `pytrec_eval` (identical metric values).

## Retrieval ideas that need no pretrained weights
| idea | nDCG@10 | Δ | p |
|---|---|---|---|
| **Doc expansion: append training claims to their gold abstracts** | **0.9283** | **+0.063** | <0.001 |
| — on claims whose gold doc was never a training gold doc (45) | 0.8868 → 0.8868 | 0 | — |
| — on claims whose gold doc was (61) | 0.8495 → 0.9588 | +0.109 | — |
| RM3 pseudo-relevance feedback (best) | 0.8686 | +0.003 | 0.30 |
| LSA-256 alone / fused with BM25 | 0.673 / ≤0.861 | ≤ −0.004 | — |
| Max-sentence BM25 fused | 0.854 | −0.011 | — |
| Title-field boost | 0.863 | −0.003 | — |
| Learning-to-rank (logreg on lexical features, trained on train claims) | 0.8668 | +0.002 | 0.40 |

Takeaway: lexical signals are saturated; the only big lever is using the training claims as document expansion
(exploits SciFact's paired-claim construction, uses only training data, so within the rules — but worth stating openly).

## Task 5 – verification without pretrained weights (lower bound)
Retriever: expanded BM25. Scorer: logistic regression on lexical features (overlap, negation/direction cue mismatch,
rank, score margin), trained on training claims; n/τ/δ tuned on val F1.
| system | P | R | F1 |
|---|---|---|---|
| lexical scorer | 0.396 | 0.492 | 0.439 |
| + paired-claim prior (Jaccard ≥ 0.6 → copy twin's docs, flip labels) | 0.557 | 0.516 | **0.536** |

## Code ready to run once HuggingFace is reachable
- T2: `BAAI/bge-base-en-v1.5` (general, 110M) and `ncbi/MedCPT` query+article encoders (biomedical, 2×110M).
- T3: z-score / min-max / RRF fusion over {BM25, BM25+expansion} × {general, scientific}; `BAAI/bge-reranker-base` (278M), k chosen on val.
- T4: InfoNCE with in-batch + 1 explicit negative (random vs hard from BM25 ∪ base-dense ranks 3–30, gold removed);
  per-epoch val nDCG and train-subset nDCG to detect overfitting; best epoch kept.
- T5: sentence-level zero-shot NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, 184M) or fine-tuned on training
  rationales; rule-based error tagging (retrieval_miss / needs_multiple_docs / negation / numerical / entity_mismatch / other).
