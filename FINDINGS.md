# Findings

All numbers are on the validation split (162 claims; 106 have gold evidence, so one claim ≈ 0.009 nDCG@10).
"p" = paired bootstrap (5000 resamples) against the stated reference. All runs on CPU (4 vCPU), seed 42.
Every submitted file passes `check_format.py`.

## Submitted systems

| Run | System | nDCG@10 | R@100 | MRR |
|---|---|---|---|---|
| t1_bm25 | BM25 k1=1.2 b=0.3, stem + stopwords + title | 0.865 | 0.981 | 0.850 |
| t2_general | BAAI/bge-base-en-v1.5 | 0.898 | 1.000 | 0.888 |
| t2_scientific | ncbi/MedCPT query + article encoders | 0.826 | 1.000 | 0.791 |
| t3_hybrid | min-max fusion: BM25 over abstracts + citing training claims (1.0), bge-base (0.5), MedCPT (0.25) | 0.949 | 1.000 | 0.942 |
| t3_rerank | MedCPT-Cross-Encoder **fine-tuned on training claims**, hybrid top-30, doc text = citing training claims + abstract | **0.968** | 1.000 | 0.964 |
| t4_random | bge-base fine-tuned, random negatives (best of 4 epochs: 1) | 0.906 | 0.991 | 0.900 |
| t4_hard | bge-base fine-tuned, BM25 ∪ dense hard negatives (best epoch: 2) | 0.920 | 1.000 | 0.914 |

| Verification | P | R | F1 | 5-fold CV F1 |
|---|---|---|---|---|
| zero-shot DeBERTa-v3 NLI, threshold on max(entail, contradict) | 0.603 | 0.613 | 0.608 | 0.593 |
| fine-tuned NLI (training rationales + mined neutrals) | 0.624 | 0.629 | 0.627 | 0.608 |
| MedCPT cross-encoder selects evidence, NLI ensemble labels it | 0.711 | 0.774 | 0.741 | 0.741 |
| same with the fine-tuned cross-encoder only | 0.769 | 0.750 | 0.759 | 0.737 |
| **submitted: mean of both cross-encoders selects evidence, NLI ensemble labels it** | 0.768 | 0.774 | **0.771** | **0.751** |

Errors (submitted verifier): 43/162 validation claims; retrieval_miss 4, entity_mismatch 14, needs_multiple_docs 10,
other 9, numerical 5, negation 1. By outcome: 15 partially right (missing/extra abstract), 14 NEI false alarms
(30 with NLI alone), 8 misses, 6 wrong label.

## What mattered (in order of size)

1. **Cross-encoder as the evidence selector in verification: F1 0.608 → 0.741 (CV).** NLI is good at *which label*
   (gold SUPPORT: mean p(entail) 0.74; gold CONTRADICT: mean p(contradict) 0.89) but bad at *whether an abstract is
   evidence* (pair AUC 0.68). The MedCPT cross-encoder with expanded text gets pair AUC 0.96 and separates NEI claims
   at AUC 0.90 (hybrid retrieval score: 0.72). Split the two jobs.
2. **Training claims as document expansion: BM25 0.865 → 0.928 (p<0.001).** 62/112 val gold abstracts are gold for some
   training claim (SciFact pairs claims with their negations). No effect on the other 45 claims (0.887 → 0.887).
3. **Hybrid fusion: +0.021 over the best single system** (0.915 vs 0.898 without expansion).
4. **Domain + expansion + fine-tuning for the reranker.** bge-reranker-base: 0.870 (hurts). ms-marco-MiniLM: 0.890 → 0.940 with
   expanded text. MedCPT-Cross-Encoder: 0.913 → 0.956 with expanded text → 0.968 after fine-tuning on training claims (+0.019 over the hybrid, p=0.10). Fusing reranker and first-stage scores
   reaches 0.962 but the task format requires pure reranker order.
5. **Hard negatives: 0.920 vs 0.906 for random negatives** (bge-base, 0.877 before training at 256 tokens; bge-small: 0.899
   vs 0.882). Overfitting: the random-negative run peaks at epoch 1 (0.906) and falls to 0.891 by epoch 3 while
   training-claim nDCG rises 0.960 → 0.972; the hard-negative run peaks at epoch 2 (0.920 → 0.917). Batch 16 ran
   out of memory on CPU for bge-base, so batch 8 was used.

## Tried and rejected (no significant gain)

| Idea | Result |
|---|---|
| BM25 k1/b tuning beyond defaults | +0.003 (noise) |
| RM3 pseudo-relevance feedback | +0.003, p=0.30 |
| LSA "dense" retriever / fused | 0.673 / −0.004 |
| Max-sentence BM25, title boost | −0.011 / −0.003 |
| Learning-to-rank on lexical features | +0.002, p=0.40 |
| Adding fine-tuned bi-encoder to the hybrid | +0.003, p=0.30 (but 0.915 → 0.926 without expansion) |
| Dense claim-to-claim kNN in the hybrid | +0.0025, p=0.12 (0.915 → 0.942 without expansion: same signal as the expansion) |
| Retrieval-score gate for NEI claims | CV F1 0.571 vs 0.593 (overfits) |

## Bugs found on the way

- The DeBERTa NLI checkpoint is stored in fp16 and `transformers` now keeps that dtype: fine-tuning produced NaN after
  one step. All models are now force-loaded in float32 (zero-shot results unchanged).
- Unsorted NLI batches padded every sentence to full-abstract length; length-sorting made scoring several times faster.

## Optional, not in the submitted files: paired-claim prior

If a training claim is a near-duplicate (word Jaccard ≥ 0.7) of the test claim, copy its gold abstracts with flipped
labels (30/35 val pairs with evidence flip; 18/18 NEI twins stay NEI). On top of fine-tuned NLI: F1 0.627 → 0.690
(CV 0.682). Left out because it bypasses the retriever and the NLI classifier the task asks for. Files in
`exploration/`.

## Caveats

- Validation is small: a single tuned threshold is optimistic by ≈0.015 F1 (in-sample vs 5-fold CV), and nDCG
  differences below ≈0.01 are not significant.
- Expansion and the paired-claim structure exploit how SciFact was built; they use only training claims (allowed by
  the rules) but would not transfer to a corpus without labelled neighbouring claims.
