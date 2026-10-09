# Finding Evidence for Scientific Claims (SciFact)

**Inter-IIT Tech Meet 15.0 — NLP Selection Bootcamp** · Mihir Gune (24b2443)

Given a scientific claim, find the abstracts in a 5,183-abstract corpus that support or contradict it (**retrieval**),
then decide which (**verification**); if no abstract does, the claim is NOT ENOUGH INFO. This is the retrieval and
grounding core of a RAG system, built with open-source encoders only (≤ 400M parameters, no generative LLMs), with all
tuning and model selection on the training and validation claims.

## Results (validation split)

| Task | System | nDCG@10 | Recall@100 | MRR |
|---|---|---|---|---|
| 1 | BM25, tuned (k1 = 1.2, b = 0.3, stemming, stop words, title) | 0.865 | 0.981 | 0.850 |
| 2 | Dense, general: `BAAI/bge-base-en-v1.5` | 0.898 | 1.000 | 0.888 |
| 2 | Dense, biomedical: `ncbi/MedCPT` query + article encoders | 0.826 | 1.000 | 0.791 |
| 3 | Hybrid: BM25 over abstracts + citing training claims, bge-base, MedCPT (min-max fusion 1.0 / 0.5 / 0.25) | 0.949 | 1.000 | 0.942 |
| 3 | Rerank: hybrid top 30 by MedCPT cross-encoder fine-tuned on training claims | **0.968** | 1.000 | 0.964 |
| 4 | bge-base fine-tuned, random negatives | 0.906 | 0.991 | 0.900 |
| 4 | bge-base fine-tuned, hard negatives (from our BM25 + dense rankings) | 0.920 | 1.000 | 0.914 |

| Task 5 – verification | Precision | Recall | F1 | 5-fold CV F1 |
|---|---|---|---|---|
| Zero-shot NLI (starting point) | 0.603 | 0.613 | 0.608 | 0.593 |
| **Submitted: cross-encoders select evidence, NLI ensemble labels it** | 0.760 | 0.766 | **0.763** | **0.763** |

Validation has 162 claims (106 with evidence), so retrieval differences below ≈ 0.01 nDCG@10 and verification
differences below ≈ 0.02 F1 are within noise. Every adopted change was checked with a paired bootstrap test
(retrieval) or 5-fold cross-validation inside validation (verification). All timings are CPU numbers (no GPU).

## The final system

```
claim
  │
  ├─ BM25 over (abstract + training claims that cite it) ─┐
  ├─ bge-base dense similarity ───────────────────────────┼─ min-max fusion → hybrid top 100        (t3_hybrid)
  └─ MedCPT dense similarity ─────────────────────────────┘
                                                            │
                     fine-tuned MedCPT cross-encoder re-orders the top 30                           (t3_rerank)
                                                            │
                                         top 3 abstracts per claim
                                                            │
   is it evidence?  → average of zero-shot + fine-tuned cross-encoder score ≥ threshold
   which direction? → average of zero-shot + fine-tuned DeBERTa NLI (max over sentences)            (t5_verification)
   nothing passes   → NOT ENOUGH INFO
```

## What made the difference

1. **Splitting verification into two jobs** — F1 0.63 → 0.76. NLI is good at *direction* but poor at deciding whether
   an abstract is evidence at all (AUC 0.68); a biomedical cross-encoder does that far better (AUC 0.96). Letting the
   cross-encoder decide relevance and NLI decide the label cut false alarms on NOT-ENOUGH-INFO claims from 30 to 10.
2. **Using training claims as document expansion** — BM25 0.865 → 0.928 (p < 0.001). 62 of 112 validation gold
   abstracts are also gold for a training claim (SciFact pairs each claim with negated/sibling claims). No effect on
   claims whose gold abstract is never cited, so it does not hurt anything.
3. **Hybrid fusion** — +0.021 over the best single retriever; keyword and dense search fail on different claims
   (rare exact terms vs paraphrases, report Q1).
4. **The right reranker** — a general cross-encoder *hurt* (0.870: it prefers related studies over the annotated
   evidence); a biomedical one with expanded text reached 0.956, and 0.968 after fine-tuning on training claims.
5. **Hard negatives** — 0.920 vs 0.906 for random negatives; random-negative training overfits after one epoch
   (validation 0.906 → 0.891 while training-claim nDCG rises 0.960 → 0.972).
6. **Verifying over the reranker's top 3** instead of the hybrid's — 118 vs 112 of 124 gold abstracts reachable;
   CV F1 0.751 → 0.763.

## What did not help

| Idea | Result |
|---|---|
| BM25 parameter tuning beyond defaults | +0.003 (noise) |
| RM3 pseudo-relevance feedback | +0.003, p = 0.30 |
| LSA retrieval / fused with BM25 | 0.673 / −0.004 |
| Sentence-level BM25, title boosting | −0.011 / −0.003 |
| Learning-to-rank on lexical features | +0.002, p = 0.40 |
| Fine-tuned bi-encoder or dense claim matching added to the hybrid | +0.003 / +0.0025 (same signal as the expansion) |
| Retrieval-score gate for NOT ENOUGH INFO | CV 0.571 vs 0.593 (overfits) |
| Group-specific thresholds, label-bias correction | CV 0.743–0.748 vs 0.751 (overfit) |
| Learned "how many abstracts" model trained on training claims | 0.745 held-out vs 0.763 (unbiased features had to come from weaker zero-shot models) |

Deliberately **not submitted**: the *paired-claim prior* (copy a near-duplicate training claim's abstracts with flipped
labels; F1 0.627 → 0.690 on top of NLI alone). It bypasses the retriever and NLI classifier the task asks for.

## Repository layout

```
├── README.md                  ← you are here
├── report.md                  ← submission: answers to Q1–Q5 (≤ 100 words each)
├── results.json               ← submission: settings, models, validation scores, timings, memory
├── predictions/               ← submission: 7 eval runs, 2 validation runs (Q1), 3 Task 5 CSVs
├── code/                      ← submission: pipeline scripts + README with exact reproduction commands
│   ├── experiments/           ←   exploration scripts for ideas that were tried (not in the zip)
│   └── tools/                 ←   helper for long background jobs
├── experiments/               ← outputs of every alternative system, grouped by task, with a README
├── docs/
│   ├── WALKTHROUGH.md         ← in-depth explanation of every idea: intuition, implementation, result, why
│   ├── FINDINGS.md            ← all numbers, including rejected ideas
│   ├── ERROR_ANALYSIS.md      ← confusion matrix, every validation error, error rate by claim type and topic
│   └── SESSION_HANDOFF.md     ← chronological log of the work, decisions, and how to rebuild cached state
├── submission/
│   └── 24b2443_nlp_bootcamp.zip   ← the submitted zip (passes check_format.py)
├── assignment/                ← problem statement, results template, data zip (as provided)
├── check_format.py            ← official format checker (as provided)
└── make_submission.sh         ← builds the zip from code/, predictions/, results.json, report.md and checks it
```

## Reproducing

```bash
pip install -r code/requirements.txt
unzip assignment/data-*.zip            # creates data/
cd code && python t1_bm25.py ...       # full ordered list of 15 commands with times: code/README.md
cd .. && ./make_submission.sh 24b2443  # rebuilds submission/24b2443_nlp_bootcamp.zip and runs the checker
```

Pretrained models are downloaded from the HuggingFace Hub. On a 4-vCPU machine the full pipeline takes roughly 15 hours
(fine-tuning dominates); with a GPU it takes well under an hour. The final steps (`make_t3_rerank.py`,
`t5_verify_v2.py`) reproduce the submitted files byte for byte from cached scores.

## Where to read more

| If you want… | Read |
|---|---|
| The answers to the five questions | `report.md` |
| Why each design choice was made, step by step | `docs/WALKTHROUGH.md` |
| Every number, including failed experiments | `docs/FINDINGS.md`, `experiments/README.md` |
| Which claims the system gets wrong and why | `docs/ERROR_ANALYSIS.md` |
| How to run each step | `code/README.md` |

## Limitations

- The training-claim expansion and the reranker's use of citing training claims work because of how SciFact was built
  (sibling and negated claims share abstracts). They use only training data, as the rules allow, but would not transfer
  to a corpus without labelled neighbouring claims.
- 32 of the 40 remaining validation errors are about *how many* abstracts to return (missed second abstracts, false
  alarms, misses); only 6 have the wrong label. The next step would be cross-fitted fine-tuning of the cross-encoder so a
  learned selection model can be trained on unbiased but strong scores (report Q5).
