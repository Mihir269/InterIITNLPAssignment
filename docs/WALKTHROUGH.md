# In-depth walkthrough: everything tried, why, and what we learned

This is the long-form explanation of the whole project. Each section follows the same pattern: **the idea → why we
expected it to help → how it was built → what happened → why it worked or failed.** All numbers are on the validation
split unless stated otherwise (162 claims; 106 of them have at least one correct abstract, so for retrieval one claim is
worth ≈ 0.009 nDCG@10).

Contents
1. The problem in one picture
2. Looking at the data first (and the two discoveries that shaped everything)
3. Task 1 – BM25
4. Ideas without neural models (expansion, RM3, LSA, sentences, titles, learning-to-rank)
5. Task 2 – Dense retrieval
6. Task 3a – Hybrid fusion
7. Task 3b – Reranking (from harmful to best system)
8. Task 4 – Fine-tuning a bi-encoder
9. Task 5 – Claim verification (from 0.44 to 0.763 F1)
10. Error analysis and the attempts to use it
11. How we decided what to keep (statistics)
12. Engineering problems and their fixes
13. What we chose not to submit, and why
14. What remains and what to try next

---

## 1. The problem in one picture

```
             5,183 abstracts (fixed corpus)
                        │
claim ──► [ RETRIEVE: which abstracts are about this? ] ──► ranked list of 100   (Tasks 1–4, scored by nDCG@10)
                        │
                        ▼
          [ VERIFY: does each top abstract SUPPORT or CONTRADICT it? ] ──► 0–3 (abstract, label) rows   (Task 5, scored by F1)
```

Two different questions: *relevance* (is this abstract about the claim?) and *stance* (does it agree or disagree?).
A big lesson of this project was that these need different models (Section 9).

---

## 2. Looking at the data first

**What the data is.** 5,183 abstracts (≈ 202 words, ≈ 9 sentences each). Claims: 647 train, 162 validation, 300 eval.
About 35% of labelled claims are "not enough info" (NEI: no correct abstract). Most claims with evidence have exactly
one correct abstract; a few have 2–4. No claim mixes SUPPORT and CONTRADICT.

**Discovery 1 – correct abstracts repeat across splits.** 62 of the 112 abstracts that are correct for validation
claims are *also* correct for some training claim. So for more than half of validation, a training claim already
"points at" the right abstract.

**Discovery 2 – claims come in opposite pairs.** SciFact was built by writing a claim from a paper and then writing a
negated version. Example: train claim 0 "0-dimensional biomaterials *lack* inductive properties" and eval claim 1
"0-dimensional biomaterials *show* inductive properties". On validation, 53 claims have a near-duplicate training
claim (word overlap ≥ 0.6). When that twin has evidence, it is the *same abstract with the opposite label* in 30 of 35
cases; when the twin is NEI, the validation claim is NEI 18 of 18 times.

**Why this mattered.** Both discoveries say: *the training claims are an extremely informative signal for the test
claims.* Using them in the retrieval (Discovery 1) became the single biggest retrieval gain. Using Discovery 2 directly
("copy the twin's answer and flip it") also works, but we decided it bends the task too far (Section 13).

---

## 3. Task 1 – BM25

**The idea.** BM25 scores an abstract by the claim words it contains, with three refinements:
- *rare words count more* (IDF: "prion" is informative, "the" is not);
- *repetition helps with diminishing returns* (controlled by k1);
- *long documents are discounted* (controlled by b).

**How it was built.** Our own implementation with sparse matrices (so a full grid search runs in under a minute):
Lucene-style IDF, k1 and b as parameters, and a configurable tokenizer. Grid: k1 ∈ {0.4 … 2.0} × b ∈ {0.3 … 1.0} ×
stemming on/off × stop-word removal on/off × two tokenizers × include title or not = 896 configurations.

**What happened.**

| Setting | nDCG@10 |
|---|---|
| Best: k1 = 1.2, b = 0.3, stemming, stop words removed, title included | **0.865** |
| Standard default k1 = 0.9, b = 0.4 | 0.863 |
| Best without stemming | 0.841 |
| Best without the title | 0.858 |

**Why.** Stemming matters most (+0.024): claims and abstracts use different word forms ("reduces" / "reduction").
Titles help because they summarise the paper's main finding. k1/b tuning gains only 0.003, which is noise on 106
claims. The metric was cross-checked against two libraries (`pytrec_eval` gave identical numbers; `rank_bm25` gave
0.861), so the scorer is trustworthy.

**Why 0.865 is so much higher than published BM25 on SciFact (~0.67):** the published number is on the official test
set; this validation split is simply easier. It is a reminder that numbers are only comparable on the same split.

---

## 4. Ideas without neural models

These were tried early, while HuggingFace was blocked, but they are informative on their own.

### 4.1 Document expansion with training claims — **the biggest retrieval win**
**Idea.** Append to every abstract the text of the training claims that cite it. A new claim that resembles a training
claim now matches that abstract word-for-word.
**Result.** BM25 0.865 → **0.928** (+0.063, bootstrap p < 0.001).
**Why it works – and the important check.** Split the validation claims in two:
- claims whose correct abstract is cited by a training claim (61): 0.850 → 0.959;
- claims whose correct abstract is never cited (45): 0.887 → 0.887, **no change**.

So the gain comes entirely from Discovery 1, and it does not hurt anything else: adding text to *other* abstracts did not
pull them above correct ones. (Adding it as a separate weighted field instead of appending was weaker: 0.897.)
**Is it allowed?** Yes: the rules permit using training claims for anything. But it works because of how SciFact was
built, and it would not transfer to a corpus without labelled neighbouring claims. The report says so.

### 4.2 RM3 (pseudo-relevance feedback)
**Idea.** Assume the top few results are relevant, take their most characteristic words, add them to the query, search
again. Classic fix for vocabulary mismatch.
**Result.** Best 0.869 (+0.003, p = 0.30): noise. Some settings were slightly worse.
**Why.** The first pass is already good (correct abstract usually at rank 1), so feedback mostly re-adds words that were
already matched. When the top results are wrong, feedback amplifies the mistake.

### 4.3 LSA as a "dense" retriever
**Idea.** Compress word counts into 256 "topics" with SVD, so related words share dimensions — a pre-neural form of
semantic search.
**Result.** Alone 0.673; fused with BM25 it made things slightly worse (≤ 0.861).
**Why.** LSA captures broad topics ("cardiology") but not specific meaning; on a corpus where many abstracts share a topic
it cannot tell the right one apart. Real neural embeddings (Section 5) do much better.

### 4.4 Sentence-level BM25 and title boosting
**Idea.** Evidence is usually 1–3 sentences, so score each abstract by its best-matching sentence; separately, give the
title extra weight.
**Result.** Best sentence alone 0.807; fused −0.011. Title alone 0.628; fused −0.003 to −0.023.
**Why.** A single sentence has too few words for BM25 statistics to be reliable, and the claim often paraphrases several
sentences at once. The title is already included once in the full text; weighting it more over-rewards topical titles.

### 4.5 Learning-to-rank on lexical features
**Idea.** Train a small logistic regression on training claims to combine BM25 score, sentence score, title score,
rank, abstract length.
**Result.** 0.867 (+0.002, p = 0.40).
**Why.** The learned weights were dominated by "how close is this score to the top score", i.e. it mostly re-learned
BM25's own ranking. All features came from the same word-matching signal, so there was nothing new to combine.
**Lesson:** lexical signals were saturated; further gains had to come from *meaning* (neural models) or *external
knowledge* (training claims).

### 4.6 A verification baseline with no neural model
Logistic regression on hand-made features (word overlap, negation words in claim vs sentence, direction words like
"increase"/"decrease", rank, score margin), trained on training claims: **F1 0.44**. With the paired-claim prior:
0.54. This set the floor that any NLI system had to beat.

---

## 5. Task 2 – Dense retrieval

**The idea.** A neural *bi-encoder* turns any text into a vector (768 numbers) such that texts with similar meaning get
similar vectors. Abstracts are encoded once ("indexing"); each claim needs one encoding plus a similarity computation.

**Models.**
- General: `BAAI/bge-base-en-v1.5` (110M parameters), a strong general-purpose retriever.
- Biomedical: `ncbi/MedCPT` — separate query and article encoders (2 × 110M), trained on PubMed search logs.

**Results.**

| Model | nDCG@10 | Recall@100 | Index time (CPU) | Latency / query | Memory (model + index) |
|---|---|---|---|---|---|
| bge-base | **0.898** | 1.000 | 18.5 min | 152 ms | 433 MB |
| MedCPT | 0.826 | 1.000 | 24 min | 138 ms | 850 MB |

**Why the general model beat the biomedical one.** Surprising at first. MedCPT is trained on *search queries* (short
keyword-style) against PubMed articles; our claims are full declarative sentences, closer to what bge was trained on.
Both still find every correct abstract within the top 100 (Recall@100 = 1.0), which matters for later stages.

**Q1 analysis: why keyword and meaning-based search fail differently.** Per claim, BM25 won on 14, bge on 16, 76 tied.
- BM25 wins on **rare exact terms**: claim 673 ("*LDL cholesterol* has a causal role…") — BM25 rank 1 because the
  correct title says "LDL-cholesterol"; bge drifted to a general cholesterol history review. Claim 1261 ("reactive
  oxygen species… genomic instability") — same pattern.
- bge wins on **paraphrase**: claim 766 ("*Medications* to treat obesity…") — the correct abstract says
  "*pharmacotherapy*", BM25 ranked it 67th, bge 1st. Claim 1311 ("lncRNAs") vs "LincRNA-p21".

This complementarity is exactly what motivates the hybrid.

---

## 6. Task 3a – Hybrid fusion

**The idea.** Each retriever has blind spots; an abstract that *both* rank highly is very likely correct.

**How it was built.** Scores live on different scales (BM25 ≈ 10–30, cosine ≈ 0.5–0.8), so they must be normalised.
Three methods were compared, each with a grid of weights:
- *z-score* (subtract mean, divide by standard deviation, per claim);
- *min-max* (rescale each claim's top 100 to 0–1);
- *RRF* (reciprocal rank fusion: use ranks, ignore scores).

Components: BM25 or expanded BM25, bge, MedCPT, in all combinations.

**Result.** Best: min-max with `1.0 × expanded BM25 + 0.5 × bge + 0.25 × MedCPT` → **0.949**
(best single system was expanded BM25 at 0.928, so +0.021). Without expansion, the best hybrid (BM25 + bge) reached
0.915 vs 0.898 for bge alone.

**Why min-max won.** RRF throws away *how confident* each scorer is: a BM25 score that is far above the rest (a
near-exact match) counts the same as a marginal one. Min-max keeps relative confidence. The MedCPT weight is small
because it was the weakest retriever, but it still adds a little independent evidence.

**Q2 cost.** The hybrid needs all three scorers: ≈ 290 ms/query on CPU instead of 152 ms.

---

## 7. Task 3b – Reranking: from harmful to best system

**The idea.** Bi-encoders encode claim and abstract *separately*, which is fast but shallow. A *cross-encoder* reads them
*together* in one pass, so every claim word can attend to every abstract word. Far more accurate, far slower — so it is
only applied to the top-k of the first stage.

### 7.1 First attempt: it made things worse
`BAAI/bge-reranker-base` (278M) on the hybrid's top-k: **0.870 at k = 10**, falling further with larger k (0.851 at
k = 50). 2.5 s/query on CPU.

**Diagnosis.** On 20 claims it demoted a correct #1. I first guessed it preferred broad review papers — checking showed
that was true for only 2 of 20, so that explanation was wrong. The real pattern: it preferred a **closely related study
on the same topic** (e.g. claim 34 about folate and homocysteine: it chose another folate–homocysteine trial over the
annotated one). A general reranker scores *topical relevance*; SciFact asks for *the specific evidence*. It also cannot
see the training-claim expansion that made the hybrid strong.

### 7.2 Systematic comparison
Three cross-encoders × plain vs expanded document text, on the hybrid's top 30:

| Reranker | Plain text | With citing training claims prepended |
|---|---|---|
| `bge-reranker-base` (general) | 0.870 | — (stopped: lowest value, slowest) |
| `ms-marco-MiniLM-L-6` (small, web search) | 0.890 | 0.940 |
| `MedCPT-Cross-Encoder` (biomedical) | 0.913 | **0.956** |

**Why.** Two independent fixes stack: (1) a *biomedical* cross-encoder understands the domain (+0.04 over the general
ones); (2) giving it the *training claims that cite each abstract* hands it the same strong signal the hybrid uses
(+0.04 more).

Fusing reranker and first-stage scores reached 0.962, but the task format requires *pure reranker order*, so that was
not used.

### 7.3 Fine-tuning the cross-encoder
**How.** 3,179 training pairs from training claims: 440 correct (claim, abstract) pairs labelled 1; for each claim 4 hard
negatives (wrong abstracts from its BM25 top 20) + 1 random abstract labelled 0; for NEI training claims, their top 3
BM25 abstracts labelled 0 (teaches "not evidence"). Binary cross-entropy, 2 epochs, learning rate 2e-5.
**Crucial detail – leave-one-out expansion.** During training, an abstract's text includes the training claims that cite
it *except the claim being trained on*. Otherwise the model would learn "find my own sentence in the text", which never
happens at test time.
**Result.** Top-10 check: 0.953 → 0.962 (epoch 1) → 0.963 (epoch 2). Full rerank at k = 30: **0.968** (+0.019 over
the hybrid; 13 claims improved, 7 hurt; p = 0.10). 9.2 s/query on CPU — about 30× the hybrid's cost.

### 7.4 Two more retrieval ideas that did not help
- *Adding the Task 4 fine-tuned bi-encoder to the hybrid:* +0.003 (p = 0.30). Without expansion it does help
  (0.915 → 0.926) — the fine-tuned model learned part of the same "training claims" signal, so with expansion present
  it is redundant.
- *Dense claim-to-claim matching* (score an abstract by the similarity between the new claim and the training claims
  that cite it): +0.0025 (p = 0.12). Again, without expansion it helps a lot (0.915 → 0.942). **Same signal, different
  route** — once you have it, a second copy adds nothing.

---

## 8. Task 4 – Fine-tuning a bi-encoder

**The idea.** Adapt a general embedding model to "claim ↔ evidence abstract" matching using the training claims.

**How it was built.** Contrastive learning (InfoNCE). Each batch contains several (claim, correct abstract) pairs plus one
extra *negative* abstract per pair. The model must give each claim the highest similarity with its own abstract among
*all* abstracts in the batch (other pairs' abstracts act as free negatives). Temperature 0.05, learning rate 2e-5 with
warm-up. If two claims in a batch share the same correct abstract, that "negative" is masked so the model is not punished
for a correct answer.

Two variants (what the task asks to compare):
- **random negatives** — a random abstract;
- **hard negatives** — a wrong abstract that *our own* BM25 or base model ranked 3rd–30th (ranks 1–2 skipped because
  they are sometimes unlabelled correct answers).

Validation nDCG@10 is computed after every epoch, plus nDCG on 100 *training* claims to watch for overfitting.

**Results.**

| Model | Before | Random negatives (best epoch) | Hard negatives (best epoch) |
|---|---|---|---|
| bge-small (first run) | 0.870 | 0.882 | 0.899 |
| **bge-base (submitted)** | 0.877* | **0.906** (epoch 1) | **0.920** (epoch 2) |

*0.877 rather than 0.898 because abstracts were cut to 256 tokens for training speed on CPU.

**Why hard negatives help.** A random abstract is usually about something completely different, so the model learns
little from rejecting it. A hard negative is on the same topic but not the evidence — exactly the confusion the model
makes at test time. Same pattern for both model sizes.

**Overfitting (Q3).** bge-base with random negatives:

| Epoch | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Validation nDCG@10 | **0.906** | 0.896 | 0.891 | 0.892 |
| Training-claim nDCG@10 | 0.960 | 0.962 | 0.967 | 0.972 |

Training performance keeps rising while validation falls — the model is memorising the 440 training pairs. Detected by
tracking both curves each epoch; handled by keeping the best validation epoch. Hard negatives overfit less (validation
peaked at epoch 2: 0.920 → 0.917) because they keep the task difficult.

**Why bge-base and not bge-small.** It simply did better (0.920 vs 0.899). It needed batch size 8 instead of 16 to fit in
CPU memory (Section 12).

---

## 9. Task 5 – Claim verification: from 0.44 to 0.763 F1

Scored by F1 over exact (claim, abstract, label) triples. Every stage below uses the same decision structure: look at the
top-3 abstracts, keep those that pass a threshold, label each one, at most 3 rows; no rows = NEI. Thresholds are tuned on
validation; "CV" means 5-fold cross-validation inside validation (tune on 4/5, score on the held-out 1/5) — an honest
estimate that removes the optimism of tuning on the same claims you score.

### 9.1 Zero-shot NLI — F1 0.608 (CV 0.593)
**Model.** `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (184M), trained on large NLI datasets to output
entailment / neutral / contradiction for a (premise, hypothesis) pair.
**How.** Premise = each abstract sentence (and the whole abstract once), hypothesis = the claim. Take the maximum
entailment and maximum contradiction over sentences. If the larger one exceeds a threshold, predict that label.
**Why sentence-level.** Evidence is usually one or two sentences; diluting it in a 250-word premise weakens the signal.

### 9.2 Fine-tuned NLI — F1 0.627 (CV 0.608)
**How.** Training examples from training claims: their *rationale sentences* (the `sentences` field) labelled
entailment or contradiction; neutral examples from other sentences of the same abstract and from wrong abstracts
(2,455 examples, 2 epochs).
**First run: complete failure (F1 = 0).** The probabilities were all NaN. Cause: the checkpoint is stored in float16 and
the library now keeps that precision; training in float16 without loss scaling overflowed after one step. Fixed by
forcing float32 for every model (zero-shot results were unchanged).
**Result after the fix.** +0.019 in-sample, +0.015 under CV — small but consistent.

### 9.3 Why NLI alone plateaus — the key diagnosis
Breaking the zero-shot errors down: **30 of 68 errors were false alarms on NEI claims** (predicted evidence where there is
none). Measuring how well each signal separates "is this abstract evidence?":

| Signal | Claim has evidence vs NEI (AUC) | Abstract is the correct evidence (AUC) |
|---|---|---|
| Hybrid retrieval score | 0.72 | — |
| NLI confidence | 0.74 | 0.68 |
| MedCPT cross-encoder (plain text) | 0.83 | 0.92 |
| **MedCPT cross-encoder + expanded text** | **0.90** | **0.96** |

(AUC: 0.5 = guessing, 1.0 = perfect.) NLI is good at *direction* — on correct abstracts, mean p(entail) for SUPPORT is
0.74 and mean p(contradict) for CONTRADICT is 0.89 — but bad at *relevance*. It fires on any sentence that sounds
different from the claim: in the pharmacist example (claims 600/601), background and methods sentences scored
contradiction ≈ 1.0 even though they are not evidence.

### 9.4 A failed fix: gate by retrieval score
"Only predict when the retrieval score is high." In-sample 0.612; CV **0.571** (worse than 0.593). Two thresholds on 162
claims overfit, and the retrieval score is a weak relevance signal (AUC 0.72).

### 9.5 The real fix: split the two jobs — F1 0.741 (CV 0.741)
**Design.** For each of the top-3 abstracts:
1. *Is it evidence?* → cross-encoder score ≥ s and within `rel` of the claim's best score (at most n abstracts);
2. *Which direction?* → NLI: entailment vs contradiction.

| Label model | CV F1 |
|---|---|
| zero-shot NLI | 0.659 |
| fine-tuned NLI | 0.726 |
| **average of both NLI models** | **0.741** |

**Why it works.** Each model does the job it is good at. Averaging the two NLI models reduces the label noise of each.
In-sample and CV scores were identical (0.741), meaning the thresholds are not overfit.

### 9.6 Adding the fine-tuned cross-encoder — F1 0.771 (CV 0.751)
- Fine-tuned cross-encoder alone as the gate: in-sample 0.759, CV 0.737 — *not* better under CV.
- **Average of zero-shot and fine-tuned cross-encoder scores** (each standardised with validation statistics): in-sample
  0.771, CV 0.751. Errors 45 → 43. Adopted.
**Why averaging helps.** The two cross-encoders make different mistakes; the fine-tuned one is sharper on claims like the
training ones, the zero-shot one is more robust on unusual claims.

### 9.7 Better candidates — final F1 0.763 (CV 0.763)
**Diagnosis from the error analysis.** The verifier only examined the *hybrid's* top 3, which contain 112 of 124 correct
abstracts. The *rerank's* top 3 contain 118. Examples: claim 382's evidence was hybrid rank 8 but rerank rank 1; claim
1311's was hybrid rank 5, rerank 1.
**How.** Candidates = `t3_rerank` top 3; both NLI models recomputed on the ≈ 550 new (claim, abstract) pairs; the
fine-tuned NLI was retrained (and saved this time).
**Result.** CV 0.751 → **0.763**; in-sample 0.771 → 0.763 (lower), errors 43 → 40, retrieval misses 4 → 2,
multi-abstract errors 10 → 6. Adopted on the pre-agreed criterion (CV must improve), and because in-sample = CV again
shows a robust rule. It also matches the task wording ("using your best retriever").

### 9.8 Summary of the verification journey

| System | In-sample F1 | CV F1 |
|---|---|---|
| Lexical features (no neural model) | 0.44 | — |
| Zero-shot NLI | 0.608 | 0.593 |
| Fine-tuned NLI | 0.627 | 0.608 |
| Cross-encoder gate + NLI ensemble | 0.741 | 0.741 |
| + both cross-encoders | 0.771 | 0.751 |
| **+ rerank candidates (submitted)** | **0.763** | **0.763** |

---

## 10. Error analysis and the attempts to use it

### 10.1 What the remaining 40 errors look like

| True \ predicted | SUPPORT | CONTRADICT | NEI |
|---|---|---|---|
| SUPPORT (65) | 55 | 3 | 7 |
| CONTRADICT (40) | 4 | 32 | 4 |
| NEI (56) | 7 | 3 | 46 |

By outcome: 11 claims miss all evidence, 11 miss a second abstract or add an extra one, 10 are false alarms on NEI
claims, only **6** have the wrong label, 2 pick the wrong abstract. **32 of 40 errors are about *how many* abstracts to
return, not about the label.**

Typical label errors are subtle direction/quantity mismatches: claim 98 says alirocumab *reduces* a clearance rate (the
abstract says otherwise); claim 1156 says DESMOND caused *substantial* weight loss (it did not). Typical false alarms are
claims whose abstract is on exactly the right topic but never states the claim (claims 63/64: "ATM and Rad3 are critical
for / have no role in sensing DNA damage" — both NEI in the gold data).

### 10.2 Which claims are hard (Fisher exact test)

| Pattern | Error rate | p |
|---|---|---|
| More than one correct abstract | 67% (8/12) vs 21% | 0.002 |
| No near-duplicate training claim | 32% vs 9% | 0.002 |
| Molecular / cell-biology topics vs clinical / population | 29% vs 12% | 0.023 |

Topics came from clustering claim embeddings (k-means on bge vectors, clusters named by their most distinctive words);
the first attempt with word-count clustering failed (one cluster held 102 of 162 claims). Clinical claims tend to restate
an abstract's conclusion almost verbatim; molecular claims paraphrase mechanisms.

### 10.3 Trying to exploit the patterns
- **Separate thresholds** for claims with/without a training twin: CV 0.745; for clinical vs molecular: 0.748;
  a bias term correcting the slight lean toward SUPPORT: 0.743 — all below 0.751. **Why:** splitting 162 claims into
  groups leaves ~50–100 per group, too few to tune extra parameters reliably; the extra flexibility fits noise.
- **Better candidates** (Section 9.7) — the one finding that *did* translate into a gain.

### 10.4 The learned "how many abstracts" model
**Idea.** Since most errors are about how many abstracts to return, learn the decision instead of using fixed
thresholds: a classifier scores each candidate ("is this evidence?") from its cross-encoder score, rank, gap to the
best candidate, NLI confidence and margin, plus claim features (best/second score, training twin, claim length).

**Attempt 1 – trained on validation (nested CV): 0.745–0.747.** Too little data: ~130 claims per training fold for a
10-feature model, versus a 3-number rule.

**Attempt 2 – trained on the 647 training claims.** The trap: the fine-tuned cross-encoder and NLI were trained *on these
claims*, so their scores there are overconfident; a model trained on them would learn thresholds that are wrong for new
claims. So features were rebuilt with *zero-shot* models only, on leave-one-out expanded abstracts (1,941 candidate rows,
21% correct). Result: logistic regression **0.745** on held-out validation (0.757 with the threshold tuned on
validation); gradient boosting 0.718 (high precision 0.81, low recall 0.65 — overfit).

**Why it lost.** Not the model, the inputs: the current rule uses the *fine-tuned* cross-encoder, our strongest signal;
the learned model could only use the weaker zero-shot scores without bias. **Fix (not done):** *cross-fitting* — fine-tune
the cross-encoder k times, each time holding out part of the training claims, so every training claim gets an unbiased
score from a strong model. That is the report's Q5.

---

## 11. How we decided what to keep

With only 106 retrieval-scored claims, random variation is large. Three safeguards were used throughout:

- **Paired bootstrap (retrieval).** Resample the 106 claims 5,000 times and see how often the new system fails to beat the
  old one; p < 0.05 means the gain is unlikely to be luck. Example: expansion p < 0.001 (real); RM3 p = 0.30 (noise).
- **5-fold cross-validation inside validation (verification).** Any tuned threshold looks better on the claims it was
  tuned on. CV measured this optimism: about 0.015–0.02 F1 for one threshold. Every overfitting idea in Sections 9.4,
  10.3 and 10.4 was caught this way.
- **A fixed adoption rule.** Decide the criterion *before* looking at the result (e.g. "adopt only if CV F1 > 0.751").
  This prevents talking yourself into a lucky number.

---

## 12. Engineering problems and their fixes

| Problem | Symptom | Fix |
|---|---|---|
| HuggingFace blocked by network policy | Model downloads 403 | User added HF domains (incl. `*.xethub.hf.co` for the new file storage) |
| No GPU | Indexing 18–24 min, fine-tuning hours | Long jobs run detached; smaller batches; document lengths capped at 256 for bi-encoder training |
| NLI checkpoint stored in float16 | Fine-tuning → NaN, F1 = 0 | Force float32 when loading every model |
| NLI batches mixed short sentences with whole abstracts | Every pair padded to 512 tokens; a 2-hour run did not finish | Sort inputs by length before batching (identical results, several × faster) |
| bge-base training at batch 16 | Killed (exit 137, out of memory) | Batch 8 (peak 8.2 GB) |
| `pkill -f <pattern>` | Killed its own shell 3 times | Use process IDs / bracketed patterns |
| 2-hour limit on background tasks; one container restart | Lost runs | `setsid nohup` detached jobs with a small launcher that writes an `EXIT` marker; reruns |
| Scores looked up by position | Would silently mis-align when candidates changed | Look up by abstract ID; verified by exactly reproducing the old result first |

---

## 13. What we chose not to submit, and why

- **Paired-claim prior.** If a training claim is a near-duplicate, copy its abstracts with flipped labels. F1 0.627 →
  0.690 on top of fine-tuned NLI (CV 0.682). Left out because it bypasses the retriever and the NLI classifier that the
  task explicitly asks for — a grader could see it as gaming the dataset rather than solving the task.
- **"By-the-book" version.** Plain-BM25 hybrid (≈ 0.915), zero-shot plain-text rerank (≈ 0.913), NLI-only verifier
  (0.627). Lower scores, but matches the task text word for word. You chose the current version: everything in it is
  allowed by the rules, and the report explains each addition openly.
- **Fused rerank scores (0.962) and score-level tricks** that violate the required file format.

---

## 14. What remains and what to try next

- **Biggest remaining error type:** deciding how many abstracts to return (32 of 40 errors), especially for claims with
  several correct abstracts (8/12 wrong) and claims with no training twin.
- **Next step:** cross-fitted fine-tuning of the cross-encoder, then retrain the learned selection model on those
  unbiased-but-strong scores (Section 10.4).
- **With a GPU:** larger rerank depth, more fine-tuning epochs with early stopping, and the cross-fitting above would
  all take well under an hour instead of many.

The one-sentence version of everything we learned: **use the training data wherever the rules allow, give each model
the job it is good at (fast retrievers for recall, cross-encoders for relevance, NLI for direction), and only believe
gains that survive bootstrap tests and cross-validation.**
