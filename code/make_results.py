"""Fill results.json from the stats each task script saved in cache/. Run after Tasks 1-5."""
import json
import os

from common import CACHE, ROOT

C = lambda f: json.load(open(os.path.join(CACHE, f)))
r = json.load(open(os.path.join(ROOT, "results_template.json")))
old = json.load(open(os.path.join(ROOT, "results.json"))) if os.path.exists(os.path.join(ROOT, "results.json")) else {}
t1, t2, t3, t4 = C("t1_bm25.json"), C("t2_stats.json"), C("t3_stats.json"), C("t4_stats.json")
t5 = C(os.environ.get("T5_STATS", "t5_stats_nli.json"))
b = t1["best"]
r.update(name=old.get("name", "YOUR_NAME"), roll_no=old.get("roll_no", "ROLLNO"), seed=42,
         gpu_used="none - CPU only (4 vCPU cloud container); all latencies are CPU numbers")
r["task1"] = dict(bm25_settings=f"Okapi BM25 (own scipy implementation, Lucene IDF), k1={b['k1']}, b={b['b']}, "
                                f"title+abstract, lowercase, sklearn English stopwords removed, Snowball stemming; "
                                f"grid over k1 x b x stemming x stopwords x tokenizer x title on val", val_ndcg10=round(b["ndcg10"], 4))
g, s = t2["general"], t2["scientific"]
r["task2"] = dict(general_model=g["model"], scientific_model="ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder",
                  val_ndcg10_general=round(g["val"]["ndcg10"], 4), val_ndcg10_scientific=round(s["val"]["ndcg10"], 4),
                  index_time_sec={"general": round(g["index_time_sec"], 1), "scientific": round(s["index_time_sec"], 1)},
                  query_latency_ms={"general": round(g["query_latency_ms"], 1), "scientific": round(s["query_latency_ms"], 1)},
                  memory_mb={"general": round(g["memory_mb"]), "scientific": round(s["memory_mb"])})
h = t3["hybrid"]["cfg"]
names = {"bm25x": "BM25 over abstracts expanded with training claims", "bm25": "BM25", "general": "bge-base", "scientific": "MedCPT"}
r["task3"] = dict(hybrid_method=f"{h['method']}-normalised score interpolation (per-claim top-100), weights "
                                + ", ".join(f"{names[c]}={w}" for c, w in zip(h["comps"], h["w"])),
                  reranker_model=t3["reranker"], rerank_top_k=t3["k"], val_ndcg10_hybrid=round(t3["hybrid"]["val"]["ndcg10"], 4),
                  val_ndcg10_rerank=round(t3["rerank"][str(t3["k"])]["ndcg10"], 4), rerank_query_latency_ms=round(t3["rerank_latency_ms"], 1))
r["task4"] = dict(base_model=t4["base"], val_ndcg10_random_negatives=round(t4["random"]["best_val_ndcg10"], 4),
                  val_ndcg10_hard_negatives=round(t4["hard"]["best_val_ndcg10"], 4))
if t5.get("scorer") == "ce-gated":
    r["task5"] = dict(retriever_used="t3_hybrid top-3, evidence selected by the t3_rerank cross-encoder (ncbi/MedCPT-Cross-Encoder)",
                      nli_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli: mean of zero-shot and fine-tuned (2 epochs on "
                                "training-claim rationales) probabilities; label only, the cross-encoder decides which abstracts are evidence",
                      zero_shot_or_finetuned="finetuned (ensembled with zero-shot)", val_f1=round(t5["val"]["f1"], 4))
else:
    r["task5"] = dict(retriever_used="t3_hybrid (best val nDCG@10)",
                      nli_model=t5["model"] + (", fine-tuned 2 epochs on training-claim rationales (+ mined neutral sentences)" if t5["scorer"] == "nli-ft" else ""),
                      zero_shot_or_finetuned="finetuned" if t5["scorer"] == "nli-ft" else "zero-shot",
                      val_f1=round(t5["val"]["f1"], 4))
r["report_q1"] = {"bm25_wins": [673, 1261], "general_wins": [766, 1311]}
json.dump(r, open(os.path.join(ROOT, "results.json"), "w"), indent=2)
print(json.dumps(r, indent=2))
