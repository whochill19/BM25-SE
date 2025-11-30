import json
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

# === Load dataset & preprocess for relevance ===
df = pd.read_csv("./dataset/processed_documents.csv")

# Ground truth relevance: dari kolom "Uses"
df['relevance_terms'] = df['Uses'].str.lower().fillna("").str.split()

# === Load logged search queries ===
LOG_FILE = "./search_logs.jsonl"

queries = []
results_list = []

with open(LOG_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        queries.append(entry["query"].lower())
        results_list.append(entry["results"])

print(f"Loaded {len(queries)} queries for evaluation.\n")

# === Evaluation metrics ===
def is_relevant(query_words, doc_words):
    return any(word in doc_words for word in query_words)

def evaluate_query(query, result_indices, k=10):
    query_words = query.split()

    relevance = []
    for idx in result_indices[:k]:
        doc_words = df['relevance_terms'].iloc[idx]
        relevance.append(1 if is_relevant(query_words, doc_words) else 0)

    # Precision & Recall
    precision = sum(relevance) / min(k, len(result_indices))
    total_relevant = sum(df['relevance_terms'].apply(lambda x: is_relevant(query_words, x)))
    recall = sum(relevance) / total_relevant if total_relevant > 0 else 0

    # MRR
    mrr = 0
    for i, rel in enumerate(relevance):
        if rel == 1:
            mrr = 1 / (i + 1)
            break

    # nDCG
    ndcg = ndcg_score([relevance], [relevance]) if any(relevance) else 0

    return precision, recall, mrr, ndcg


# === Evaluate All Queries ===
results = []
for q, r in zip(queries, results_list):
    precision, recall, mrr, ndcg = evaluate_query(q, r)
    results.append([q, precision, recall, mrr, ndcg])

eval_df = pd.DataFrame(results, columns=["Query", "Precision@10", "Recall@10", "MRR", "nDCG@10"])
eval_df.to_csv("./evaluation_results.csv", index=False)
print(eval_df)
print("\nSaved evaluation results → evaluation_results.csv\n")


# === Visualization ===
plt.figure(figsize=(12, 7))
plt.plot(eval_df["Precision@10"], marker='o', label="Precision@10")
plt.plot(eval_df["Recall@10"], marker='s', label="Recall@10")
plt.plot(eval_df["MRR"], marker='x', label="MRR")
plt.plot(eval_df["nDCG@10"], marker='^', label="nDCG@10")
plt.title("Search Engine Evaluation Metrics per Query")
plt.xlabel("Query Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./evaluation_plot.png")

print("📊 Grafik evaluasi tersimpan → evaluation_plot.png")
