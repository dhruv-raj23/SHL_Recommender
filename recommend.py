
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

OUT_DIR = "data_out"

print(" Loading index and metadata...")
assess_df = pd.read_csv(f"{OUT_DIR}/assessments.csv")
embeddings = np.load(f"{OUT_DIR}/assess_embeddings.npy")
index = faiss.read_index(f"{OUT_DIR}/assess_index.faiss")

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# reranker = CrossEncoder("sentence-transformers/msmarco-roberta-base-v3")
# reranker = CrossEncoder("cross-encoder/ms-marco-electra-base")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def recommend(query: str, top_k: int = 10, retrieve_k: int = 100):
    """Return top-k assessment recommendations for a query."""
    query = query.lower().strip()
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, retrieve_k)

    candidates, docs = [], []
    for cid, score in zip(I[0], D[0]):
        row = assess_df.iloc[cid]
        name, desc, url = str(row['name']).lower(), str(row['description']).lower(), str(row['url'])
        candidates.append({
            "id": int(cid),
            "name": name,
            "url": url,
            "embed_score": float(score)
        })
        docs.append(f"{name}. {desc}")

    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    min_s, max_s = min(scores), max(scores)
    scores = [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]

    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
        candidates[i]["score"] = 0.3 * candidates[i]["embed_score"] + 0.7 * candidates[i]["rerank_score"]

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

if __name__ == "__main__":
    q = "Java developer with good communication skills"
    results = recommend(q, top_k=6)
    print("\n Recommendations for:", q)
    for r in results:
        print(f"→ {r['name']} | {r['url']}")
