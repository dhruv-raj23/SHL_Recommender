import pandas as pd
from recommend import recommend
import numpy as np

DATA_PATH = "data/Gen_AI_Dataset.xlsx"
train_df = pd.read_excel(DATA_PATH, sheet_name="Train-Set")
train_df.columns = train_df.columns.str.strip().str.lower()

query_col = "query"
url_col = [c for c in train_df.columns if "url" in c][0]

k = 10
total = len(train_df)
hits = 0
ranks = []

for i, row in train_df.iterrows():
    q = str(row[query_col])
    target_url = str(row[url_col]).strip().lower()
    recs = recommend(q, top_k=k)
    pred_urls = [r["url"].strip().lower() for r in recs]

    if target_url in pred_urls:
        hits += 1
        ranks.append(pred_urls.index(target_url) + 1)
    else:
        ranks.append(k + 1)

recall_at_k = hits / total
mrr = np.mean([1 / r if r <= k else 0 for r in ranks])

print(f" Recall@{k}: {recall_at_k:.3f}")
print(f" MRR@{k}: {mrr:.3f}")
