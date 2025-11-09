import pandas as pd
from recommend import recommend

DATA_PATH = "data/Gen_AI_Dataset.xlsx"
sheet_name = "Test-Set"

print(" Reading Test Set...")
test_df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
test_df.columns = test_df.columns.str.strip().str.lower()

query_col = "query"
rows = []

for i, row in test_df.iterrows():
    q = str(row[query_col])
    print(f"[{i+1}/{len(test_df)}] Generating for query: {q[:60]}...")
    recs = recommend(q, top_k=10)
    for rec in recs:
        rows.append({"Query": q, "Assessment_url": rec["url"]})

out = pd.DataFrame(rows)
out.to_csv("submission_predictions.csv", index=False)
print(" submission_predictions.csv generated successfully!")
