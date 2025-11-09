
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import requests
from bs4 import BeautifulSoup
import time

DATA_PATH = "data/Gen_AI_Dataset.xlsx"
OUT_DIR = "data_out"
os.makedirs(OUT_DIR, exist_ok=True)

print(" Loading dataset...")
xl = pd.ExcelFile(DATA_PATH)
train_df = pd.read_excel(DATA_PATH, sheet_name='Train-Set')
train_df.columns = train_df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Columns detected in Train-Set:", list(train_df.columns))

url_col = next((c for c in train_df.columns if "url" in c or "link" in c), None)
if not url_col:
    raise ValueError("❌ Could not find any column containing URLs in the Train-Set sheet.")

urls = train_df[url_col].dropna().unique().tolist()
print(f"Found {len(urls)} unique assessment URLs")

data = []
for i, url in enumerate(urls):
    try:
        print(f"[{i+1}/{len(urls)}] Fetching: {url}")
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            title = soup.find('title').get_text().strip() if soup.find('title') else "No Title"
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            desc = meta_desc['content'].strip() if meta_desc else ''
            paras = ' '.join([p.get_text() for p in soup.find_all('p')][:5])
            full_desc = (desc + ' ' + paras).strip()
            data.append({'name': title, 'url': url, 'description': full_desc})
        else:
            data.append({'name': "Unknown", 'url': url, 'description': ""})
    except Exception as e:
        print("❌ Error fetching:", url, e)
        data.append({'name': "Error", 'url': url, 'description': ""})
    time.sleep(0.2)

assess_df = pd.DataFrame(data)
print(f" Scraped {len(assess_df)} assessments successfully")

print(" Generating embeddings using all-mpnet-base-v2...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

texts = (
    assess_df["name"].fillna('') + ". " +
    assess_df["description"].fillna('') + ". " +
    assess_df["url"].fillna('')
).str.lower().tolist()

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=16)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{OUT_DIR}/assess_index.faiss")
np.save(f"{OUT_DIR}/assess_embeddings.npy", embeddings)
assess_df.to_csv(f"{OUT_DIR}/assessments.csv", index=False)

print(" Done! Saved FAISS index and metadata to", OUT_DIR)

