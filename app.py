# app.py
import time
from pathlib import Path
import threading

import gradio as gr

DATA_OUT = Path("data_out")
ASSESS_CSV = DATA_OUT / "assessments.csv"
FAISS_INDEX = DATA_OUT / "assess_index.faiss"
EMB_FILE = DATA_OUT / "assess_embeddings.npy"

# Start index build in background if missing
def build_index_background():
    if ASSESS_CSV.exists() and FAISS_INDEX.exists() and EMB_FILE.exists():
        print("data_out exists, skipping build.")
        return
    print("data_out missing. Running build_index.py (may take several minutes)...")
    import subprocess, sys
    proc = subprocess.run([sys.executable, "build_index.py"])
    print("build_index.py returned:", proc.returncode)

# Kick off a background build thread so the web server can bind quickly
threading.Thread(target=build_index_background, daemon=True).start()
time.sleep(1.0)

# Import recommend (will fail until index ready, but import should succeed)
try:
    from recommend import recommend
except Exception as e:
    print("Warning: recommend import failed at startup (index-building may be in progress):", e)
    # fallback dummy function
    def recommend(q, top_k=6):
        return [{"name": "Index is building — try again in a minute", "url": "", "score": 0.0}]

def run_recommend(q):
    if not q or not q.strip():
        return "Please enter a job description."
    try:
        recs = recommend(q, top_k=6)
        # format into lines
        lines = []
        for r in recs:
            nm = r.get("name") or r.get("assessment_name") or ""
            url = r.get("url","")
            score = r.get("score", None)
            if score is None:
                lines.append(f"{nm} — {url}")
            else:
                lines.append(f"{nm} — {url}  (score: {score:.3f})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error running recommend(): {e}"

css = """
body { background-color: #f8fafc; }
.gradio-container { max-width: 900px; margin: auto; }
"""

with gr.Blocks(css=css, title="SHL Recommender") as demo:
    gr.Markdown("## 🔍 SHL Assessment Recommender")
    txt = gr.Textbox(lines=6, placeholder="Enter job description or hiring query...", label="Query")
    btn = gr.Button("Get Recommendations")
    out = gr.Textbox(label="Recommendations", interactive=False)
    btn.click(fn=run_recommend, inputs=txt, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
