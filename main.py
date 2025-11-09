from fastapi import FastAPI
from pydantic import BaseModel
from recommend import recommend
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SHL Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryInput(BaseModel):
    query: str
    k: int = 6

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def get_recommendations(req: QueryInput):
    recs = recommend(req.query, top_k=req.k)
    formatted = [
        {
            "assessment_name": r["name"],
            "url": r["url"],
            "score": round(r["score"], 3)
        }
        for r in recs
    ]
    return {"query": req.query, "recommendations": formatted}
