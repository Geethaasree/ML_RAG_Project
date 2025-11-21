from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import RAGService, PredictionService
import os

app = FastAPI(title="ml-rag-platform")

# instantiate services (singleton)
rag_service = RAGService()
pred_service = PredictionService()

class QueryPayload(BaseModel):
    query: str
    top_k: int = 5

class PredictPayload(BaseModel):
    features: dict

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/rag/query")
async def rag_query(payload: QueryPayload):
    return rag_service.answer(payload.query, top_k=payload.top_k)

@app.post("/predict/churn")
async def predict_churn(payload: PredictPayload):
    return pred_service.predict_churn(payload.features)

@app.post("/predict/fraud")
async def predict_fraud(payload: PredictPayload):
    return pred_service.predict_fraud(payload.features)
