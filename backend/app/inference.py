import os
from typing import List, Dict
import numpy as np
import joblib
import boto3
from .rag.embeddings import embed_texts
from .rag.retriever import FAISSRetriever
import json

# PredictionService (loads local sklearn models)
class PredictionService:
    def __init__(self):
        # models in backend/app/models/ e.g. churn.pkl, fraud.pkl
        churn_path = os.getenv("MODEL_CHURN_PATH", "/app/models/churn.pkl")
        fraud_path = os.getenv("MODEL_FRAUD_PATH", "/app/models/fraud.pkl")
        self.churn_model = joblib.load(churn_path) if os.path.exists(churn_path) else None
        self.fraud_model = joblib.load(fraud_path) if os.path.exists(fraud_path) else None

    def _prep(self, features: dict, feature_order: List[str]):
        return [features.get(k, 0) for k in feature_order]

    def predict_churn(self, features: dict):
        if not self.churn_model:
            return {"error": "churn model not found"}
        feature_order = getattr(self.churn_model, "feature_order", None) or list(features.keys())
        X = self._prep(features, feature_order)
        proba = self.churn_model.predict_proba([X])[0,1]
        return {"churn_probability": float(proba)}

    def predict_fraud(self, features: dict):
        if not self.fraud_model:
            return {"error": "fraud model not found"}
        feature_order = getattr(self.fraud_model, "feature_order", None) or list(features.keys())
        X = self._prep(features, feature_order)
        proba = self.fraud_model.predict_proba([X])[0,1]
        return {"fraud_probability": float(proba)}

# RAG service
class RAGService:
    def __init__(self):
        self.retriever = FAISSRetriever()

    def answer(self, query: str, top_k: int = 5) -> Dict:
        if not self.retriever.is_ready():
            return {"error":"retriever not ready, run ingestion to index docs"}
        q_emb = np.array(embed_texts([query])).astype(np.float32)
        results = self.retriever.retrieve(q_emb, top_k=top_k)
        docs = []
        for idx, dist in results:
            try:
                docs.append(self.retriever.docs[idx])
            except Exception:
                continue
        context = "\n\n".join(docs)
        # build prompt
        prompt = f"Use the context to answer. Context:\n{context}\n\nQuestion:\n{query}"
        # call LLM (OpenAI)
        from openai import ChatCompletion
        import openai
        if not openai.api_key:
            return {"error":"OpenAI API key not set"}
        resp = openai.ChatCompletion.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                                            messages=[{"role":"user","content":prompt}])
        answer = resp["choices"][0]["message"]["content"]
        return {"query": query, "answer": answer, "context": docs}
