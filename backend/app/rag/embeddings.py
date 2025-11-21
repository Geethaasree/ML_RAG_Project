# embedding abstraction: using OpenAI embeddings by default
import os
from typing import List
import openai

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Small example using OpenAI text-embedding-3-small
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set for embeddings")
    response = openai.Embeddings.create(input=texts, model="text-embedding-3-small")
    return [item["embedding"] for item in response["data"]]
