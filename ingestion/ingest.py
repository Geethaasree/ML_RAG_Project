# ingestion/ingest.py
import os
import sys
import pickle
from rag.embeddings import embed_texts
from rag.retriever import FAISSRetriever
import numpy as np

# assume you run this from repo root: python ingestion/ingest.py path/to/text_folder
TEXT_DIR = sys.argv[1] if len(sys.argv) > 1 else "docs"
# collect text files
texts = []
for root, _, files in os.walk(TEXT_DIR):
    for f in files:
        if f.lower().endswith((".txt", ".md")):
            with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                texts.append(fh.read())

if not texts:
    print("No text files found in", TEXT_DIR)
    sys.exit(1)

batch_size = 64
embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    emb = embed_texts(batch)   # uses OpenAI embeddings
    embeddings.extend(emb)

emb_np = np.array(embeddings).astype(np.float32)
retriever = FAISSRetriever()
retriever.add(emb_np, texts)
print("Indexed", len(texts), "documents.")
