# FAISS-based retriever for local dev
import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/faiss_index.pkl")
DOCS_PATH = os.getenv("FAISS_DOCS_PATH", "/app/docs.pkl")

class FAISSRetriever:
    def __init__(self):
        self.index = None
        self.docs = []
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            self._load()

    def _load(self):
        with open(DOCS_PATH, "rb") as f:
            self.docs = pickle.load(f)
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.docs) > 0

    def add(self, embeddings: np.ndarray, docs: List[str]):
        """
        Add embeddings (nxd) and docs list to the index and persist.
        For simplicity we rebuild index from scratch.
        """
        # store docs
        self.docs.extend(docs)
        # build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        self.index = index
        # persist
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.docs, f)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        D, I = self.index.search(query_embedding.astype(np.float32), top_k)
        # return list of (doc_index, distance)
        return list(zip(I[0].tolist(), D[0].tolist()))
