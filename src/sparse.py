# sparse.py
# A lightweight sparse retriever supporting BM25 (rank-bm25).


from __future__ import annotations

import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# Optional imports guarded to keep this file importable even if a backend is missing.
try:
    from rank_bm25 import BM25Okapi  # pip install rank-bm25
except Exception:  # pragma: no cover
    BM25Okapi = None



# -----------------------------
# Basic text preprocessing
# -----------------------------

def basic_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercase, remove punctuation, whitespace split.
    - Keep alphanumerics and underscores.
    - For multilingual or domain-specific cases, replace with your own tokenizer.
    """
    text = text.lower()
    # Replace non-word characters with space (keeps letters, digits, underscore)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


def join_tokens(tokens: List[str]) -> str:
    """Join tokens with single spaces for vectorizers that expect string input."""
    return " ".join(tokens)


# -----------------------------
# Configuration dataclass
# -----------------------------

@dataclass
class SparseConfig:
    """
    Configuration for sparse retriever.
    method: 'bm25' or 'tfidf'
    """
    method: str = "bm25"

    # BM25 hyperparameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Common
    lowercase: bool = True  # handled by our tokenizer; keep for extension
    keep_empty_chunks: bool = False  # if False, drop empty after preprocessing


# -----------------------------
# SparseRetriever class
# -----------------------------

class SparseRetriever:
    """
    A unified interface for BM25 and TF-IDF sparse retrieval.
    - Build index from a list of document chunks (strings).
    - Query with a text string and get top-k (doc_id, score) tuples.
    - Save/load index for reuse.

    Notes:
    - BM25 stores tokenized corpus internally (list[list[str]]).
    """

    def __init__(self, config: Optional[SparseConfig] = None):
        self.config = config or SparseConfig()
        self.method = self.config.method.lower()

        # Raw corpus and tokenized corpus
        self.doc_chunks: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

        # Backends
        self._bm25 = None

        # ID ↔ meta mapping (optional, but useful if you want to store extra info per chunk)
        self.doc_meta: List[Dict[str, Any]] = []

    # --------- Build ---------

    def build(self, doc_chunks: List[str], doc_meta: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build the sparse index from a list of document chunks (strings).
        doc_meta is optional, same length as doc_chunks.
        """
        if not isinstance(doc_chunks, list) or not all(isinstance(x, str) for x in doc_chunks):
            raise ValueError("doc_chunks must be List[str].")

        # Preprocess and keep a filtered view if requested
        toks = [basic_tokenize(x) for x in doc_chunks]
        if not self.config.keep_empty_chunks:
            keep_mask = [len(t) > 0 for t in toks]
            self.doc_chunks = [c for c, k in zip(doc_chunks, keep_mask) if k]
            self.tokenized_corpus = [t for t, k in zip(toks, keep_mask) if k]
            if doc_meta is not None:
                self.doc_meta = [m for m, k in zip(doc_meta, keep_mask) if k]
            else:
                self.doc_meta = [{} for _ in range(len(self.doc_chunks))]
        else:
            self.doc_chunks = doc_chunks
            self.tokenized_corpus = toks
            self.doc_meta = doc_meta or [{} for _ in range(len(self.doc_chunks))]

        if len(self.doc_chunks) == 0:
            raise ValueError("Empty corpus after preprocessing.")

        
        self._build_bm25()
        

    def _build_bm25(self) -> None:
        """Initialize BM25Okapi with tokenized corpus."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not installed. `pip install rank-bm25`")
        # BM25Okapi does not expose k1/b directly per-instance in API,
        # but we can monkey-patch attributes if needed. The default impl uses k1=1.5, b=0.75.
        # For clarity and reproducibility, we wrap Okapi as-is.
        self._bm25 = BM25Okapi(self.tokenized_corpus, k1=self.config.bm25_k1, b=self.config.bm25_b)


    # --------- Query ---------

    def search(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """
        Search top-k documents for a given query string.
        Returns: list of (doc_id, score) sorted by descending score.
        """
        if not query or not isinstance(query, str):
            return []
        return self._search_bm25(query, k)
    

    def _search_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_tokens = basic_tokenize(query)
        scores = self._bm25.get_scores(q_tokens)  # numpy array length = n_docs
        if scores is None or len(scores) == 0:
            return []

        # Fast top-k: argpartition + sort the slice
        k = min(k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(int(i), float(scores[i])) for i in idx]


    # --------- Utility: accessors ---------

    def get_doc(self, doc_id: int) -> str:
        """Return the original chunk text for a given doc_id."""
        return self.doc_chunks[doc_id]

    def get_meta(self, doc_id: int) -> Dict[str, Any]:
        """Return optional metadata for a given doc_id."""
        return self.doc_meta[doc_id]

    # --------- Save / Load ---------

    def save(self, path: str) -> None:
        """
        Persist the retriever to a directory.
        For BM25: store tokenized corpus + params.
        For TF-IDF: store vectorizer and document-term matrix.
        """
        os.makedirs(path, exist_ok=True)

        # Save config and raw data
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=2)

        with open(os.path.join(path, "doc_chunks.pkl"), "wb") as f:
            pickle.dump(self.doc_chunks, f)

        with open(os.path.join(path, "doc_meta.pkl"), "wb") as f:
            pickle.dump(self.doc_meta, f)

        # Save backend-specific
        
        with open(os.path.join(path, "tokenized_corpus.pkl"), "wb") as f:
            pickle.dump(self.tokenized_corpus, f)
            # BM25Okapi itself is not trivially pickleable across versions; rebuild on load.
            # We save only the tokenized corpus and rebuild BM25 at load time.
        

    @classmethod
    def load(cls, path: str) -> "SparseRetriever":
        """
        Load a previously saved retriever.
        BM25: BM25 index is rebuilt from tokenized_corpus on load (fast).
        TF-IDF: vectorizer + matrix are loaded directly.
        """
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg = SparseConfig(**json.load(f))

        self = cls(cfg)

        with open(os.path.join(path, "doc_chunks.pkl"), "rb") as f:
            self.doc_chunks = pickle.load(f)
        with open(os.path.join(path, "doc_meta.pkl"), "rb") as f:
            self.doc_meta = pickle.load(f)

        
        with open(os.path.join(path, "tokenized_corpus.pkl"), "rb") as f:
            self.tokenized_corpus = pickle.load(f)
            
        self._build_bm25()
        
        return self



