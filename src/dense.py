# FILE: src/dense.py
"""
Dense retrieval
- Default: Linq-AI-Research/Linq-Embed-Mistral (Sentence-Transformers)
- L2 normalize -> FAISS Inner Product (cosine)
- GPU FAISS if available, else CPU
- Batching via Sentence-Transformers

CLI:
    python dense.py build \
      --chunks_jsonl "$E2E_CMU_RAG/data/processed/chunks.jsonl" \
      --model Linq-AI-Research/Linq-Embed-Mistral \
      --normalize

    python dense.py query \
      --query "On what day did William S. Dietrich II pass away?" \
      --top_k 5 \
      --normalize
"""
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import torch

try:
    from sentence_transformers import SentenceTransformer  # pip install -U sentence-transformers
except Exception:  # pragma: no cover
    SentenceTransformer = None


# ---------------------------
# FAISS helpers
# ---------------------------
def _gpu_available():
    try:
        return faiss.get_num_gpus() > 0
    except Exception:
        return False

def _to_gpu(index):
    res = faiss.StandardGpuResources()
    try:
        return faiss.index_cpu_to_all_gpus(index)
    except Exception:
        if faiss.get_num_gpus() > 0:
            return faiss.index_cpu_to_gpu(res, 0, index)
        return index


# ---------------------------
# Device helper
# ---------------------------
def _st_device():
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------
# Config
# ---------------------------
class DenseConfig:
    def __init__(
        self,
        model_name: str = "Linq-AI-Research/Linq-Embed-Mistral",
        normalize: bool = True,
        device: str = _st_device(),
        batch_size: int = 16,
        max_length: int = 4096,
        query_instruction: str | None = None,  # optional: instruction prefix for queries
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction


# ---------------------------
# Retriever
# ---------------------------
class DenseRetriever:
    def __init__(self, config: DenseConfig | None = None):
        self.config = config or DenseConfig()
        self.model: SentenceTransformer | None = None
        self.index = None
        self.gpu_index = None
        self.meta: List[Dict[str, Any]] = []

    # ---------- model loading ----------
    def _ensure_model(self):
        if self.model is not None:
            return
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install: pip install -U sentence-transformers"
            )
        # Back-compat aliasing: treat any 'qwen' request as Linq-Embed-Mistral
        name_lower = (self.config.model_name or "").lower()
        if "qwen" in name_lower or "embedding-8b" in name_lower:
            resolved_name = "Linq-AI-Research/Linq-Embed-Mistral"
        else:
            resolved_name = self.config.model_name or "Linq-AI-Research/Linq-Embed-Mistral"

        self.model = SentenceTransformer(resolved_name, device=self.config.device)
        try:
            self.model.max_seq_length = int(self.config.max_length)
        except Exception:
            pass

    # ---------- embedding ----------
    def _encode(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        self._ensure_model()
        kwargs = dict(
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
        if is_query and self.config.query_instruction:
            prompt = f"Instruct: {self.config.query_instruction}\nQuery: "
            X = self.model.encode(texts, prompt=prompt, **kwargs).astype("float32")
        else:
            X = self.model.encode(texts, **kwargs).astype("float32")
        if self.config.normalize:
            faiss.normalize_L2(X)
        return X

    # ---------- build / search ----------
    def build_from_jsonl(self, jsonl_path: str):
        meta, texts = [], []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                t = r.get("text", "")
                if not isinstance(t, str) or not t.strip():
                    continue
                m = {"id": len(meta), "text": t}
                for k, v in r.items():
                    if k == "id":
                        m["orig_id"] = v
                    elif k != "text":
                        m[k] = v
                meta.append(m); texts.append(t)
        if not texts:
            raise ValueError("No valid chunks in JSONL. Need at least a 'text' field per line.")

        X = self._encode(texts, is_query=False)
        d = X.shape[1]
        self.index = faiss.IndexFlatIP(d) if self.config.normalize else faiss.IndexFlatL2(d)
        self.index.add(X)
        self.meta = meta

    def _ensure_gpu_index(self):
        if self.gpu_index is None and _gpu_available() and self.index is not None:
            self.gpu_index = _to_gpu(self.index)

    def get_doc(self, doc_id: int) -> str:
        return self.meta[doc_id]["text"]

    def get_meta(self, doc_id: int) -> Dict[str, Any]:
        return self.meta[doc_id]

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        Q = self._encode([query], is_query=True)
        self._ensure_gpu_index()
        idx = self.gpu_index if self.gpu_index is not None else self.index
        D, I = idx.search(Q, k)
        return [(int(i), float(s)) for i, s in zip(I[0], D[0])]

    def fit(self, texts: List[str], metas: List[Dict[str, Any]] | None = None):
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be List[str].")

        if metas is None:
            self.meta = [{"id": i, "text": t} for i, t in enumerate(texts) if t.strip()]
        else:
            if len(metas) != len(texts):
                raise ValueError("metas must match texts length.")
            self.meta = metas

        texts_clean = [m["text"] for m in self.meta if isinstance(m.get("text"), str)]
        if not texts_clean:
            raise ValueError("No non-empty chunks to index.")

        X = self._encode(texts_clean, is_query=False)
        d = X.shape[1]
        self.index = faiss.IndexFlatIP(d) if self.config.normalize else faiss.IndexFlatL2(d)
        self.index.add(X)
        return self

    # ---------- save / load ----------
    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "dense_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.config.model_name,
                "normalize": self.config.normalize,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "query_instruction": self.config.query_instruction,
            }, f)

    @classmethod
    def load(cls, out_dir: str):
        with open(os.path.join(out_dir, "dense_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self = cls(DenseConfig(
            model_name=cfg.get("model_name", "Linq-AI-Research/Linq-Embed-Mistral"),
            normalize=cfg.get("normalize", True),
            device=_st_device(),
            batch_size=cfg.get("batch_size", 16),
            max_length=cfg.get("max_length", 4096),
            query_instruction=cfg.get("query_instruction"),
        ))
        self.index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
        meta = []
        with open(os.path.join(out_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        self.meta = meta
        return self


# ---------------------------
# CLI helpers
# ---------------------------
def safe_get_env(key):
    v = os.getenv(key)
    if not v:
        raise ValueError(f"Env variable {key} was not found.")
    return v

def build_cmd(args):
    repo_root = safe_get_env("E2E_CMU_RAG")
    out_dir = args.out_dir or os.path.join(repo_root, "outputs", "dense")
    cfg = DenseConfig(
        model_name=args.model,
        normalize=args.normalize,
        device=_st_device(),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    retr = DenseRetriever(cfg)
    retr.build_from_jsonl(args.chunks_jsonl)
    retr.save(out_dir)
    print(out_dir)

def build_from_texts(
    texts: List[str],
    out_dir: str,
    model: str = "Linq-AI-Research/Linq-Embed-Mistral",
    normalize: bool = True,
    batch_size: int = 16,
    max_length: int = 4096,
    query_instruction: str | None = None,
):
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a List[str].")

    cfg = DenseConfig(
        model_name=model,
        normalize=normalize,
        device=_st_device(),
        batch_size=batch_size,
        max_length=max_length,
        query_instruction=query_instruction,
    )
    retr = DenseRetriever(cfg)

    retr.meta = [{"id": i, "text": t} for i, t in enumerate(texts) if t.strip()]
    texts_clean = [m["text"] for m in retr.meta]
    if not texts_clean:
        raise ValueError("No non-empty chunks to index.")

    X = retr._encode(texts_clean, is_query=False)
    d = X.shape[1]
    retr.index = faiss.IndexFlatIP(d) if retr.config.normalize else faiss.IndexFlatL2(d)
    retr.index.add(X)

    os.makedirs(out_dir, exist_ok=True)
    retr.save(out_dir)
    return out_dir


def query_cmd(args):
    repo_root = safe_get_env("E2E_CMU_RAG")
    index_dir = args.index_dir or os.path.join(repo_root, "outputs", "dense")
    retr = DenseRetriever.load(index_dir)
    # allow overriding at query time
    retr.config.model_name = args.model
    retr.config.normalize = args.normalize
    retr.config.device = _st_device()
    retr.config.batch_size = args.batch_size
    retr.config.max_length = args.max_length

    if args.query:
        qs = [args.query]
    else:
        with open(args.queries_path, "r", encoding="utf-8") as f:
            qs = [ln.strip() for ln in f if ln.strip()]

    for i, q in enumerate(qs, 1):
        hits = retr.search(q, args.top_k)
        print("=" * 80)
        print(f"Q{i}: {q}")
        for r, (doc_id, score) in enumerate(hits, 1):
            m = retr.meta[doc_id]
            doc = m.get("doc_path", "<unknown>")
            pv = m.get("text", "").replace("\n", " ")
            if len(pv) > 200:
                pv = pv[:200] + "..."
            print(f"[{r}] {score:.4f} {doc} id={m.get('id')} orig_id={m.get('orig_id')}")
            print(f"    {pv}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--chunks_jsonl", required=True)
    b.add_argument("--out_dir", default=None)
    b.add_argument("--model", default="Linq-AI-Research/Linq-Embed-Mistral")
    b.add_argument("--normalize", action="store_true")
    b.add_argument("--batch_size", type=int, default=16)
    b.add_argument("--max_length", type=int, default=4096)
    b.set_defaults(func=build_cmd)

    q = sub.add_parser("query")
    q.add_argument("--query")
    q.add_argument("--queries_path")
    q.add_argument("--top_k", type=int, default=5)
    q.add_argument("--index_dir", default=None)
    q.add_argument("--model", default="Linq-AI-Research/Linq-Embed-Mistral")
    q.add_argument("--normalize", action="store_true")
    q.add_argument("--batch_size", type=int, default=16)
    q.add_argument("--max_length", type=int, default=4096)
    q.set_defaults(func=query_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
