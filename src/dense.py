# FILE: src/dense.py
"""
Dense retrieval
- Default: Qwen/Qwen3-Embedding-8B via Hugging Face Transformers
- Mean pooling + L2 normalization -> FAISS Inner Product (cosine)
- Uses FAISS on GPU if available, else CPU
- Batching + bf16 on GPU for speed/VRAM
- Backward-compatible CLI:
    python dense.py build \
      --chunks_jsonl "$E2E_CMU_RAG/data/processed/chunks.jsonl" \
      --model Qwen/Qwen3-Embedding-8B \
      --normalize

    python dense.py query \
      --query "On what day did William S. Dietrich II pass away?" \
      --top_k 5 \
      --normalize
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Tuple

# Optional fallback to Sentence-Transformers if user requests an ST model
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:  # pragma: no cover
    SentenceTransformer = None

from transformers import AutoModel, AutoTokenizer


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
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        normalize: bool = True,
        device: str = _st_device(),
        batch_size: int = 16,
        max_length: int = 4096,    # plenty for most web chunks; raise if you truly need longer
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length


# ---------------------------
# Retriever
# ---------------------------
class DenseRetriever:
    def __init__(self, config: DenseConfig | None = None):
        self.config = config or DenseConfig()
        self.model = None            # HF model or ST model
        self.tokenizer = None        # HF tokenizer (for Qwen)
        self.backend = None          # "qwen" or "st"
        self.index = None
        self.gpu_index = None
        self.meta: List[Dict[str, Any]] = []

    # ---------- model loading ----------
    def _ensure_model(self):
        if self.model is not None:
            return
        name = (self.config.model_name or "").lower()

        # Heuristic: use HF path when model name suggests Qwen embedding
        if "qwen3-embedding" in name or ("qwen" in name and "embedding" in name):
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            # device_map="auto" places on GPU if available
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            self.backend = "qwen"
        else:
            # Fall back to Sentence-Transformers
            if SentenceTransformer is None:
                raise ImportError(
                    f"Requested non-Qwen model '{self.config.model_name}', "
                    "but sentence-transformers is not installed."
                )
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
            self.backend = "st"

    # ---------- embedding implementations ----------
    @torch.no_grad()
    def _embed_qwen(self, texts: List[str]) -> np.ndarray:
        bs = max(1, int(self.config.batch_size))
        vecs: List[torch.Tensor] = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            # move to model device
            dev = self.model.device
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            # Forward
            out = self.model(**inputs)

            # Mean pooling over valid tokens
            last = out.last_hidden_state            # [B, T, D]
            mask = inputs["attention_mask"].unsqueeze(-1).to(last.dtype)  # [B, T, 1]
            summed = (last * mask).sum(dim=1)       # [B, D]
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts                # [B, D]
            vecs.append(pooled.detach().cpu())

        X = torch.cat(vecs, dim=0).to(torch.float32).numpy()
        if self.config.normalize:
            faiss.normalize_L2(X)
        return X

    def _embed_st(self, texts: List[str]) -> np.ndarray:
        # Sentence-Transformers handles batching internally
        X = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False
        ).astype("float32")
        if self.config.normalize:
            faiss.normalize_L2(X)
        return X

    def _embed(self, texts: List[str]) -> np.ndarray:
        self._ensure_model()
        if self.backend == "qwen":
            return self._embed_qwen(texts)
        return self._embed_st(texts)

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

        X = self._embed(texts)
        d = X.shape[1]
        # For normalized vectors, IP == cosine similarity
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
        self._ensure_model()
        if self.backend == "qwen":
            Q = self._embed_qwen([query])
        else:
            Q = self._embed_st([query])

        self._ensure_gpu_index()
        idx = self.gpu_index if self.gpu_index is not None else self.index
        D, I = idx.search(Q, k)
        return [(int(i), float(s)) for i, s in zip(I[0], D[0])]

    def fit(self, texts: List[str], metas: List[Dict[str, Any]] | None = None):
        """
        Build the FAISS index in-memory from a list[str] of chunk texts.
        Returns self so you can chain calls. Does not save to disk.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be List[str].")

        # build meta (or use provided)
        if metas is None:
            self.meta = [{"id": i, "text": t} for i, t in enumerate(texts) if t.strip()]
        else:
            if len(metas) != len(texts):
                raise ValueError("metas must match texts length.")
            self.meta = metas

        texts_clean = [m["text"] for m in self.meta if isinstance(m.get("text"), str)]
        if not texts_clean:
            raise ValueError("No non-empty chunks to index.")

        X = self._embed(texts_clean)
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
            }, f)

    @classmethod
    def load(cls, out_dir: str):
        with open(os.path.join(out_dir, "dense_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self = cls(DenseConfig(
            model_name=cfg.get("model_name", "Qwen/Qwen3-Embedding-8B"),
            normalize=cfg.get("normalize", True),
            device=_st_device(),
            batch_size=cfg.get("batch_size", 16),
            max_length=cfg.get("max_length", 4096),
        ))
        self.index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
        meta = []
        with open(os.path.join(out_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        self.meta = meta
        return self


# ---------------------------
# Helpers used by your CLI
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
    model: str = "Qwen/Qwen3-Embedding-8B",
    normalize: bool = True,
    batch_size: int = 16,
    max_length: int = 4096,
):
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a List[str].")

    cfg = DenseConfig(
        model_name=model,
        normalize=normalize,
        device=_st_device(),
        batch_size=batch_size,
        max_length=max_length,
    )
    retr = DenseRetriever(cfg)

    # build simple meta
    retr.meta = [{"id": i, "text": t} for i, t in enumerate(texts) if t.strip()]
    texts_clean = [m["text"] for m in retr.meta]
    if not texts_clean:
        raise ValueError("No non-empty chunks to index.")

    X = retr._embed(texts_clean)
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
    # allow overriding model/normalize at query time (useful for trying other encoders)
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
    b.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    b.add_argument("--normalize", action="store_true")
    b.add_argument("--batch_size", type=int, default=16)
    b.add_argument("--max_length", type=int, default=4096)
    b.set_defaults(func=build_cmd)

    q = sub.add_parser("query")
    q.add_argument("--query")
    q.add_argument("--queries_path")
    q.add_argument("--top_k", type=int, default=5)
    q.add_argument("--index_dir", default=None)
    q.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    q.add_argument("--normalize", action="store_true")
    q.add_argument("--batch_size", type=int, default=16)
    q.add_argument("--max_length", type=int, default=4096)
    q.set_defaults(func=query_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
