"""
dense retrieval
- sentence-transformers 4 embeddings
- FAISS (Ip for cosine w/ normalize)
 - uses GPU

  python dense.py build \
    --chunks_jsonl "$E2E_CMU_RAG/data/processed/chunks.jsonl" \
    --model all-MiniLM-L6-v2 \
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

from sentence_transformers import SentenceTransformer
def _st_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class DenseConfig:
    def __init__(self, model_name="all-MiniLM-L6-v2", normalize=True, device=_st_device()):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device

class DenseRetriever:
    def __init__(self, config=None):
        self.config = config or DenseConfig()
        self.model = None
        self.index = None
        self.gpu_index = None
        self.meta = []

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def _embed(self, texts):
        self._ensure_model()
        X = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=False).astype("float32")
        if self.config.normalize:
            faiss.normalize_L2(X)
        return X

    def build_from_jsonl(self, jsonl_path):
        meta, texts = [], []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                t = r.get("text", "")
                if not isinstance(t, str) or not t.strip(): continue
                m = {"id": len(meta), "text": t}
                for k, v in r.items():
                    if k == "id": m["orig_id"] = v
                    elif k != "text": m[k] = v
                meta.append(m); texts.append(t)
        if not texts:
            raise ValueError("No valid chunks in JSONL. Need at least a 'text' field per line.")
        X = self._embed(texts)
        d = X.shape[1]
        self.index = faiss.IndexFlatIP(d) if self.config.normalize else faiss.IndexFlatL2(d)
        self.index.add(X)
        self.meta = meta

    def _ensure_gpu_index(self):
        if self.gpu_index is None and _gpu_available() and self.index is not None:
            self.gpu_index = _to_gpu(self.index)

    def search(self, query, k):
        self._ensure_model()
        Q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False).astype("float32")
        if self.config.normalize:
            faiss.normalize_L2(Q)
        self._ensure_gpu_index()
        idx = self.gpu_index if self.gpu_index is not None else self.index
        D, I = idx.search(Q, k)
        return [(int(i), float(s)) for i, s in zip(I[0], D[0])]
     
    def fit(self, texts, metas=None):
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


    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "dense_config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": self.config.model_name, "normalize": self.config.normalize}, f)

    @classmethod
    def load(cls, out_dir):
        with open(os.path.join(out_dir, "dense_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self = cls(DenseConfig(model_name=cfg["model_name"], normalize=cfg["normalize"], device=_st_device()))
        self.index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
        meta = []
        with open(os.path.join(out_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        self.meta = meta
        return self

def safe_get_env(key):
    v = os.getenv(key)
    if not v: raise ValueError(f"Env variable {key} was not found.")
    return v

def build_cmd(args):
    repo_root = safe_get_env("E2E_CMU_RAG")
    out_dir = args.out_dir or os.path.join(repo_root, "outputs", "dense")
    cfg = DenseConfig(model_name=args.model, normalize=args.normalize, device=_st_device())
    retr = DenseRetriever(cfg)
    retr.build_from_jsonl(args.chunks_jsonl)
    retr.save(out_dir)
    print(out_dir)

def build_from_texts(
    texts,
    out_dir,
    model: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
):
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a List[str].")

    cfg = DenseConfig(model_name=model, normalize=normalize, device=_st_device())
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
    retr.config.model_name = args.model
    retr.config.normalize = args.normalize
    retr.config.device = _st_device()
    if args.query:
        qs = [args.query]
    else:
        with open(args.queries_path, "r", encoding="utf-8") as f:
            qs = [ln.strip() for ln in f if ln.strip()]
    for i, q in enumerate(qs, 1):
        hits = retr.search(q, args.top_k)
        print("="*80)
        print(f"Q{i}: {q}")
        for r, (doc_id, score) in enumerate(hits, 1):
            m = retr.meta[doc_id]
            doc = m.get("doc_path", "<unknown>")
            pv = m.get("text","").replace("\n"," ")
            if len(pv) > 200: pv = pv[:200] + "..."
            print(f"[{r}] {score:.4f} {doc} id={m.get('id')} orig_id={m.get('orig_id')}")
            print(f"    {pv}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--chunks_jsonl", required=True)
    b.add_argument("--out_dir", default=None)
    b.add_argument("--model", default="all-MiniLM-L6-v2")
    b.add_argument("--normalize", action="store_true")
    b.set_defaults(func=build_cmd)

    q = sub.add_parser("query")
    q.add_argument("--query")
    q.add_argument("--queries_path")
    q.add_argument("--top_k", type=int, default=5)
    q.add_argument("--index_dir", default=None)
    q.add_argument("--model", default="all-MiniLM-L6-v2")
    q.add_argument("--normalize", action="store_true")
    q.set_defaults(func=query_cmd)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
