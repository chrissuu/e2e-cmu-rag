def _minmax_norm(pairs):
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {doc_id: 1.0 for doc_id, _ in pairs}
    return {doc_id: (s - lo) / (hi - lo) for doc_id, s in pairs}


def weighted_sum_fuse(sparse_hits, dense_hits, k_final=5, alpha=0.5):
    ns = _minmax_norm(sparse_hits)
    nd = _minmax_norm(dense_hits)
    pool = set(ns) | set(nd)
    fused = []
    for doc_id in pool:
        s = ns.get(doc_id, 0.0)
        d = nd.get(doc_id, 0.0)
        fused.append((doc_id, alpha * d + (1.0 - alpha) * s))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k_final]

class HybridRetriever:
    def __init__(self, sparse_retriever, dense_retriever, alpha=0.5, over_k=50):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.over_k = over_k

    def search(self, query, k=5):
        s_hits = self.sparse.search(query, k=self.over_k)
        d_hits = self.dense.search(query, k=self.over_k)
        return weighted_sum_fuse(s_hits, d_hits, k_final=k, alpha=self.alpha)

    def get_doc(self, doc_id):
        if hasattr(self.sparse, "get_doc"):
            return self.sparse.get_doc(doc_id)
        if hasattr(self.dense, "meta") and 0 <= doc_id < len(self.dense.meta):
            return self.dense.meta[doc_id].get("text", "")
        return ""

    def get_meta(self, doc_id):
        if hasattr(self.sparse, "get_meta"):
            return self.sparse.get_meta(doc_id)
        if hasattr(self.dense, "meta") and 0 <= doc_id < len(self.dense.meta):
            return self.dense.meta[doc_id]
        return {}
