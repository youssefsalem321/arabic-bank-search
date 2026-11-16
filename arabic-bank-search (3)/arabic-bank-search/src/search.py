import numpy as np

from .config import (FUSION_K, K_FINAL, K_RETRIEVE, MAX_ALIASES, RERANK_USE_ANSWER,
                     RERANKER_MODEL, RETRIEVER_MODEL, USE_ALIASES)
from .normalize import normalize, tokenize


def texts_of(faq):
    """The FAQ question plus its alternative phrasings (if any)."""
    extra = faq.get("aliases", [])[:MAX_ALIASES] if USE_ALIASES else []
    return [faq["question"]] + list(extra)


class BM25Retriever:
    """Keyword baseline."""

    def __init__(self, faqs):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(" ".join(texts_of(f)) + " " + f["answer"]) for f in faqs])

    def search(self, query, k=K_RETRIEVE):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]


class DenseRetriever:
    """Bi-encoder: embed FAQ questions once, then cosine similarity."""

    def __init__(self, faqs, model_name=RETRIEVER_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()
        self.n = len(faqs)
        docs, owner = [], []
        for i, f in enumerate(faqs):
            for t in texts_of(f):
                docs.append(self._p(normalize(t)))
                owner.append(i)
        self.owner = np.array(owner)
        self.emb = self.model.encode(docs, normalize_embeddings=True, show_progress_bar=False)

    def _p(self, t):
        return f"passage: {t}" if self.is_e5 else t

    def _q(self, t):
        return f"query: {t}" if self.is_e5 else t

    def search(self, query, k=K_RETRIEVE):
        q = self.model.encode([self._q(normalize(query))], normalize_embeddings=True)[0]
        sims = self.emb @ q
        scores = np.full(self.n, -1.0)
        np.maximum.at(scores, self.owner, sims)   # best-matching phrasing per FAQ
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]


def rrf_fuse(hits, rerank_scores, k=K_FINAL, K=FUSION_K):
    """Reciprocal Rank Fusion of retriever order and reranker order.
    Robust when the reranker is noisy: it can only overrule the retriever when it is clearly better.
    Ties go to the retriever."""
    dense_rank = {idx: r for r, (idx, _) in enumerate(hits)}
    order = np.argsort(-np.asarray(rerank_scores))
    rr_rank = {hits[j][0]: r for r, j in enumerate(order)}
    fused = {i: 1 / (K + dense_rank[i] + 1) + 1 / (K + rr_rank[i] + 1) for i in dense_rank}
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


class Reranker:
    """Cross-encoder: scores (query, candidate) jointly."""

    def __init__(self, model_name=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)

    def score(self, query, faqs, hits, use_answer=RERANK_USE_ANSWER):
        """One score per candidate FAQ = best score over its question and aliases."""
        pairs, owner = [], []
        for h, (i, _) in enumerate(hits):
            f = faqs[i]
            for t in texts_of(f):
                pairs.append((query, t + ("\n" + f["answer"] if use_answer else "")))
                owner.append(h)
        raw = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=float)
        out = np.full(len(hits), -np.inf)
        np.maximum.at(out, np.array(owner), raw)
        return out

    def rerank(self, query, faqs, hits, k=K_FINAL, use_answer=RERANK_USE_ANSWER):
        scores = self.score(query, faqs, hits, use_answer)
        order = np.argsort(-scores)[:k]
        return [(hits[j][0], float(scores[j])) for j in order]


class SearchPipeline:
    """Retrieve (fast) -> Rerank (precise) -> best answer = reranker top-1.
    (RRF fusion was tested in evaluate.py and scored lower on our test set, so it is not used here.)"""

    def __init__(self, faqs, retriever_model=RETRIEVER_MODEL, reranker_model=RERANKER_MODEL):
        self.faqs = faqs
        self.dense = DenseRetriever(faqs, retriever_model)
        self.reranker = Reranker(reranker_model)

    def pack(self, hits):
        return [{**self.faqs[i], "score": s} for i, s in hits]

    def search(self, query, k_retrieve=K_RETRIEVE, k_final=K_FINAL):
        hits = self.dense.search(query, k_retrieve)
        scores = self.reranker.score(query, self.faqs, hits)
        order = np.argsort(-scores)
        reranked = [(hits[j][0], float(scores[j])) for j in order[:k_final]]
        agree = hits[0][0] == reranked[0][0]
        return {
            "retrieved": self.pack(hits[:k_final]),
            "reranked": self.pack(reranked),
            "confidence": "high" if agree else "medium",
            "top_similarity": hits[0][1],
        }
