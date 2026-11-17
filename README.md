# 🏦 Arabic Semantic Search + Reranker for Banking FAQs

Dialect-aware semantic search for banking FAQs. A customer types a question in any Arabic dialect (Egyptian, Gulf, or Modern Standard Arabic), and the system returns the closest FAQ entry, even when the question and the answer share no words.

This is an **encoder-only** project: no text generation, just retrieval and ranking with Transformer encoders.

## Architecture

```
User question
   ↓  normalize (diacritics / hamza variants / digits)
Bi-Encoder (multilingual-e5) + cosine similarity  →  Top 20 candidates
   ↓
Cross-Encoder (bge-reranker-v2-m3)                →  Top 5, re-ranked
```

A **BM25** baseline is included so every stage can be compared against classic keyword search.

## Results

Recall@1 on the evaluation set (Egyptian, MSA, and Gulf phrasings):

| Method | Recall@1 |
|---|---|
| BM25 (baseline) | 0.678 |
| Dense (multilingual-e5) | 0.789 |
| Dense + Reranker (bge-reranker-v2-m3) | **0.944** |

`evaluate.py` also reports Recall@3, Recall@5, MRR, and a per-dialect breakdown.

## Quick Start

```bash
pip install -r requirements.txt

python evaluate.py              # BM25 vs Dense vs Dense + Reranker
python evaluate.py --no-rerank  # skip the reranker (lighter)
streamlit run app.py            # launch the demo UI
```

The first run downloads the models from Hugging Face (the reranker is roughly 2 GB). On a weaker machine, change `RERANKER_MODEL` in `src/config.py` to a smaller model.

## Project Structure

| File | Purpose |
|---|---|
| `data/bank_faq.json` | Bank FAQ entries, each with test queries in Egyptian, MSA, and Gulf phrasings |
| `src/normalize.py` | Arabic text normalization |
| `src/search.py` | BM25, dense retriever, reranker, and the full pipeline |
| `evaluate.py` | Recall@1/3/5 and MRR, with a per-dialect breakdown |
| `app.py` | Streamlit UI showing results before and after reranking |

## Notes

- The FAQ data is a **sample dataset for a fictional bank**, and the test queries were written by hand. For production-quality numbers, replace it with real FAQs (from bank websites or real customers) and real user phrasings.
- Metrics reflect this specific dataset and should not be read as general benchmarks.

## Roadmap

- [ ] Hybrid search (BM25 + Dense) with Reciprocal Rank Fusion
- [ ] Fine-tune the bi-encoder on (colloquial question → FAQ) pairs with `MultipleNegativesRankingLoss`
- [ ] Confidence threshold: return "no answer found" when the top score is low, instead of a wrong result
- [ ] Scale the dataset and add more dialects (Levantine, Maghrebi)

## Tech Stack

Python · sentence-transformers · rank_bm25 · Hugging Face Transformers · Streamlit
