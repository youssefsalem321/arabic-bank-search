from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bank_faq.json"
RESULTS_DIR = ROOT / "results"

# Bi-encoder (retriever). Alternatives: "intfloat/multilingual-e5-base", "BAAI/bge-m3"
RETRIEVER_MODEL = "intfloat/multilingual-e5-small"
# Cross-encoder (reranker). Heavy (~2GB) but the strongest multilingual option.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

K_RETRIEVE = 20   # candidates passed from retriever to reranker
K_FINAL = 5       # results shown to the user

# Reranker input: False = (query, FAQ question) only, True = (query, question + answer).
# Run evaluate.py to see which one works better, then set it here.
RERANK_USE_ANSWER = False
FUSION_K = 60     # RRF constant for fusing retriever rank + reranker rank

OOD_PATH = ROOT / "data" / "out_of_domain.json"
THRESHOLD_PATH = RESULTS_DIR / "threshold.json"   # written by evaluate.py, read by app.py

USE_ALIASES = True   # index FAQ "aliases" (extra phrasings) together with the question
MAX_ALIASES = 3
