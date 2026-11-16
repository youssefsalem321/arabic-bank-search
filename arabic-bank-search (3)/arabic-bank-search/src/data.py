import json
from pathlib import Path

from .config import DATA_PATH, OOD_PATH, THRESHOLD_PATH


def load_data(path: Path = DATA_PATH):
    """Returns (faqs, test_queries). Each test query points to its gold faq_id."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    faqs = [{k: v for k, v in r.items() if k != "q"} for r in raw]
    queries = [
        {"query": text, "faq_id": r["id"], "dialect": dialect}
        for r in raw
        for text, dialect in r["q"]
    ]
    return faqs, queries


def load_ood(path: Path = OOD_PATH):
    """Questions that are NOT answerable from the FAQ (should be rejected)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_threshold(path: Path = THRESHOLD_PATH):
    """Recommended rejection threshold written by evaluate.py (or None if not run yet)."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
