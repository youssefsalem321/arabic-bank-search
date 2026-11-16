"""Set the rejection thresholds by hand (no need to re-run evaluate.py).
Usage:  python scripts/set_threshold.py --reject 0.003 [--confident 0.03] [--signal reranker]
- reject:    score below this  -> "no answer"
- confident: score below this  -> answer shown with a low-confidence warning
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import THRESHOLD_PATH  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reject", type=float, required=True)
ap.add_argument("--confident", type=float)
ap.add_argument("--signal", choices=["reranker", "dense"])
a = ap.parse_args()

rule = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")) if THRESHOLD_PATH.exists() else {"signal": "reranker", "threshold": a.reject}
rule["reject_threshold"] = a.reject
if a.confident is not None:
    rule["threshold"] = a.confident
if a.signal:
    rule["signal"] = a.signal
rule["threshold"] = max(rule["threshold"], a.reject)
THRESHOLD_PATH.parent.mkdir(exist_ok=True)
THRESHOLD_PATH.write_text(json.dumps(rule, indent=2), encoding="utf-8")
print(f"saved: signal={rule['signal']} reject<{rule['reject_threshold']} low-confidence<{rule['threshold']}")
