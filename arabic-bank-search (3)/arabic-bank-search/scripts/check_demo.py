"""Run the demo questions through the full pipeline (same logic as the app) and print PASS/FAIL.
Run evaluate.py first so the rejection threshold exists.
Usage: python scripts/check_demo.py [path/to/queries.json]     (default: data/demo_queries.json)
Try data/heldout_queries.json for questions that were NOT used to tune anything.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT  # noqa: E402
from src.data import load_data, load_threshold  # noqa: E402
from src.search import SearchPipeline  # noqa: E402


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "demo_queries.json"
    demo = json.loads(path.read_text(encoding="utf-8"))
    faqs, _ = load_data()
    pipe = SearchPipeline(faqs)
    rule = load_threshold()
    if rule is None:
        print("WARNING: results/threshold.json not found -> run evaluate.py first; rejection checks will fail.\n")
    passed = ans_ok = ans_n = rej_ok = rej_n = 0
    for d in demo:
        out = pipe.search(d["query"])
        top = out["reranked"][0]
        score = out["top_similarity"] if (rule and rule["signal"] == "dense") else top["score"]
        rejected = bool(rule) and score < rule.get("reject_threshold", rule["threshold"])
        got = None if rejected else top["id"]
        ok = got == d["expect"]
        passed += ok
        if d["expect"] is None:
            rej_n += 1
            rej_ok += ok
        else:
            ans_n += 1
            ans_ok += ok
        print(f'{"PASS" if ok else "FAIL"}  expect={d["expect"]}  got={got}  (top id {top["id"]}, score {score:.4f})  {d["query"]}')
    print(f"\n{passed}/{len(demo)} passed | answerable correct: {ans_ok}/{ans_n} | out-of-domain rejected: {rej_ok}/{rej_n}")


if __name__ == "__main__":
    main()
