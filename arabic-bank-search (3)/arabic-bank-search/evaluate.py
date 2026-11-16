"""Compare BM25 / Dense / Rerank / Fusion.
Usage: python evaluate.py [--no-rerank]
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np

from src.config import K_RETRIEVE, RERANK_USE_ANSWER, RESULTS_DIR, THRESHOLD_PATH
from src.data import load_data, load_ood
from src.search import BM25Retriever, DenseRetriever, Reranker, rrf_fuse


def metrics(ranked_lists, gold):
    r1 = r3 = r5 = mrr = 0.0
    for ranked, g in zip(ranked_lists, gold):
        if g in ranked:
            pos = ranked.index(g) + 1
            mrr += 1 / pos
            r1 += pos <= 1
            r3 += pos <= 3
            r5 += pos <= 5
    n = len(gold)
    return {"Recall@1": r1 / n, "Recall@3": r3 / n, "Recall@5": r5 / n, "MRR": mrr / n}


def best_threshold(pos, neg):
    """pos = top-1 scores of answerable queries (accept if score >= t),
    neg = top-1 scores of out-of-domain queries (reject if score < t).
    Picks t maximizing balanced accuracy."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    best = None
    for t in sorted(set(pos) | set(neg)):
        tpr, tnr = float(np.mean(pos >= t)), float(np.mean(neg < t))
        bal = (tpr + tnr) / 2
        if best is None or bal > best["balanced_accuracy"]:
            best = {"threshold": float(t), "balanced_accuracy": bal,
                    "in_domain_accepted": tpr, "ood_rejected": tnr}
    return best


def calibrate(pos, neg, min_accept=0.98):
    """Two thresholds:
    - threshold (confident): best balanced accuracy -> below it we answer with a low-confidence warning
    - reject_threshold: highest cut that still accepts >= min_accept of answerable queries -> below it: 'no answer'"""
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = best_threshold(pos, neg)
    t_rej = min(float(np.quantile(pos, 1 - min_accept)), r["threshold"])
    r.update({"reject_threshold": t_rej,
              "accepted_at_reject": float(np.mean(pos >= t_rej)),
              "ood_rejected_at_reject": float(np.mean(neg < t_rej))})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true", help="skip the heavy reranker")
    ap.add_argument("--fast", action="store_true", help="rerank only with the RERANK_USE_ANSWER setting (half the time)")
    ap.add_argument("--max-queries", type=int, default=0, help="evaluate on a random sample of N queries")
    args = ap.parse_args()

    faqs, queries = load_data()
    if args.max_queries and len(queries) > args.max_queries:
        queries = random.Random(0).sample(queries, args.max_queries)
    ids = [f["id"] for f in faqs]
    gold = [q["faq_id"] for q in queries]
    print(f"{len(faqs)} FAQs | {len(queries)} test queries\n")

    bm25 = BM25Retriever(faqs)
    dense = DenseRetriever(faqs)
    dense_hits = [dense.search(q["query"], K_RETRIEVE) for q in queries]

    rankings = {
        "BM25 (keywords)": [[ids[i] for i, _ in bm25.search(q["query"], K_RETRIEVE)] for q in queries],
        "Dense (bi-encoder)": [[ids[i] for i, _ in h] for h in dense_hits],
    }

    signals_in = {"dense": [h[0][1] for h in dense_hits]}
    rr = None
    if not args.no_rerank:
        rr = Reranker()
        variants = [(True, "Q+A"), (False, "Q only")]
        if args.fast:
            variants = [v for v in variants if v[0] == RERANK_USE_ANSWER]
        for use_answer, label in variants:
            print(f"Reranking ({label}) ...")
            all_scores = [rr.score(q["query"], faqs, h, use_answer) for q, h in zip(queries, dense_hits)]
            if use_answer == RERANK_USE_ANSWER:
                signals_in["reranker"] = [float(np.max(sc)) for sc in all_scores]
            rankings[f"Rerank ({label})"] = [
                [ids[h[j][0]] for j in np.argsort(-s)] for h, s in zip(dense_hits, all_scores)
            ]
            rankings[f"Fusion ({label})"] = [
                [ids[i] for i, _ in rrf_fuse(h, s, K_RETRIEVE)] for h, s in zip(dense_hits, all_scores)
            ]

    results = {name: metrics(r, gold) for name, r in rankings.items()}

    cols = ["Recall@1", "Recall@3", "Recall@5", "MRR"]
    print(f"\n{'System':<24}" + "".join(f"{c:>10}" for c in cols))
    for name, m in results.items():
        print(f"{name:<24}" + "".join(f"{m[c]:>10.3f}" for c in cols))

    print("\nRecall@1 by dialect")
    by_d = defaultdict(list)
    for i, q in enumerate(queries):
        by_d[q["dialect"]].append(i)
    print(f"{'System':<24}" + "".join(f"{d:>10}" for d in by_d))
    per_dialect = {}
    for name in rankings:
        row = {
            d: metrics([rankings[name][i] for i in idxs], [gold[i] for i in idxs])["Recall@1"]
            for d, idxs in by_d.items()
        }
        per_dialect[name] = row
        print(f"{name:<24}" + "".join(f"{row[d]:>10.3f}" for d in by_d))

    # ---- Out-of-domain rejection: how to say "no answer" ----
    ood = load_ood()
    ood_hits = [dense.search(q, K_RETRIEVE) for q in ood]
    signals_ood = {"dense": [h[0][1] for h in ood_hits]}
    if rr is not None:
        signals_ood["reranker"] = [
            float(np.max(rr.score(q, faqs, h, RERANK_USE_ANSWER))) for q, h in zip(ood, ood_hits)
        ]
    print(f"\nOut-of-domain rejection ({len(ood)} unanswerable vs {len(queries)} answerable queries)")
    print(f"{'Signal':<10}{'mean(in)':>10}{'mean(ood)':>10}{'conf.thr':>11}{'reject.thr':>12}{'accept in':>11}{'reject ood':>12}")
    calib = {}
    for sig in signals_in:
        r = calibrate(signals_in[sig], signals_ood[sig])
        calib[sig] = r
        print(f"{sig:<10}{np.mean(signals_in[sig]):>10.4f}{np.mean(signals_ood[sig]):>10.4f}"
              f"{r['threshold']:>11.5f}{r['reject_threshold']:>12.5f}{r['accepted_at_reject']:>11.3f}{r['ood_rejected_at_reject']:>12.3f}")
    best_sig = max(calib, key=lambda k: calib[k]["ood_rejected_at_reject"])
    THRESHOLD_PATH.parent.mkdir(exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps({"signal": best_sig, **calib[best_sig]}, indent=2), encoding="utf-8")
    c = calib[best_sig]
    print(f"-> saved {THRESHOLD_PATH.name}: no answer if {best_sig} < {c['reject_threshold']:.5f}; low-confidence warning if < {c['threshold']:.5f}")

    best = max(results, key=lambda n: results[n]["MRR"])
    print(f"\nBest system by MRR: {best}")
    print("Misses at rank 1:")
    for q, ranked in zip(queries, rankings[best]):
        if ranked[0] != q["faq_id"]:
            print(f"  [{q['dialect']}] {q['query']}  -> got id {ranked[0]}, expected {q['faq_id']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps({"overall": results, "recall@1_by_dialect": per_dialect}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
