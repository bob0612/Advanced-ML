"""Fit the candidate ranker and check-order prior on the TRAIN split only, and write config.json.

Usage (from the official package's top-level directory, with this export on PYTHONPATH):
    PYTHONPATH="$PWD/submission" python submission/tools/train_ranker.py --output submission/config.json

Every candidate is executed with the official read-only executor (mp3_sql.sqlite_tool). The
public-database result reproduces exactly the feedback the agent receives from the `check`
tool; the three train grading instances (train_references.json) give the label. Validation and
test files are never opened. Requires numpy and scikit-learn (training only).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from mp3_sql.common import ROOT, load_bank, result_fingerprint
from mp3_sql.sqlite_tool import execute_sql
from mp3agent import features, sqlops
from mp3agent.candidates import build_pool

REWRITES = ["align", "complete", "dropjoin", "nodistinct", "distinct", "capital"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--rewrites", default=",".join(REWRITES))
    parser.add_argument("--no-hints", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rewrites = [r for r in args.rewrites.split(",") if r]
    rows, manifest, folder = load_bank("train", args.data_dir)
    public = json.loads((folder / "public_results.json").read_text())
    refs = json.loads((folder / "train_references.json").read_text())["tasks"]
    hints = json.loads((folder / "repair_hints.json").read_text())

    def run(path, row, sql):
        tables = [t["name"] for t in manifest["databases"][row["db_id"]]["schema"]]
        return execute_sql(folder / path, sql, tables)

    def task(row):
        observation = {"task": {"id": row["id"], "question": row["question"]},
                       "issue_sql": row["issue_sql"], "example": row["example"],
                       "ordered": row["ordered"]}
        suggestions = [] if args.no_hints else hints[row["id"]]["suggestions"]
        pool = build_pool(row["issue_sql"], suggestions, observation, tuple(rewrites))
        feedback, labels = [], []
        for cand in pool:
            result = run(manifest["databases"][row["db_id"]]["public_path"], row, cand["sql"])
            fb = {"ok": result["ok"], "columns": result["columns"], "rows": result["rows"][:8],
                  "row_count": len(result["rows"]), "truncated": len(result["rows"]) > 8,
                  "error": result["error"],
                  "matches_example": bool(result["ok"] and result_fingerprint(
                      result, row["ordered"]) == public[row["id"]])}
            ok = True
            for case in refs[row["id"]]:
                r = run(case["path"], row, cand["sql"])
                if not (r["ok"] and result_fingerprint(r, row["ordered"]) == case["expected"]):
                    ok = False
                    break
            feedback.append(fb)
            labels.append(int(ok))
        draft_fb = feedback[0] if pool and pool[0]["sql"] == row["issue_sql"] else None
        context = {"draft_feedback": draft_fb, "draft_sql": row["issue_sql"],
                   "max_votes": max([c["votes"] for c in pool] + [0]),
                   "n_hints": max(1, sum(c["votes"] for c in pool)),
                   "draft_struct": sqlops.features_of(row["issue_sql"])}
        X = [features.vector(observation, c, fb, context) for c, fb in zip(pool, feedback)]
        P = [features.vector(observation, c, None, context) for c in pool]
        return X, P, labels

    with ThreadPoolExecutor(args.workers) as executor:
        results = list(executor.map(task, rows))
    X = np.array([x for r in results for x in r[0]])
    P = np.array([x for r in results for x in r[1]])
    y = np.array([v for r in results for v in r[2]])
    ranker = LogisticRegression(C=args.C, max_iter=20000).fit(X, y)
    prior = LogisticRegression(C=args.C, max_iter=20000).fit(P, y)
    oracle = sum(max(r[2]) if r[2] else 0 for r in results)
    config = {
        "version": 1,
        "rewrites": rewrites,
        "use_hints": not args.no_hints,
        "max_checks": 8,
        "features": list(features.NAMES),
        "ranker": {"weights": [round(float(w), 6) for w in ranker.coef_[0]],
                   "intercept": round(float(ranker.intercept_[0]), 6), "C": args.C},
        "prior": {"weights": [round(float(w), 6) for w in prior.coef_[0]]},
        "training": {"split": "train", "tasks": len(rows), "candidates": int(len(y)),
                     "positive_candidates": int(y.sum()), "tasks_with_a_correct_candidate": int(oracle),
                     "model": "sklearn LogisticRegression (L2, lbfgs)"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=1) + "\n")
    print(json.dumps(config["training"]))


if __name__ == "__main__":
    main()
