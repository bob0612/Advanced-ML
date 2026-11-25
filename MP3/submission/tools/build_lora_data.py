"""Build chat-format LoRA training data from the TRAIN split only.

Each example: the prompt the agent would build at inference time (schema from the manifest, which
is what the `schema` tool returns; draft and best-candidate feedback from the official executor
on the public database) and, as the target, the train reference SQL from train_memory.jsonl.
With --split validation it writes prompts without targets, for evaluation only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from mp3_sql.common import ROOT, load_bank, read_jsonl, result_fingerprint
from mp3_sql.sqlite_tool import execute_sql
from mp3agent import features, prompting, sqlops
from mp3agent.candidates import build_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=1,
                        help="train only: one example per each of the top-k ranked candidates shown as 'best so far'")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows, manifest, folder = load_bank(args.split, args.data_dir)
    public = json.loads((folder / "public_results.json").read_text())
    hints = json.loads((folder / "repair_hints.json").read_text())
    gold = {m["id"]: m["gold_sql"] for m in read_jsonl(folder / "train_memory.jsonl")} if args.split == "train" else {}
    weights, bias = config["ranker"]["weights"], config["ranker"]["intercept"]

    def feedback(row, sql):
        meta = manifest["databases"][row["db_id"]]
        r = execute_sql(folder / meta["public_path"], sql, [t["name"] for t in meta["schema"]])
        return {"ok": r["ok"], "columns": r["columns"], "rows": r["rows"][:8], "row_count": len(r["rows"]),
                "truncated": len(r["rows"]) > 8, "error": r["error"],
                "matches_example": bool(r["ok"] and result_fingerprint(r, row["ordered"]) == public[row["id"]])}

    def build(row):
        obs = {"task": {"id": row["id"], "question": row["question"]}, "issue_sql": row["issue_sql"],
               "example": row["example"], "ordered": row["ordered"]}
        pool = build_pool(row["issue_sql"], hints[row["id"]]["suggestions"], obs, tuple(config["rewrites"]))
        checked = {c["sql"]: feedback(row, c["sql"]) for c in pool}
        ctx = {"draft_feedback": checked[row["issue_sql"]], "draft_sql": row["issue_sql"],
               "max_votes": max(c["votes"] for c in pool), "n_hints": max(1, sum(c["votes"] for c in pool)),
               "draft_struct": sqlops.features_of(row["issue_sql"]), "twins": features.twin_matches(pool, checked)}
        ranked = sorted(pool, key=lambda c: -(bias + sum(w * v for w, v in zip(
            weights, features.vector(obs, c, checked[c["sql"]], ctx)))))
        k = args.variants if row["id"] in gold else 1
        out = []
        for v, best in enumerate(ranked[:k]):
            prompt = prompting.user_prompt(row["question"], manifest["databases"][row["db_id"]]["schema"],
                                           row["example"], row["issue_sql"], checked[row["issue_sql"]],
                                           best["sql"], checked[best["sql"]])
            record = {"id": row["id"], "variant": v,
                      "messages": [{"role": "system", "content": prompting.SYSTEM},
                                   {"role": "user", "content": prompt}]}
            if row["id"] in gold:
                record["messages"].append({"role": "assistant", "content": prompting.clean_sql(gold[row["id"]])})
            record["any_nonempty_match"] = any(fb["matches_example"] for fb in checked.values()) and row["example"]["row_count"] > 0
            out.append(record)
        return out

    with ThreadPoolExecutor(4) as pool:
        records = [r for group in pool.map(build, rows) for r in group]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(len(records), "records ->", args.output)


if __name__ == "__main__":
    main()
