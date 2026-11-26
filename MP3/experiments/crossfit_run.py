"""Dev-only (train/validation only, never test): run the agent on a subset of tasks and record, for every
checked SQL, whether it reproduces the public example and whether it passes all grading instances.
Used for cross-fitting: fold k's ranker and LoRA adapter were trained without the fold's tasks.
Usage: python crossfit_run.py --split train --ids fold.json --checkpoint cfg.json --output out.jsonl
Env: MP3_BASE_MODEL, MP3_ADAPTER_DIR, MP3_LLM_DEVICE, MP3_LLM_LOG.
"""
import argparse, json
from pathlib import Path
from mp3_sql.common import load_bank, result_fingerprint
from mp3_sql.environment import Tools, rollout
from mp3_sql.sqlite_tool import execute_sql
from mp3_sql.predict import load_factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "validation"), required=True)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--method", type=Path, default=Path("submission/submission.py"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows, manifest, folder = load_bank(a.split)
    refs = json.loads((folder / f"{a.split}_references.json").read_text())["tasks"]
    keep = set(json.loads(a.ids.read_text())) if a.ids else None
    tools = Tools(folder, manifest)
    factory = load_factory(a.method)
    cache = {}

    def correct(row, sql):
        if sql not in cache:
            tables = [t["name"] for t in manifest["databases"][row["db_id"]]["schema"]]
            ok = True
            for case in refs[row["id"]]:
                r = execute_sql(folder / case["path"], sql, tables)
                if not (r["ok"] and result_fingerprint(r, row["ordered"]) == case["expected"]):
                    ok = False
                    break
            cache[sql] = ok
        return cache[sql]

    done = set()
    if a.output.exists():
        done = {json.loads(l)["id"] for l in a.output.read_text().splitlines()}
    with a.output.open("a") as out:
        for row in rows:
            if (keep is not None and row["id"] not in keep) or row["id"] in done:
                continue
            cache.clear()
            pred = rollout(row, tools, factory(a.checkpoint))
            checks = [{"sql": s["sql"], "match": bool((s.get("observation") or {}).get("matches_example")),
                       "correct": correct(row, s["sql"])} for s in pred["trace"] if s["tool"] == "check"]
            out.write(json.dumps({"id": row["id"], "db_id": row["db_id"], "empty": row["example"]["row_count"] == 0,
                                  "sql": pred["sql"], "passed": correct(row, pred["sql"]),
                                  "tools": [s["tool"] for s in pred["trace"]], "checks": checks}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
