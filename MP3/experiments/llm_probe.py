"""Dev-only probe (train/validation only, never test): run the LoRA proposer on every task's prompt
and record, per proposal, whether it reproduces the public example and whether it passes all three
grading instances. Input: prompts JSONL from tools/build_lora_data.py.
Usage: python llm_probe.py --split validation --prompts P.jsonl --output O.jsonl [--n 4 --temperature 0.7]
Env: MP3_BASE_MODEL, MP3_ADAPTER_DIR, MP3_LLM_DEVICE=cuda, MP3_LLM_LOG.
"""
import argparse, json, os
from pathlib import Path
from mp3_sql.common import load_bank, result_fingerprint
from mp3_sql.sqlite_tool import execute_sql
from mp3agent import llm, sqlops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "validation"), required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ids", type=Path, help="optional JSON list of ids to probe")
    a = ap.parse_args()
    rows, manifest, folder = load_bank(a.split)
    byid = {r["id"]: r for r in rows}
    refs = json.loads((folder / f"{a.split}_references.json").read_text())["tasks"]
    public = json.loads((folder / "public_results.json").read_text())
    keep = set(json.loads(a.ids.read_text())) if a.ids else None
    prop = llm.get_proposer({})

    def run(row, path, sql):
        tables = [t["name"] for t in manifest["databases"][row["db_id"]]["schema"]]
        return execute_sql(folder / path, sql, tables)

    out = a.output.open("w")
    for line in a.prompts.read_text().splitlines():
        rec = json.loads(line)
        if keep is not None and rec["id"] not in keep:
            continue
        row = byid[rec["id"]]
        user = rec["messages"][1]["content"]
        seed = int(sqlops.canonical(row["id"]).encode().hex()[:8], 16)
        raw = prop.propose(row["id"], user, n=a.n, seed=seed, temperature=a.temperature)
        res = []
        for i, sql in enumerate(raw):
            fixed = sqlops.align_projection(sql, row["example"]["columns"]) or sql
            r = run(row, manifest["databases"][row["db_id"]]["public_path"], fixed)
            match = bool(r["ok"] and result_fingerprint(r, row["ordered"]) == public[row["id"]])
            ok = all((lambda x: x["ok"] and result_fingerprint(x, row["ordered"]) == c["expected"])(run(row, c["path"], fixed))
                     for c in refs[row["id"]])
            res.append({"i": i, "sql": fixed, "match": match, "correct": ok})
        out.write(json.dumps({"id": row["id"], "proposals": res}) + "\n"); out.flush()


if __name__ == "__main__":
    main()
