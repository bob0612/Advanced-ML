"""Replay a logged live-LLM run without the model: the proposer returns the completions recorded in an
llm-log (keyed by task id) and asserts that the prompt the agent builds is identical to the logged one.
Used to evaluate a change that only alters the final choice among already-checked SQL (e.g. plain_twin).
Usage: python replay_run.py --split validation --log outputs/llm-log-validation.jsonl --checkpoint cfg.json --output out.json
"""
import argparse, json, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    os.environ.setdefault("MP3_BASE_MODEL", "replay")
    from mp3agent import llm, prompting
    logged = {}
    for line in a.log.read_text().splitlines():
        r = json.loads(line)
        logged[r["task"]] = r

    class Replay:
        calls = 0
        def propose(self, task_id, user_prompt, **kw):
            r = logged[task_id]
            assert user_prompt in r["prompt"], f"prompt differs for {task_id}"
            Replay.calls += 1
            return [s for s in (prompting.extract_sql(c["text"]) for c in r["completions"]) if s]

    proposer = Replay()
    llm.get_proposer = lambda config: proposer
    sys.argv = ["predict", "--split", a.split, "--method", "submission/submission.py", "--checkpoint",
                str(a.checkpoint), "--workers", "1", "--output", str(a.output)]
    from mp3_sql import predict
    predict.main()
    print(json.dumps({"replayed_llm_calls": Replay.calls, "logged": len(logged)}))


if __name__ == "__main__":
    main()
