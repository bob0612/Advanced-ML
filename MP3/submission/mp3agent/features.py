"""Candidate features. Inputs: the observation (question, draft, public example) and tool feedback."""
import math
import re

from mp3agent import sqlops

REWRITES = ("align", "complete", "dropjoin", "nodistinct", "distinct", "capital")

NAMES = (["bias", "o_draft", "o_hint", "root_hint"] + ["rw_" + r for r in REWRITES] +
         ["votes", "votes_frac", "match", "match_nonempty", "match_empty", "ok", "error",
          "rowcount_eq", "ncols_eq", "first_rows_eq", "col_exact", "col_soft", "draft_match",
          "draft_same", "distinct", "limit", "order", "group", "joins", "joins_vs_draft",
          "tables_vs_draft", "subq", "setop", "ordered_x_order", "limit_x_one",
          "log_len_ratio", "q_distinct_word", "distinct_x_qd", "is_draft_x_draftmatch",
          "max_votes_other"])


def _cells(rows):
    out = []
    for row in rows:
        cells = []
        for cell in row:
            if isinstance(cell, float):
                cell = round(cell, 6)
                if cell == int(cell):
                    cell = int(cell)
            cells.append(repr(cell))
        out.append(tuple(cells))
    return out


def vector(observation, cand, feedback, context):
    """context: dict(draft_feedback, draft_sql, max_votes)."""
    example = observation["example"]
    columns = example["columns"]
    f = dict.fromkeys(NAMES, 0.0)
    f["bias"] = 1.0
    for origin in cand["origins"]:
        if origin in ("draft", "hint"):
            f["o_" + origin] = 1.0
        else:
            kind, _, root = origin.partition("_")
            if "rw_" + kind in f:
                f["rw_" + kind] = 1.0
            if root == "hint":
                f["root_hint"] = 1.0
    f["votes"] = float(cand["votes"])
    f["votes_frac"] = cand["votes"] / max(1, context.get("n_hints", 4))
    f["max_votes_other"] = float(context.get("max_votes", 0) > cand["votes"])
    if feedback is not None:
        match = bool(feedback.get("matches_example"))
        f["match"] = float(match)
        f["match_nonempty"] = float(match and example["row_count"] > 0)
        f["match_empty"] = float(match and example["row_count"] == 0)
        f["ok"] = float(bool(feedback.get("ok")))
        f["error"] = float(not feedback.get("ok"))
        f["rowcount_eq"] = float(feedback.get("row_count") == example["row_count"])
        f["ncols_eq"] = float(len(feedback.get("columns") or []) == len(columns))
        if feedback.get("ok"):
            a = sorted(_cells(feedback["rows"][:8]))
            b = sorted(_cells(example["rows"][:8]))
            f["first_rows_eq"] = float(a == b)
    exact, soft, n = sqlops.column_agreement(cand["sql"], columns)
    f["col_exact"], f["col_soft"] = exact, soft
    dfb = context.get("draft_feedback")
    f["draft_match"] = float(bool(dfb and dfb.get("matches_example")))
    f["draft_same"] = float(cand["sql"] == context.get("draft_sql"))
    f["is_draft_x_draftmatch"] = f["draft_same"] * f["draft_match"]
    s = sqlops.features_of(cand["sql"])
    d = context.get("draft_struct") or {}
    if s.get("parse"):
        f["distinct"] = float(s["distinct"])
        f["limit"] = float(s["limit"])
        f["order"] = float(s["order"])
        f["group"] = float(s["group"])
        f["joins"] = float(min(s["joins"], 4))
        f["subq"] = float(min(s["subqueries"], 3))
        f["setop"] = float(s["setop"])
        if d.get("parse"):
            f["joins_vs_draft"] = float(s["joins"] - d["joins"])
            f["tables_vs_draft"] = float(s["tables"] - d["tables"])
            f["log_len_ratio"] = math.log((s["length"] + 1) / (d["length"] + 1))
    f["ordered_x_order"] = float(observation.get("ordered", False)) * f["order"]
    f["limit_x_one"] = f["limit"] * float(example["row_count"] == 1)
    q = observation["task"]["question"].lower()
    f["q_distinct_word"] = float(bool(re.search(r"\b(distinct|different|unique)\b", q)))
    f["distinct_x_qd"] = f["distinct"] * f["q_distinct_word"]
    return [f[name] for name in NAMES]
