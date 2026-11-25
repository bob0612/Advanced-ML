"""Candidate pool: the draft, the shared repair hints, and deterministic rewrites of both."""
from mp3agent import sqlops

REWRITERS = {
    "align": lambda sql, obs: sqlops.align_projection(sql, obs["example"]["columns"]),
    "complete": lambda sql, obs: sqlops.complete_projection(sql, obs["example"]["columns"]),
    "dropjoin": lambda sql, obs: sqlops.drop_unused_joins(sql),
    "nodistinct": lambda sql, obs: sqlops.remove_distinct(sql),
    "distinct": lambda sql, obs: sqlops.add_distinct(sql),
    "capital": lambda sql, obs: sqlops.capitalise_literals(sql),
}


def build_pool(draft, hints, observation, rewrites=("align",), depth=2):
    """Return a list of dict(sql, origin, origins, votes, parent) with unique canonical SQL.

    origin names: draft, hint, or '<rewrite>_<root>' where root is draft/hint.
    Rewrites are applied up to `depth` times (compositions such as dropjoin then align).
    """
    pool, index = [], {}

    def add(sql, origin, root, parent=None):
        if not sql or len(sql.encode()) > 15000:
            return None
        key = sqlops.canonical(sql)
        if key in index:
            item = pool[index[key]]
            item["origins"].add(origin)
            if origin == "hint":
                item["votes"] += 1
            return None
        index[key] = len(pool)
        item = {"sql": sql, "origin": origin, "origins": {origin}, "root": root,
                "votes": int(origin == "hint"), "parent": parent}
        pool.append(item)
        return item

    add(draft, "draft", "draft")
    for hint in hints or []:
        if isinstance(hint, str):
            add(hint, "hint", "hint")
    frontier = list(pool)
    for _ in range(depth):
        new_items = []
        for item in frontier:
            for name in rewrites:
                try:
                    sql = REWRITERS[name](item["sql"], observation)
                except Exception:  # noqa: BLE001 - a failed rewrite simply yields nothing
                    sql = None
                created = add(sql, name + "_" + item["root"], item["root"], item["sql"])
                if created is not None:
                    new_items.append(created)
        frontier = new_items
    return pool
