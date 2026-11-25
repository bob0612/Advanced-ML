"""Pure-text SQL utilities: normalisation, projection alignment and simple rewrites.

Nothing here touches a database; all database evidence comes from the official tools.
"""
import re

import sqlglot
from sqlglot import exp

_QUAL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.")


def norm_name(text):
    """Normalise a result column label or expression for comparison."""
    text = str(text).strip().strip('`"[]').lower()
    text = _QUAL.sub("", text)
    text = re.sub(r"[`\"\[\]]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def parse(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001 - any parse failure means "no rewrite"
        return None
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return None
    return tree


def render(tree):
    try:
        return tree.sql(dialect="sqlite")
    except Exception:  # noqa: BLE001
        return None


def canonical(sql):
    """Whitespace/case-insensitive identity used for de-duplication."""
    tree = parse(sql)
    text = render(tree) if tree is not None else None
    if text is None:
        text = sql
    text = re.sub(r"\s+", " ", text.strip().rstrip(";"))
    # lower-case everything except string literals (literal case changes the result)
    parts = re.split(r"('(?:[^']|'')*')", text)
    return "".join(part if part.startswith("'") else part.lower() for part in parts)


def projection_labels(select):
    """(alias, normalised expression) for each projected expression."""
    out = []
    for item in select.expressions:
        alias = item.alias if isinstance(item, exp.Alias) else ""
        inner = item.this if isinstance(item, exp.Alias) else item
        try:
            text = inner.sql(dialect="sqlite")
        except Exception:  # noqa: BLE001
            text = str(inner)
        out.append((norm_name(alias) if alias else "", norm_name(text), inner))
    return out


def _label_match(label, alias, expr_text, inner):
    if label == expr_text or (alias and label == alias):
        return 3
    if isinstance(inner, exp.Column) and label == norm_name(inner.name):
        return 3
    # count(*) vs count(x), and other aggregate-function agreement
    m = re.match(r"^(count|sum|avg|min|max)\((.*)\)$", label)
    if m:
        fn, arg = m.groups()
        m2 = re.match(r"^(count|sum|avg|min|max)\((.*)\)$", expr_text)
        if m2 and m2.group(1) == fn:
            if arg == m2.group(2).replace("distinct", "distinct"):
                return 3
            return 2
    if label.split("(")[-1].rstrip(")") == expr_text.split("(")[-1].rstrip(")") and "(" not in label + expr_text:
        return 2
    return 0


def column_agreement(sql, columns):
    """Fraction of example labels exactly matched, and whether the column count agrees."""
    tree = parse(sql)
    if tree is None:
        return 0.0, 0.0, 0
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return 0.0, 0.0, 0
    labels = projection_labels(select)
    if not columns:
        return 0.0, 0.0, len(labels)
    exact = 0
    soft = 0
    for position, label in enumerate(columns):
        label = norm_name(label)
        best = 0
        for index, (alias, text, inner) in enumerate(labels):
            best = max(best, _label_match(label, alias, text, inner) - (0 if index == position else 0.5))
        exact += best >= 3
        soft += best >= 2
    return exact / len(columns), soft / len(columns), len(labels)


def _inline_aliases(select, dropped):
    """Replace references to dropped aliases in ORDER BY/HAVING with their expressions."""
    if not dropped:
        return
    for clause in ("order", "having", "group"):
        node = select.args.get(clause)
        if node is None:
            continue
        for column in list(node.find_all(exp.Column)):
            if not column.table and column.name.lower() in dropped:
                column.replace(dropped[column.name.lower()].copy())


def align_projection(sql, columns):
    """Rewrite the top-level SELECT list so it matches the example's column labels.

    Returns new SQL or None when no confident alignment exists or nothing changes.
    """
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select) or not columns:
        return None
    labels = projection_labels(tree)
    if any(isinstance(item, exp.Star) for item in tree.expressions):
        return None
    chosen, used = [], set()
    for label in columns:
        label = norm_name(label)
        best, best_index = 0, None
        for index, (alias, text, inner) in enumerate(labels):
            if index in used:
                continue
            score = _label_match(label, alias, text, inner)
            if score > best:
                best, best_index = score, index
        if best_index is None or best < 2:
            return None
        used.add(best_index)
        chosen.append(best_index)
    if chosen == list(range(len(labels))):
        return None
    dropped = {}
    for index, (alias, text, inner) in enumerate(labels):
        if index not in used and alias:
            dropped[alias] = inner
    new = tree.copy()
    originals = [item.copy() for item in tree.expressions]
    new.set("expressions", [originals[i] for i in chosen])
    _inline_aliases(new, dropped)
    return render(new)


def toggle_distinct(sql):
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select):
        return None
    new = tree.copy()
    if new.args.get("distinct"):
        new.set("distinct", None)
    else:
        new.set("distinct", exp.Distinct())
    return render(new)


def features_of(sql):
    tree = parse(sql)
    if tree is None:
        return {"parse": 0}
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    return {
        "parse": 1,
        "distinct": int(bool(select is not None and select.args.get("distinct"))),
        "limit": int(tree.args.get("limit") is not None),
        "order": int(tree.args.get("order") is not None),
        "group": int(bool(select is not None and select.args.get("group"))),
        "joins": len(list(tree.find_all(exp.Join))),
        "subqueries": len(list(tree.find_all(exp.Subquery))),
        "setop": int(isinstance(tree, exp.SetOperation)),
        "tables": len({t.name.lower() for t in tree.find_all(exp.Table)}),
        "length": len(sql),
    }


def _label_expression(label):
    """Parse an example column label (e.g. 'count(*)', 'T2.Name') into an unqualified expression."""
    text = _QUAL.sub("", str(label))
    try:
        node = sqlglot.parse_one(text, read="sqlite")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(node, (exp.Column, exp.AggFunc, exp.Count)):
        return None
    if any(isinstance(n, (exp.Select, exp.Subquery)) for n in node.walk()):
        return None
    return node


def complete_projection(sql, columns):
    """Like align_projection, but unmatched labels are inserted as expressions parsed from the label."""
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select) or not columns:
        return None
    if any(isinstance(item, exp.Star) for item in tree.expressions):
        return None
    labels = projection_labels(tree)
    originals = [item.copy() for item in tree.expressions]
    chosen, used, inserted = [], set(), 0
    for label_text in columns:
        label = norm_name(label_text)
        best, best_index = 0, None
        for index, (alias, text, inner) in enumerate(labels):
            if index in used:
                continue
            score = _label_match(label, alias, text, inner)
            if score > best:
                best, best_index = score, index
        if best_index is not None and best >= 2:
            used.add(best_index)
            chosen.append(originals[best_index])
            continue
        node = _label_expression(label_text)
        if node is None:
            return None
        chosen.append(node)
        inserted += 1
    if not inserted:
        return None
    dropped = {}
    for index, (alias, text, inner) in enumerate(labels):
        if index not in used and alias:
            dropped[alias] = inner
    new = tree.copy()
    new.set("expressions", chosen)
    _inline_aliases(new, dropped)
    return render(new)


def _table_refs(tree):
    """Map lower-case alias/name -> exp.Table for the top-level FROM/JOIN tables."""
    refs = {}
    from_ = tree.args.get("from")
    tables = []
    if from_ is not None and isinstance(from_.this, exp.Table):
        tables.append(from_.this)
    for join in tree.args.get("joins") or []:
        if isinstance(join.this, exp.Table):
            tables.append(join.this)
    for table in tables:
        refs[(table.alias or table.name).lower()] = table
    return refs, tables


def drop_unused_joins(sql):
    """Remove top-level INNER JOINed tables referenced only by their own ON condition."""
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select):
        return None
    joins = tree.args.get("joins") or []
    if not joins:
        return None
    refs, tables = _table_refs(tree)
    if len(tables) != len(joins) + 1:
        return None
    if any(c.table == "" for c in tree.find_all(exp.Column)) and len(tables) > 1:
        unqualified = True
    else:
        unqualified = False
    if unqualified:
        return None
    keep = []
    changed = False
    for join in joins:
        if join.args.get("side") or join.args.get("kind") not in (None, "", "INNER"):
            keep.append(join)
            continue
        name = (join.this.alias or join.this.name).lower()
        used = False
        for column in tree.find_all(exp.Column):
            if column.table.lower() != name:
                continue
            # references inside this join's own ON clause do not count
            parent = column
            inside_own = False
            while parent is not None:
                if parent is join:
                    inside_own = True
                    break
                parent = parent.parent
            if not inside_own:
                used = True
                break
        if used:
            keep.append(join)
        else:
            changed = True
    if not changed:
        return None
    # remaining ON clauses must not reference removed tables
    removed = {(j.this.alias or j.this.name).lower() for j in joins if j not in keep}
    for join in keep:
        on = join.args.get("on")
        if on is not None and any(c.table.lower() in removed for c in on.find_all(exp.Column)):
            return None
    new = tree.copy()
    new.set("joins", [j.copy() for j in keep])
    return render(new)


def capitalise_literals(sql):
    """Title-case lower-case string literals used in equality/IN comparisons (value grounding guess)."""
    tree = parse(sql)
    if tree is None:
        return None
    new = tree.copy()
    changed = False
    for literal in list(new.find_all(exp.Literal)):
        if not literal.is_string:
            continue
        parent = literal.parent
        if not isinstance(parent, (exp.EQ, exp.NEQ, exp.In)):
            continue
        value = literal.this
        if value and value == value.lower() and any(ch.isalpha() for ch in value):
            literal.replace(exp.Literal.string(value[:1].upper() + value[1:]))
            changed = True
    return render(new) if changed else None


def remove_distinct(sql):
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select) or not tree.args.get("distinct"):
        return None
    new = tree.copy()
    new.set("distinct", None)
    return render(new)


def add_distinct(sql):
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select) or tree.args.get("distinct"):
        return None
    new = tree.copy()
    new.set("distinct", exp.Distinct())
    return render(new)


# ---------------------------------------------------------------- phase-2 (feedback-driven) edits

def _literal_variants(sql, fn):
    tree = parse(sql)
    if tree is None:
        return None
    new = tree.copy()
    changed = False
    for literal in list(new.find_all(exp.Literal)):
        if not literal.is_string or not isinstance(literal.parent, (exp.EQ, exp.NEQ, exp.In, exp.Like)):
            continue
        value = literal.this
        if not value or not any(ch.isalpha() for ch in value):
            continue
        updated = fn(value)
        if updated != value:
            literal.replace(exp.Literal.string(updated))
            changed = True
    return render(new) if changed else None


def lower_literals(sql):
    return _literal_variants(sql, lambda v: v.lower())


def title_literals(sql):
    return _literal_variants(sql, lambda v: v.title())


def like_literals(sql):
    """col = 'x'  ->  col LIKE '%x%' (partial value match)."""
    tree = parse(sql)
    if tree is None:
        return None
    new = tree.copy()
    changed = False
    for node in list(new.find_all(exp.EQ)):
        left, right = node.this, node.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
            node.replace(exp.Like(this=left.copy(), expression=exp.Literal.string("%" + right.this + "%")))
            changed = True
    return render(new) if changed else None


def flip_inequalities(sql):
    swap = {exp.GT: exp.GTE, exp.GTE: exp.GT, exp.LT: exp.LTE, exp.LTE: exp.LT}
    tree = parse(sql)
    if tree is None:
        return None
    new = tree.copy()
    changed = False
    for node in list(new.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE)):
        if isinstance(node.expression, exp.Literal) and not node.expression.is_string:
            node.replace(swap[type(node)](this=node.this.copy(), expression=node.expression.copy()))
            changed = True
    return render(new) if changed else None


def flip_order(sql):
    tree = parse(sql)
    if tree is None or tree.args.get("order") is None:
        return None
    new = tree.copy()
    for ordered in new.args["order"].expressions:
        ordered.set("desc", not ordered.args.get("desc"))
    return render(new)


def drop_limit(sql):
    tree = parse(sql)
    if tree is None or tree.args.get("limit") is None:
        return None
    new = tree.copy()
    new.set("limit", None)
    return render(new)


def add_limit_one(sql):
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select) or tree.args.get("limit") is not None \
            or tree.args.get("order") is None:
        return None
    return render(tree.copy().limit(1))


def drop_conjuncts(sql):
    """One candidate per removable top-level WHERE conjunct."""
    tree = parse(sql)
    if tree is None or not isinstance(tree, exp.Select) or tree.args.get("where") is None:
        return []
    condition = tree.args["where"].this
    parts = list(condition.flatten()) if isinstance(condition, exp.And) else [condition]
    if len(parts) < 2:
        return []
    out = []
    for skip in range(len(parts)):
        keep = [p.copy() for i, p in enumerate(parts) if i != skip]
        new = tree.copy()
        new.set("where", exp.Where(this=exp.and_(*keep)))
        text = render(new)
        if text:
            out.append(text)
    return out
