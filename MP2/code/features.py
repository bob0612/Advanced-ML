"""Hand-crafted candidate features shared by training and inference.

Every feature is computed from the question, the four candidate texts and the
fixed embeddings only; no reference answer is ever read here.
"""
from collections import Counter
from fractions import Fraction
import math
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import numpy as np

from common import parse_answer

NUM_RE = re.compile(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?')
HEDGES = ('wait', 'however', 'but ', 'mistake', 'let’s try', "let's try", 'try again',
          'recheck', 're-evaluate', 'reconsider', 'assume', 'interpret', 'contradiction',
          'not an integer', 'approximately', '\\approx', '≈', 'actually', 'correct answer',
          'double-check', 'hmm')


def _to_fraction(text):
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------- arithmetic

_LATEX_SUBS = [
    (re.compile(r'\\text\{[^{}]*\}'), ' '),
    (re.compile(r'\\(?:mathrm|textbf|mathbf)\{([^{}]*)\}'), r'\1'),
    (re.compile(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}'), r'((\1)/(\2))'),
    (re.compile(r'\\times|\\cdot|×|·|\*'), '*'),
    (re.compile(r'\\div|÷'), '/'),
    (re.compile(r'\\left|\\right|\\[,;!quad]+|\\\$|\$|\\\(|\\\)|\\\[|\\\]'), ' '),
    (re.compile(r'(?<=\d),(?=\d{3}\b)'), ''),
    (re.compile(r'\\%|%'), ' '),
    (re.compile(r'\{|\}'), ' '),
]
_EXPR_OK = re.compile(r'^[\d.+\-*/() ]+$')


def _clean(seg):
    for pat, rep in _LATEX_SUBS:
        seg = pat.sub(rep, seg)
    return seg


def _eval_side(seg, left):
    """Evaluate the numeric expression adjacent to an '=' sign, or None."""
    seg = _clean(seg).strip()
    # keep the trailing (left side) / leading (right side) arithmetic run only
    if left:
        m = re.search(r'[\d.+\-*/() ]+$', seg)
    else:
        m = re.match(r'^[\d.+\-*/() ]+', seg)
    if not m:
        return None
    expr = m.group().strip()
    # two numbers separated only by spaces belong to different expressions
    pieces = re.split(r'(?<=[\d)])\s+(?=[\d(])', expr)
    expr = (pieces[-1] if left else pieces[0]).strip()
    if not expr or not re.search(r'\d', expr) or not _EXPR_OK.match(expr):
        return None
    if expr.count('(') != expr.count(')'):
        return None
    try:
        value = eval(expr, {'__builtins__': {}}, {})  # digits/operators only
    except Exception:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    has_op = bool(re.search(r'[\d)]\s*[+\-*/]\s*[\d(]', expr))
    return value, has_op


def arithmetic_check(text):
    """Count checkable '=' steps and how many are numerically wrong."""
    checked = wrong = 0
    for line in text.replace('\\\\', '\n').splitlines():
        line = line.replace('\\approx', '≈').replace('\\neq', '≠')
        if '=' not in line or '≈' in line:
            continue
        parts = line.split('=')
        for a, b in zip(parts, parts[1:]):
            if not a or not b or a.endswith(('<', '>', '!')):
                continue
            left, right = _eval_side(a, True), _eval_side(b, False)
            if left is None or right is None or not (left[1] or right[1]):
                continue
            checked += 1
            x, y = left[0], right[0]
            if abs(x - y) > 1e-6 + 0.011*max(abs(x), abs(y)):
                wrong += 1
    return checked, wrong


# ---------------------------------------------------------------- features

FEATURE_NAMES = [
    # answer form
    'ans_none', 'ans_int', 'ans_neg', 'ans_zero', 'ans_log_mag', 'ans_in_question',
    'ans_is_question_num_count',
    # consensus
    'vote_share', 'vote_is_max', 'vote_unique_max', 'n_distinct', 'n_none_group',
    'vote_rank',
    # surface
    'log_chars', 'log_lines', 'n_boxed', 'has_hash', 'n_steps', 'hedge_count',
    'n_numbers', 'frac_numbers_from_q', 'rel_len',
    # arithmetic verification
    'arith_checked', 'arith_wrong', 'arith_any_wrong', 'arith_wrong_frac',
    'final_num_matches_ans',
    # embeddings
    'emb_sim_same', 'emb_sim_diff', 'emb_sim_mean', 'emb_sim_centroid',
]


def question_numbers(question):
    out = set()
    for tok in NUM_RE.findall(question):
        f = _to_fraction(tok.replace(',', ''))
        if f is not None:
            out.add(f)
    return out


def candidate_basic(question_nums, text):
    ans = parse_answer(text)
    f = _to_fraction(ans) if ans is not None else None
    low = text.lower()
    nums = [_to_fraction(t.replace(',', '')) for t in NUM_RE.findall(text)]
    nums = [n for n in nums if n is not None]
    checked, wrong = arithmetic_check(text)
    last_nums = nums[-1:] if nums else []
    return {
        'answer': ans,
        'ans_none': float(f is None),
        'ans_int': float(f is not None and f.denominator == 1),
        'ans_neg': float(f is not None and f < 0),
        'ans_zero': float(f is not None and f == 0),
        'ans_log_mag': math.log10(1 + abs(float(f))) if f is not None else 0.0,
        'ans_in_question': float(f is not None and f in question_nums),
        'ans_is_question_num_count': float(sum(1 for n in nums if n in question_nums)) / (1 + len(nums)),
        'log_chars': math.log(1 + len(text)),
        'log_lines': math.log(1 + text.count('\n')),
        'n_boxed': float(len(re.findall(r'\\boxed\{', text))),
        'has_hash': float('####' in text),
        'n_steps': float(len(re.findall(r'(?im)^#+\s*step', text))),
        'hedge_count': float(sum(low.count(h) for h in HEDGES)),
        'n_numbers': math.log(1 + len(nums)),
        'frac_numbers_from_q': sum(1 for n in nums if n in question_nums) / max(1, len(nums)),
        'arith_checked': math.log(1 + checked),
        'arith_wrong': float(wrong),
        'arith_any_wrong': float(wrong > 0),
        'arith_wrong_frac': wrong / max(1, checked),
        'final_num_matches_ans': float(f is not None and bool(last_nums) and last_nums[-1] == f),
    }


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def group_features(row, emb):
    """Return a [4, F] feature matrix and the four parsed answers."""
    qn = question_numbers(row['question'])
    basics = [candidate_basic(qn, c['text']) for c in row['candidates']]
    answers = [b['answer'] for b in basics]
    counts = Counter(a for a in answers if a is not None)
    top = max(counts.values()) if counts else 0
    n_top = sum(1 for v in counts.values() if v == top)
    sorted_counts = sorted(set(counts.values()), reverse=True)
    lens = np.array([b['log_chars'] for b in basics])
    e = _unit(emb.astype(np.float64))
    sim = e @ e.T
    centroid = _unit(e.mean(0))
    out = []
    for i, b in enumerate(basics):
        a = answers[i]
        v = counts.get(a, 0) if a is not None else 0
        same = [sim[i, j] for j in range(4) if j != i and a is not None and answers[j] == a]
        diff = [sim[i, j] for j in range(4) if j != i and not (a is not None and answers[j] == a)]
        feats = dict(b)
        feats.update({
            'vote_share': v/4,
            'vote_is_max': float(v == top and v > 0),
            'vote_unique_max': float(v == top and v > 0 and n_top == 1),
            'n_distinct': float(len(counts)),
            'n_none_group': float(sum(x is None for x in answers)),
            'vote_rank': float(sorted_counts.index(v)) if v in sorted_counts else 4.0,
            'rel_len': b['log_chars'] - lens.mean(),
            'emb_sim_same': float(np.mean(same)) if same else 0.0,
            'emb_sim_diff': float(np.mean(diff)) if diff else 0.0,
            'emb_sim_mean': float((sim[i].sum()-1)/3),
            'emb_sim_centroid': float(e[i] @ centroid),
        })
        out.append([feats[k] for k in FEATURE_NAMES])
    return np.array(out, dtype=np.float64), answers


def bank_features(bank, embeddings):
    feats, answers = [], []
    for row, emb in zip(bank, embeddings):
        f, a = group_features(row, emb)
        feats.append(f)
        answers.append(a)
    return np.stack(feats), answers
