"""Initial method: majority vote over parsed final answers (ties -> lowest index). Training-free."""
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parse_answer  # noqa: E402


def predict(bank, arrays, checkpoint):
    out = []
    for row in bank:
        answers = [parse_answer(c['text']) for c in row['candidates']]
        votes = Counter(a for a in answers if a is not None)
        score = [votes.get(a, 0) if a is not None else -1 for a in answers]
        out.append(int(max(range(4), key=lambda i: (score[i], -i))))
    return out
