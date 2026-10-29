"""Cache features, answers and correctness labels for every split (development only)."""
import pickle
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_inputs, parse_answer, read_jsonl  # noqa: E402
from features import bank_features  # noqa: E402

OUT = Path(__file__).resolve().parents[1]/'runs'/'cache'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation'):  # test references are never used for development
        bank, arrays = load_inputs(split)
        X, answers = bank_features(bank, arrays['embeddings'])
        gold = [parse_answer(g['answer']) for g in read_jsonl(DATA/f'gsm8k_{split}.jsonl')]
        y = np.array([[a is not None and a == g for a in row] for row, g in zip(answers, gold)], float)
        hard = np.array([len({a for a in row if a is not None}) >= 2 for row in answers])
        with open(OUT/f'{split}.pkl', 'wb') as f:
            pickle.dump({'X': X, 'answers': answers, 'y': y, 'hard': hard,
                         'emb': arrays['embeddings'], 'bank': bank}, f)
        print(split, X.shape, 'hard', hard.sum())


if __name__ == '__main__':
    main()
