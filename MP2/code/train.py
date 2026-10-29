"""Train the two-stage verifier on the training split and write the inference bundle.

Usage (from code/):
    python train.py --output checkpoint/model_bundle
Only training inputs and training references are read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import KFold

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import bank_features
from model import EmbeddingVerifier, HAND_COLS, STACK_NAMES, make_ranker, save_bundle, stack_features

SEED = 0
N_FOLDS = 5
RANKER_SEEDS = (0, 1, 2)
VERIFIER_C = 0.003


def labels(answers, split, data_dir=DATA):
    gold = [parse_answer(g['answer']) for g in read_jsonl(Path(data_dir)/f'gsm8k_{split}.jsonl')]
    return np.array([[a is not None and a == g for a in row] for row, g in zip(answers, gold)], float)


def oof_verifier_scores(emb, y, C=VERIFIER_C, seed=SEED):
    """Out-of-fold stage-1 scores so stage 2 sees honest (not memorised) inputs."""
    oof = np.zeros(y.shape)
    for a, b in KFold(N_FOLDS, shuffle=True, random_state=seed).split(emb):
        oof[b] = EmbeddingVerifier().fit(emb[a], y[a], C=C).predict_proba(emb[b])
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('checkpoint/model_bundle'))
    args = parser.parse_args()
    tick = time.perf_counter()
    bank, arrays = load_inputs('train')
    emb = arrays['embeddings']
    X, answers = bank_features(bank, emb)
    y = labels(answers, 'train')

    oof = oof_verifier_scores(emb, y)
    verifier = EmbeddingVerifier().fit(emb, y, C=VERIFIER_C)
    S = stack_features(X, oof, answers)
    flat_S, flat_y = S.reshape(-1, S.shape[-1]), y.ravel()
    rankers = [make_ranker(s).fit(flat_S, flat_y) for s in RANKER_SEEDS]

    config = {'feature_columns': HAND_COLS + STACK_NAMES, 'verifier_C': VERIFIER_C,
              'n_folds': N_FOLDS, 'fold_seed': SEED, 'ranker_seeds': list(RANKER_SEEDS),
              'train_questions': len(bank)}
    save_bundle(args.output, verifier, rankers, config)
    sizes = {p.name: p.stat().st_size for p in sorted(args.output.iterdir())}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())}
    print(json.dumps({'train_seconds': round(time.perf_counter()-tick, 2), 'bytes': sizes,
                      'total_MiB': sum(sizes.values())/2**20, 'sha256': hashes}, indent=2))


if __name__ == '__main__':
    main()
