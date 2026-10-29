"""Submission interface: two-stage verifier (embedding LR -> boosted group ranker)."""
from pathlib import Path

import numpy as np

from features import bank_features
from model import choose, load_bundle, stack_features

DEFAULT_BUNDLE = Path(__file__).resolve().parent/'checkpoint'/'model_bundle'


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    verifier, rankers, config = load_bundle(Path(checkpoint) if checkpoint else DEFAULT_BUNDLE)
    emb = arrays['embeddings']
    X, answers = bank_features(bank, emb)
    S = stack_features(X, verifier.predict_proba(emb), answers)
    flat = S.reshape(-1, S.shape[-1])
    scores = np.mean([m.predict_proba(flat)[:, 1] for m in rankers], axis=0).reshape(-1, 4)
    return choose(scores)
