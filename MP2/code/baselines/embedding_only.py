"""Ablation: stage-1 embedding verifier alone (uses verifier.npz from the main bundle)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import choose, load_bundle  # noqa: E402


def predict(bank, arrays, checkpoint):
    verifier, _, _ = load_bundle(Path(checkpoint))
    return choose(verifier.predict_proba(arrays['embeddings']))
