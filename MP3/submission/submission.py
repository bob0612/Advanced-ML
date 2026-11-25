"""DASE7506 MP3 entry point: create_agent(checkpoint) -> agent with act(observation)."""
import json
from pathlib import Path

from mp3agent.agent import create

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint is not None else DEFAULT_CONFIG
    return create(json.loads(path.read_text(encoding="utf-8")))
