"""Perceive-plan-act controller.

Loop: check the original draft -> read the shared repair hints -> build a candidate pool
(draft, hints, deterministic rewrites) -> spend the remaining budget checking candidates in
prior order, adding feedback-driven rewrites as evidence arrives -> submit the candidate with
the highest learned score. All state is re-derived from the observation history every step.
"""
import math

from mp3agent import features, sqlops
from mp3agent.candidates import build_pool

MAX_SQL = 15000


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class RepairAgent:
    def __init__(self, config):
        self.config = config
        self.weights = config["ranker"]["weights"]
        self.bias = config["ranker"].get("intercept", 0.0)
        self.prior_weights = config.get("prior", {}).get("weights")
        self.rewrites = tuple(config.get("rewrites", ["align"]))
        self.use_hints = config.get("use_hints", True)
        self.max_checks = int(config.get("max_checks", 8))
        self.selection = config.get("selection", "learned")

    # ---------------------------------------------------------------- perception
    @staticmethod
    def _parse_history(observation):
        checked, hints, hint_called, executed = {}, None, False, {}
        for event in observation["history"]:
            tool = event.get("tool")
            if tool == "check":
                checked[event["sql"]] = event["observation"]
            elif tool == "execute":
                executed[event["sql"]] = event["observation"]
            elif tool == "repair_hint":
                hint_called = True
                value = event.get("observation")
                hints = [h for h in value if isinstance(h, str)] if isinstance(value, list) else []
        return checked, hints, hint_called, executed

    def _context(self, observation, pool, checked):
        draft = observation["issue_sql"]
        return {"draft_feedback": checked.get(draft), "draft_sql": draft,
                "max_votes": max([c["votes"] for c in pool] + [0]),
                "n_hints": max(1, sum(c["votes"] for c in pool)),
                "draft_struct": sqlops.features_of(draft),
                "twins": features.twin_matches(pool, checked)}

    def _score(self, observation, cand, feedback, context):
        x = features.vector(observation, cand, feedback, context)
        return self.bias + sum(w * v for w, v in zip(self.weights, x))

    def _prior(self, observation, cand, context):
        if not self.prior_weights:
            return 0.0
        x = features.vector(observation, cand, None, context)
        return sum(w * v for w, v in zip(self.prior_weights, x))

    # ---------------------------------------------------------------- policy
    def act(self, observation):
        draft = observation["issue_sql"]
        remaining = observation["remaining"]
        checked, hints, hint_called, executed = self._parse_history(observation)
        n_checks = len(checked)
        # Step 1: always observe the exact original draft first.
        if draft not in checked and remaining > 0 and len(draft.encode()) <= MAX_SQL:
            return {"tool": "check", "sql": draft}
        # Step 2: shared hint cache (permitted only after the draft has been checked).
        if self.use_hints and not hint_called and remaining > 0 and draft in checked:
            return {"tool": "repair_hint"}
        columns = observation["example"]["columns"]
        key = (draft, tuple(hints or []))
        if getattr(self, "_pool_key", None) != key:  # pure function of the observation: memoise
            self._pool_key, self._pool = key, build_pool(draft, hints or [], observation, self.rewrites)
        pool = [dict(c, origins=set(c["origins"])) for c in self._pool]
        context = self._context(observation, pool, checked)
        # Step 3: check unverified candidates in prior order while budget remains.
        if remaining > 0 and n_checks < self.max_checks:
            pending = [c for c in pool if c["sql"] not in checked]
            if pending:
                pending.sort(key=lambda c: -self._prior(observation, c, context))
                return {"tool": "check", "sql": pending[0]["sql"]}
        # Step 4: submit the best-scoring candidate.
        if self.selection == "first_match":  # ablation: first candidate (pool order) matching the example
            for cand in pool:
                fb = checked.get(cand["sql"])
                if fb and fb.get("matches_example"):
                    return {"tool": "submit", "sql": cand["sql"]}
            return {"tool": "submit", "sql": pool[1]["sql"] if len(pool) > 1 else draft}
        if self.selection == "prior":  # ablation: ignore execution feedback when choosing
            best = max(pool, key=lambda c: self._prior(observation, c, context))
            return {"tool": "submit", "sql": best["sql"]}
        best, best_score = draft, -1e9
        for cand in pool:
            score = self._score(observation, cand, checked.get(cand["sql"]), context)
            if score > best_score:
                best, best_score = cand["sql"], score
        return {"tool": "submit", "sql": best}


def create(config):
    return RepairAgent(config)
