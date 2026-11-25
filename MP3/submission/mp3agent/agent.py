"""Perceive-plan-act controller.

Loop: check the original draft -> read the shared repair hints -> build a candidate pool
(draft, hints, deterministic rewrites) -> spend the remaining budget checking candidates in
prior order, adding feedback-driven rewrites as evidence arrives -> submit the candidate with
the highest learned score. All state is re-derived from the observation history every step.
"""
import math
import os
import sys

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
        llm = config.get("llm") or {}
        if llm.get("enabled") and not (os.environ.get("MP3_BASE_MODEL") or llm.get("base_dir")):
            # No local base model provided: run the CPU-only controller (identical to llm.enabled = false).
            if not getattr(RepairAgent, "_warned", False):
                print("mp3agent: MP3_BASE_MODEL not set; running without the live LLM", file=sys.stderr)
                RepairAgent._warned = True
            self.config = dict(config, llm=dict(llm, enabled=False))

    # ---------------------------------------------------------------- perception
    @staticmethod
    def _parse_history(observation):
        checked, hints, hint_called, executed = {}, None, False, {}
        for event in observation["history"]:
            tool = event.get("tool")
            if tool == "schema":
                executed["__schema__"] = event.get("observation")
            elif tool == "check":
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
        llm = self.config.get("llm") or {}
        any_match = any((checked.get(c["sql"]) or {}).get("matches_example") for c in pool)
        pending = [c for c in pool if c["sql"] not in checked]
        pending.sort(key=lambda c: -self._prior(observation, c, context))
        reserve = int(llm.get("reserve", 3))
        # Step 3b (optional): when no candidate reproduces the example, ask the LoRA model for new SQL.
        if llm.get("enabled") and not any_match and remaining > 0 and (remaining <= reserve or not pending):
            if "__schema__" not in executed:
                return {"tool": "schema"}
            proposals = self._llm_candidates(observation, pool, checked, context, executed["__schema__"], llm)
            for sql in proposals:
                if sql not in checked and len(sql.encode()) <= MAX_SQL:
                    return {"tool": "check", "sql": sql}
            # Lazy mode: the greedy completion came first; sample more only if it did not reproduce the example.
            if llm.get("lazy") and self._llm_stage == 1 and not any(
                    (checked.get(s) or {}).get("matches_example") for s in proposals):
                proposals = self._llm_candidates(observation, pool, checked, context, executed["__schema__"], llm,
                                                 stage=2)
                for sql in proposals:
                    if sql not in checked and len(sql.encode()) <= MAX_SQL:
                        return {"tool": "check", "sql": sql}
        # Step 3: check unverified candidates in prior order while budget remains.
        if remaining > 0 and n_checks < self.max_checks and pending:
            return {"tool": "check", "sql": pending[0]["sql"]}
        # Step 4a: a model proposal that reproduces the example wins when nothing else does.
        if llm.get("enabled") and not any_match:
            for sql in getattr(self, "_llm", None) or []:
                if (checked.get(sql) or {}).get("matches_example"):
                    return {"tool": "submit", "sql": sql}
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


    def _llm_candidates(self, observation, pool, checked, context, schema, llm, stage=1):
        """Generate once per task (lazy mode: greedy first, samples in stage 2); returns
        de-duplicated SQL ordered by vote count then rank."""
        if getattr(self, "_llm", None) is not None and getattr(self, "_llm_stage", 0) >= stage:
            return self._llm
        from mp3agent import llm as llm_module, prompting
        best = max(pool, key=lambda c: self._score(observation, c, checked.get(c["sql"]), context))
        draft = observation["issue_sql"]
        prompt = prompting.user_prompt(observation["task"]["question"], schema if isinstance(schema, list) else [],
                                       observation["example"], draft, checked.get(draft),
                                       best["sql"], checked.get(best["sql"]))
        seed = int(sqlops.canonical(observation["task"]["id"]).encode().hex()[:8], 16)
        n = int(llm.get("n", 4))
        if llm.get("lazy"):  # stage 1: greedy only; stage 2: the n-1 samples
            mode = "greedy" if stage == 1 else "sample"
            n = 1 if stage == 1 else n - 1
        else:
            mode = "both"
        raw = llm_module.get_proposer(llm).propose(observation["task"]["id"], prompt, n=n,
                                                   max_new_tokens=llm.get("max_new_tokens", 256), seed=seed,
                                                   temperature=llm.get("temperature", 0.7), top_p=llm.get("top_p", 0.9),
                                                   mode=mode)
        raw = list(getattr(self, "_llm_raw", [])) + raw if stage == 2 else raw
        self._llm_raw, self._llm_stage = raw, stage
        votes, order = {}, []
        for sql in raw:
            fixed = sqlops.align_projection(sql, observation["example"]["columns"]) or sql
            key = sqlops.canonical(fixed)
            if key not in votes:
                votes[key] = [0, fixed]
                order.append(key)
            votes[key][0] += 1
        known = {sqlops.canonical(c["sql"]) for c in pool}
        ranked = sorted((k for k in order if k not in known), key=lambda k: -votes[k][0])
        self._llm = [votes[k][1] for k in ranked]
        return self._llm


def create(config):
    return RepairAgent(config)
