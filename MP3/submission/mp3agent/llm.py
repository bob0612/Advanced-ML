"""Optional live LLM proposer: Qwen/Qwen3-0.6B (pinned revision) + the submitted LoRA adapter.

Loaded once per process from local files only (MP3_BASE_MODEL, MP3_ADAPTER_DIR), safetensors
only, trust_remote_code=False. Every completion is logged (prompt, output, token counts, model
and adapter identity) to MP3_LLM_LOG (default: outputs/llm-log.jsonl).
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from mp3agent import prompting

BASE_REPO = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

_lock = threading.Lock()
_model = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Proposer:
    def __init__(self, base_dir, adapter_dir, threads=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False,
                                                     use_safetensors=True, torch_dtype=torch.float32)
        self.adapter_sha = None
        if adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
            self.adapter_sha = _sha(Path(adapter_dir) / "adapter_model.safetensors")
        # CPU float32 is the graded configuration; MP3_LLM_DEVICE=cuda is for development speed only.
        self.device = os.environ.get("MP3_LLM_DEVICE", "cpu")
        self.model = model.to(self.device).eval()
        self.identity = {"base": BASE_REPO, "revision": BASE_REVISION, "adapter_sha256": self.adapter_sha,
                         "device": os.environ.get("MP3_LLM_DEVICE", "cpu")}
        self.log_path = Path(os.environ.get("MP3_LLM_LOG", "outputs/llm-log.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def propose(self, task_id, user_prompt, n=4, max_new_tokens=256, seed=0, temperature=0.7, top_p=0.9, mode="both"):
        """Return up to n SQL strings. mode "both": one greedy completion plus n-1 seeded samples;
        "greedy": the greedy completion only; "sample": n seeded samples. At most 6 per task overall."""
        n = max(1, min(int(n), 6))
        max_new_tokens = min(int(max_new_tokens), 2048)
        messages = [{"role": "system", "content": prompting.SYSTEM}, {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        torch = self.torch
        outputs = []
        started = time.perf_counter()
        with torch.no_grad():
            if mode in ("both", "greedy"):
                greedy = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                outputs.append(greedy[0])
            k = n - 1 if mode == "both" else (n if mode == "sample" else 0)
            if k > 0:
                torch.manual_seed(seed)
                sampled = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                              temperature=temperature, top_p=top_p, num_return_sequences=k)
                outputs.extend(sampled)
        width = inputs["input_ids"].shape[1]
        texts, sqls = [], []
        for seq in outputs:
            generated = seq[width:]
            texts.append((self.tokenizer.decode(generated, skip_special_tokens=True),
                          int((generated != self.tokenizer.pad_token_id).sum())))
            sqls.append(prompting.extract_sql(texts[-1][0]))
        record = {"task": task_id, "model": self.identity, "prompt": text, "prompt_tokens": int(width),
                  "completions": [{"text": t, "new_tokens": k} for t, k in texts],
                  "seconds": round(time.perf_counter() - started, 3), "max_new_tokens": max_new_tokens,
                  "n": n, "mode": mode, "temperature": temperature, "top_p": top_p, "seed": seed}
        with _lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return [s for s in sqls if s]


def get_proposer(config):
    global _model
    with _lock:
        if _model is None:
            base = os.environ.get("MP3_BASE_MODEL") or config.get("base_dir")
            adapter = os.environ.get("MP3_ADAPTER_DIR", config.get("adapter_dir"))
            if not base:
                raise RuntimeError("Set MP3_BASE_MODEL to the local Qwen/Qwen3-0.6B snapshot directory")
            _model = Proposer(base, adapter or None, int(config.get("threads", 4)))
        return _model
