"""LoRA fine-tuning of Qwen/Qwen3-0.6B (pinned revision) on TRAIN-split repair prompts.

Only LoRA A/B matrices are trained (no embeddings, lm_head, biases or modules_to_save). Loss is on
the assistant answer tokens only. Writes adapter_model.safetensors + adapter_config.json and a
training log/config JSON next to them.
"""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="local Qwen/Qwen3-0.6B snapshot (revision c1899de...)")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, help="optional JSON list of task ids to leave out (cross-fitting)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro", type=int, default=2, help="micro-batch size (gradient accumulation up to --batch)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora).to(device)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_A" in n or "lora_B" in n for n in trainable), "only LoRA A/B may be trained"
    exclude = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()
    records = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    records = [r for r in records if r["id"] not in exclude and r["messages"][-1]["role"] == "assistant"]
    examples = []
    for r in records:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        full = prompt + r["messages"][-1]["content"] + tok.eos_token
        p_ids = tok(prompt)["input_ids"]
        ids = tok(full)["input_ids"][:1536]
        labels = [-100] * len(p_ids) + ids[len(p_ids):]
        examples.append((ids, labels[:len(ids)]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * ((len(examples) + args.batch - 1) // args.batch)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    log, started = [], time.time()
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch):
            batch = examples[i:i + args.batch]
            n_tok = sum(sum(1 for t in y if t != -100) for _, y in batch)
            total = 0.0
            for j in range(0, len(batch), args.micro):
                part = batch[j:j + args.micro]
                width = max(len(x) for x, _ in part)
                ids = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x, _ in part], device=device)
                lab = torch.tensor([y + [-100] * (width - len(y)) for _, y in part], device=device)
                att = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in part], device=device)
                part_tok = sum(sum(1 for t in y if t != -100) for _, y in part)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss * part_tok / max(1, n_tok)
                loss.backward()
                total += float(loss)
            loss = torch.tensor(total)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            log.append({"epoch": epoch, "step": len(log), "loss": round(float(loss), 5)})
        print(f"epoch {epoch} mean loss {sum(x['loss'] for x in log[-(len(examples)//args.batch):]) / max(1, len(examples)//args.batch):.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    meta = {"base": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "device": device, "examples": len(examples), "excluded": len(exclude), "args": {k: str(v) for k, v in vars(args).items()},
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "seconds": round(time.time() - started, 1), "log": log}
    (args.output / "training_log.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "log"}))


if __name__ == "__main__":
    main()
