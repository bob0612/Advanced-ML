---
title: "DASE7506 MP3: A Feedback-Driven SQL Repair Agent with a LoRA Proposer"
author: "Student ID 3036797441"
---

**Official test result: 187 / 240 correct (77.92%)**, measured with the unchanged `mp3_sql.predict` and `mp3_sql.score` on all three grading instances per task, in the graded live-LLM configuration: CPU only, 4 threads, `--workers 1`. The starter draft scores 88 / 240 (36.67%). Validation: 101 / 120 (84.17%) against 47 / 120 for the draft. The agent is a deterministic controller with a learned ranker (Sections 1–5) plus one approved live model, `Qwen/Qwen3-0.6B` with a LoRA adapter trained on the train split (Section 6), which proposes new SQL when no other candidate reproduces the public example. Test prediction takes 11 min on 4 CPU threads with a peak of 5.0 GB resident memory. Without the model (no `MP3_BASE_MODEL`), the same export runs as the CPU-only controller: 182 / 240 on test in about 32 s.

# 1. Agent loop

The agent is a perceive-plan-act controller. Its decisions depend only on the current observation and history, so a replayed trace leads to the same actions. The one per-task cache it keeps holds the candidate pool, which is a pure function of the draft and the hints. Each task proceeds as follows.

1. **Perceive the draft.** `check` the exact original draft. This is required before hints, and it shows whether the draft already reproduces the public example and what columns it returns.
2. **Read the hint cache.** `repair_hint` returns four cached Qwen3-8B repairs of the draft. They are treated as untrusted candidate SQL, not as answers.
3. **Plan: build a candidate pool.** The pool holds the draft, the unique hints (with a vote count for repeated hints) and deterministic rewrites of each, applied up to twice (Section 2). Duplicates are removed with a canonical form that keeps the case of string literals.
4. **Act: verify candidates.** While budget remains, the agent `check`s the unverified candidate with the highest *prior* score, a logistic model over features that need no execution. Each check returns real execution feedback: success or error, columns, the first 8 rows, the row count and `matches_example`.
5. **Submit.** A second logistic model, the *ranker*, scores every candidate. It uses the static features, the candidate's own check feedback, and the feedback of its *DISTINCT twins* (Section 4). The agent submits the top candidate. `submit` is free.

6. **Ask the LoRA model (live-LLM configuration only).** If no checked candidate reproduces the example and at most 6 charged calls remain (or nothing is left to check), the agent calls `schema`, builds a prompt and asks the LoRA model for SQL. Its proposals are checked like any other candidate, and a proposal that reproduces the example is submitted when no pool candidate does (Section 6).

Without the model, the agent uses 5.91 charged calls per task on test (minimum 3, maximum 8) and never calls `schema` or `retrieve` (see Section 3).

# 2. Candidate generation: deterministic rewrites

The public example reveals the *labels* of the expected result columns, such as `count(*)`, `max(Silver)` or `T2.Name`. Many drafts and hints compute the right rows but project the wrong columns, in the wrong order or with an extra aggregate. The failure analysis on the train split (Section 7) also showed systematic differences between LLM-style SQL and the Spider-style references. All rewrites are pure text transformations with `sqlglot`. They never touch a database.

| Rewrite | What it does |
|---|---|
| `align` | Reorders or drops SELECT items so that each example label matches one item (by alias, normalised expression, bare column name or aggregate function). Aliases of dropped items used in ORDER BY or HAVING are inlined. |
| `complete` | Like `align`, but inserts an example label that is a simple column or aggregate (for example `COUNT(*)`) when no item matches. |
| `dropjoin` | Removes inner-joined tables whose columns are used only in their own ON clause. This counters over-joining by the LLM. |
| `nodistinct` / `distinct` | Removes or adds DISTINCT on the top-level SELECT. |
| `countstar` / `countall` | `COUNT(DISTINCT x)` or `COUNT(x)` becomes `COUNT(*)`. |
| `argmax` | `WHERE x = (SELECT MAX(x) FROM t)` becomes `ORDER BY x DESC LIMIT 1`. The subquery form also returns ties, while the references return one row. |
| `intersect` | For questions containing "both", `c IN (A, B)` or `c = A OR c = B` becomes the INTERSECT of the two filters. |
| `capital` | Capitalises lower-case string literals in `=`, `<>` and `IN` comparisons (for example `'rock'` becomes `'Rock'`). |

The pool has 5.7 candidates per task on average. With all rewrites, train tasks with at least one fully correct candidate rise from 357/480 (draft and hints only) to 408/480. On validation they rise from 86 to 100 of 120.

# 3. Context, memory and retrieval

* **Context.** The decision context is the question, the draft, the public example (labels, first rows, row count, order flag) and the tool history. The agent keeps the feedback for each checked SQL string. Rewrites and features read the rows and labels only. No free text from the task is executed or interpreted as an instruction.
* **Memory.** There is no memory across tasks. A fresh agent is created per task, as the protocol requires. The only long-term memory is learned: two weight vectors of 48 features each, fitted on the train split and stored in `config.json` (2.5 KB). A process-wide cache of `sqlglot` parse trees only saves time. It returns copies and does not change any output.
* **Retrieval.** The `retrieve` tool was not used. Its results are training tasks from *different* schemas, because schemas are disjoint across splits. Without an LLM to adapt them, their gold SQL cannot be executed against the current database. A charged call spent on retrieval is better spent checking one more candidate: the budget ablation shows the last checks still matter.
* **Hint cache.** The hints are fixed outputs of an untrained LLM. Majority agreement is used as a feature (`votes`), and every hint must pass the same execution check as any other candidate.

# 4. Learned selection

Each candidate has 48 features:

* origin flags (draft, hint, which rewrite) and hint votes;
* check feedback: `matches_example`, a match on a non-empty versus an empty example, error, equal row count, equal column count, equal first rows;
* agreement between the SELECT labels and the example labels;
* structure: DISTINCT, LIMIT, ORDER BY, GROUP BY, joins and subqueries, also relative to the draft;
* question cues, for example the words "distinct", "different" or "unique".

**DISTINCT twins.** Two candidates that differ only in DISTINCT or `COUNT(DISTINCT ...)` return the same rows when the public instance has no duplicates, so both match. On the other instances, duplicates decide which one is right. The train split shows a clear pattern: when both twins match and the question has no "distinct/different/unique" cue, the version without DISTINCT is right in 19 of 23 disagreements. With the cue, the DISTINCT version is right in all 5. Four twin features (the candidate uses DISTINCT or not, crossed with the cue, active only when a twin also matches) let the ranker learn this. The ranker learns positive weights for `twin_plain_noq` (+1.2) and `twin_dist_q` (+1.5).

**Models.** Both models are L2 logistic regressions (C = 1), trained on 2,744 train-split candidates. A label is positive when the candidate is correct on all three train grading instances. Labels come from `train_references.json` only. Training (`tools/train_ranker.py`) runs every candidate through the official read-only executor and is deterministic. The largest ranker weights are `match` +2.7, `limit_x_one` −2.4, `col_exact` +2.3, `match_nonempty` +2.3 and `first_rows_eq` +1.8. A match on an empty example is weak evidence (+0.4), because many wrong queries also return nothing.

**Model selection.** Validation has only 120 tasks, so design choices were also checked with 5-fold cross-validation on train, grouped by database (all candidates checked, selection accuracy out of 480). The original six rewrites score 391. Adding `countstar` and `argmax` gives 394, adding `intersect` gives 395, and adding `countall` gives 396. Gradient boosting (393–394) did not beat logistic regression, so the simpler model was kept.

# 5. Controlled ablations (validation, 120 tasks)

Each row changes one factor. Every configuration that changes the candidate pool is retrained on train with the same procedure. All scores are official validation scores (all three instances).

| Configuration | Correct | Acc. % | Calls/task |
|---|---:|---:|---:|
| Draft only (starter baseline) | 47 | 39.17 | 0.00 |
| No repair hints (draft + rewrites + ranker) | 55 | 45.83 | 3.02 |
| Hints, no rewrites (+ ranker) | 81 | 67.50 | 3.04 |
| Full pool, first matching candidate (no learned ranker) | 85 | 70.83 | 5.97 |
| Full pool, prior only (feedback not used for the choice) | 89 | 74.17 | 5.97 |
| `align` rewrite only | 91 | 75.83 | 3.24 |
| First six rewrites only (no countstar/countall/argmax/intersect) | 94 | 78.33 | 5.65 |
| Full minus `dropjoin` | 94 | 78.33 | 5.84 |
| Full minus DISTINCT toggles | 94 | 78.33 | 4.05 |
| Full, at most 3 checks | 94 | 78.33 | 3.83 |
| Full, at most 5 checks | 95 | 79.17 | 5.17 |
| Full agent, ranker C = 0.1 / 10 | 94 / 96 | 78.33 / 80.00 | 5.97 |
| **Full controller (no LLM; base of the submitted agent)** | **96** | **80.00** | **5.97** |

![Validation ablations](ablations.png){width=92%}

**Findings.**

1. **Execution feedback is essential.** Choosing with the prior alone loses 7 tasks. A naive "first candidate that matches the example" rule loses 11, because matches on empty or tiny examples are often wrong on the other instances.
2. **The hint cache is the main source of new SQL.** It adds 26 tasks over rewrites of the draft alone.
3. **The rewrites add 15 tasks over hints alone.** Most of the gain comes from projection alignment (+10).
4. **Spending fewer checks costs only 1–2 tasks**, which suggests the prior ordering puts the right candidate early.

The validation differences between the rewrite variants are only 1–2 tasks. The cross-validation in Section 4 is the more reliable signal for those choices.

# 6. Live LoRA proposer

**Why.** Selection is close to its ceiling (train cross-validation 396 against an oracle of 408 out of 480), and 20 of the 24 validation failures of the CPU-only controller have *no* correct SQL in the candidate pool. More gain therefore has to come from generating new SQL, which local rewrites cannot do (wrong table, wrong join path, misread question).

**Model and training.** `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`, loaded with `local_files_only=True`, `use_safetensors=True` and `trust_remote_code=False`. LoRA with r = 16, alpha = 32 and dropout 0.05 on the q, k, v, o, gate, up and down projections; only the LoRA A/B matrices are trainable (asserted in `tools/train_lora.py`), 10.1 M parameters. Training data, built by `tools/build_lora_data.py` from **train** tasks only: the user turn is exactly the prompt the agent builds at run time: the question, the schema as the `schema` tool returns it, the public example, the draft and its execution feedback on the public database, and a "best candidate so far" with its feedback. The target is the train reference SQL. Loss is on the answer tokens only. The first adapter used one example per task (480), with the ranker's top candidate as "best so far". The submitted adapter uses `--variants 3`: one example for each of the ranker's top three candidates (1,359 examples), so the model sees the different states the agent can be in when it calls the model. 2 epochs, AdamW, learning rate 2e-4 with cosine decay, effective batch 8, bf16 on one rented GPU (about 10 minutes on an A40); training loss fell from 1.85 to 0.02. No validation or test material was used for training.

**Inference.** Per triggered task: one greedy completion plus five samples (temperature 1.0, top-p 0.9, seed derived from the task id), at most 256 new tokens each, so at most 6 completions per task as the rules require. The SQL is extracted, its projection is aligned to the example labels, duplicates of pool candidates are removed and the rest are ordered by vote. Every prompt, completion, token count and the model and adapter identity (adapter SHA-256 `ff2d94e1…`) are logged to `outputs/llm-log-*.jsonl`.

**Selection on validation (GPU, same model, run for speed).** Each row is one full validation run with the official scorer.

| LLM setting | Correct / 120 |
|---|---:|
| No LLM (CPU-only controller) | 96 |
| Base model without adapter, 4 completions, trigger when ≤ 3 calls remain | 96 |
| LoRA, 4 completions (temperature 0.7), trigger ≤ 3 | 98 |
| LoRA, 6 completions (temperature 1.0), trigger ≤ 3 | 98 |
| LoRA, 6 completions, trigger ≤ 5 | 97 |
| LoRA, 6 completions, trigger ≤ 6 | 99 |
| LoRA, 6 completions, trigger ≤ 7 | 99 |
| LoRA, greedy first and samples only if greedy misses ("lazy"), trigger ≤ 6 | 97 |
| LoRA trained 5 epochs, trigger ≤ 6 | 97 |
| LoRA r = 32, trigger ≤ 6 | 97 |
| Same as the 99 row, retrained on a second GPU (reproducibility check) | 99 |
| Augmented data (3 variants per task, 1,359 examples), 2 epochs, trigger ≤ 6 | 101 |
| Augmented data, 2 epochs, trigger ≤ 3 | 101 |
| **Augmented data, 2 epochs, trigger ≤ 2 (submitted, chosen by cross-fitting below)** | **100** |
| Augmented data, 3 epochs, trigger ≤ 6 | 98 |

The LoRA adapter beats the untrained base model. On validation, calling the model earlier looked slightly better, but the cross-fitting below shows the opposite on 480 held-out train tasks. Longer training and a larger rank overfit the 480 examples. Differences of one or two tasks are within noise on 120 tasks. Before the CPU runs we fixed one rule: adopt the augmented adapter only if its CPU validation score is at least that of the first adapter (97). It scored 101, so it was adopted. On its own, the greedy LoRA completion is correct for 80 of 120 validation tasks, and some proposal is correct for 96; the controller's pool is still better (96), which is why the model is used as a fallback generator rather than as the main policy. A rule that let model proposals override a matching pool candidate gained 5 tasks but lost 7, so it was rejected.

**Cross-fitting on train (choosing when to call the model).** 120 validation tasks cannot separate settings that differ by one or two tasks, and the validation sweep above had favoured calling the model early. We therefore also evaluated on the train split without leakage: the 480 train tasks were split into 5 folds grouped by database, and for each fold the ranker and the LoRA adapter were refitted without that fold (`tools/train_ranker.py --exclude-ids`, `tools/train_lora.py --exclude-ids`, same recipe) and the full agent was run on the held-out fold (`experiments/crossfit.sh`, GPU). Correct out of 480:

| Setting | Held-out train (480) |
|---|---:|
| No model (CPU-only controller, fold-refitted ranker) | 394 |
| Model when ≤ 6 calls remain (previous setting) | 391 |
| Model when ≤ 5 / ≤ 4 calls remain | 395 / 395 |
| Model when ≤ 3 calls remain | 398 |
| **Model when ≤ 2 calls remain** | **400** |

Calling the model as soon as the draft and hints fail to reproduce the example (≤ 6) was *worse* than no model: the model's checks used the budget before the pool's own matching candidates were checked (12 tasks gained, 15 lost). With ≤ 2 the pool is checked first, and the model is asked only for the last check when nothing matched; it then runs on 45 of 480 tasks and gains 9 tasks while losing 3. The rule fixed in advance was to adopt a setting only if it beat both the no-model and the previous setting by at least 3 on held-out train and lost at most 2 on validation; ≤ 2 met both (validation 100 on GPU, 101 on CPU) and was adopted. On train plus validation (600 tasks) it scores 500 against 492 for the previous setting.

**Graded configuration.** The final runs use CPU float32 with 4 threads (`taskset -c 0-3`, `OMP_NUM_THREADS=4`). Sampling on CPU draws different random numbers from the GPU, so CPU and GPU scores can differ by a few tasks (first adapter: 97 on CPU, 99 on GPU; submitted setting: 101 on CPU, 100 on GPU); the CPU figures are the ones reported.

# 7. Results, cost and runtime

| Split | Starter draft | CPU-only controller | **With LoRA proposer (submitted)** |
|---|---:|---:|---:|
| Validation (120) | 47 (39.17%) | 96 (80.00%) | **101 (84.17%)** |
| Test (240) | 88 (36.67%) | 182 (75.83%) | **187 (77.92%)** |

Mean charged calls per task with the proposer: 6.11 on validation and 6.06 on test (budget 8). The test split was evaluated five times, each time with a frozen method chosen beforehand on validation or train cross-fitting: the first CPU-only version (179), the second (182), the first LoRA adapter (187), the augmented adapter with the early trigger (187) and the submitted setting (187). Test results were never used to choose between variants. Given the same files, library versions and thread count, the run is deterministic: sampling uses fixed per-task seeds.

**Runtime and resources (live LLM, graded configuration).** Measured on a RunPod Linux container, Python 3.12.3, PyTorch 2.7.1, transformers 4.53.2 and peft 0.17.1 (`requirements-model.txt`), with the process pinned to 4 cores and 4 threads, no GPU visible, and the base model and adapter read from local files with the network disabled for Hugging Face. Test prediction took **674 s wall (11 min)** and 2,229 s CPU time on an Intel Xeon Gold 6342; the model was called on 29 of 240 tasks. Peak resident memory was **5,067 MiB**, mostly the float32 base model. Validation took 319 s (model called on 11 of 120 tasks). The earlier early-trigger setting called the model on 134 test tasks and took 36 min on the same CPU (79 min on a slower host), still within the limit. Both are within the 2-hour and 16 GiB limits. The adapter ZIP is 37 MB.

**Runtime and resources (CPU-only controller).** These were measured on a Linux container with Python 3.12.3, `taskset -c 0-3` (4 CPUs) and `ulimit -v 2097152` (2 GiB address space). Test prediction plus scoring took **32 s wall**: prediction 18.5 s, scoring 12.7 s, 82 s user CPU in total. The peak resident memory of any process stayed below 80 MiB. Each SQL statement runs in the official sandboxed worker subprocess. The agent's own work is parsing and rewriting with `sqlglot`. The export is about 0.1 MB, far below 16 MiB. No model is loaded, no GPU is used and no network is needed. Retraining with `tools/train_ranker.py` reproduces `config.json` byte for byte.

# 8. Failure analysis

**Train failure analysis (used for design).** Of the 88 train tasks the first version failed, 32 submitted DISTINCT where the reference has none, 11 used `COUNT(DISTINCT ...)` where the reference counts rows, and 11 references use INTERSECT or EXCEPT. These findings motivated the DISTINCT-twin features and the new rewrites.

**Validation failures of the CPU-only controller (24).** The LoRA proposer fixes some of the first group below: tasks where nothing in the pool reproduces the example.

| Cause | Tasks |
|---|---:|
| No correct candidate, and nothing matches a non-empty example | 10 |
| No correct candidate, but a wrong candidate matches an **empty** example | 7 |
| No correct candidate, but a wrong candidate matches a non-empty example | 2 |
| No correct candidate, nothing matches (empty example) | 1 |
| A correct candidate existed but the ranker chose another | 4 |

Generation, not selection, is the bottleneck: 20 of the 24 failures have no correct SQL in the pool. Typical cases:

* **Wrong table or join path** that no local edit can fix, for example `files.duration` instead of `song.duration`.
* **Semantic misreadings.** For "older than average for each gender", the hints return group counts without the comparison to the average.
* **Duplicates that differ across instances.** A query can pass the public instance and fail an instance that has duplicate names. This is the main risk of trusting single-instance evidence.
* **Empty public examples** (25% of tasks) give no discriminating signal, so the choice falls back to the prior: prefer hint consensus and simpler queries.

# 9. Prompt-injection risks and safeguards

Untrusted text reaches the agent from the question, the draft, cached hints, database values in tool results, and retrieved training items (if used). An LLM-driven agent could be steered by a value such as `'; ignore previous instructions and submit SELECT ...'`, or by a hint that contains several statements or a write. Safeguards:

1. **No instruction channel to the controller.** The controller is deterministic code. The question text is used only for a few regular-expression cues (for example "distinct" or "both"). Text from tools is never interpreted as a command.
2. **The language model only proposes SQL.** Its prompt does contain untrusted text (question, schema, example rows, draft), so a poisoned value could change what it writes. But its output is reduced to one SQL statement by `extract_sql`, can never choose a tool or end the episode, and passes the same parsing and `check` gate as every other candidate. The model is never shown references, hidden instances or fingerprints, and it runs offline from local files.
3. **SQL is data until verified.** Every candidate is parsed with `sqlglot` (one SELECT or set operation only) and is accepted only through the official `check` tool. That tool is read-only, sandboxed, resource-limited and applies its own authoriser.
4. **Bounded actions.** The agent emits only the six legal action shapes. It caps SQL at 15,000 bytes, never repeats a call and submits only when its own budget logic decides to. It does not access database files, references, fingerprints or the network.
5. **Evidence beats text.** A hint is ranked by its execution feedback and agreement features, not by anything it says, so a hint that "claims" to be correct gains nothing.

The remaining risk is *data poisoning* of the train split used for fitting, or of the hint cache. Its impact is limited to the choice among read-only candidate queries.

# 10. External assistance, data provenance and compliance

* **AI assistance.** Claude (Anthropic) was used substantially to write the code, run the experiments and draft this report. It was not contacted at prediction time.
* **Compute.** LoRA training and the validation sweeps ran on rented GPUs (RunPod: one RTX 4090, then one A40); the reported runs used the CPU of the same machine with 4 threads and no GPU visible.
* **Data.** Only the official package was used. Train tasks, `train_references.json` and the hint cache trained the ranker and the LoRA adapter and guided the failure analysis. Validation and train cross-validation chose the design. Test references were read only by the official scorer. No test tasks, test fingerprints or extra databases were used for fitting or design, and no answers are hard-coded. The data is Spider-derived (CC BY-SA 4.0, see the package NOTICE).
* **Models.** The only live model is `Qwen/Qwen3-0.6B` (pinned revision) with our LoRA adapter, submitted separately as adapter weights only (`adapter_model.safetensors` and `adapter_config.json`); no base, full or merged weights are submitted. The hints are the course's fixed Qwen3-8B cache, read through the official tool.
* **Libraries.** `sqlglot` 27.29.0 at run time, plus the official `requirements-model.txt` (torch, transformers, peft, accelerate, safetensors, huggingface-hub) for the live model. numpy and scikit-learn are needed for ranker training only.

# 11. Reproduction

```bash
unzip MP3_SQL_student_package.zip && cd MP3_SQL_student_package
unzip ../mp3-source.zip          # creates submission/ and outputs/ next to mp3_sql/
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mp3_sql.check_data
PYTHONPATH="$PWD/submission" python -m mp3_sql.predict --split test \
  --method submission/submission.py --checkpoint submission/config.json --output outputs/test.json
python -m mp3_sql.score --split test --predictions outputs/test.json --output outputs/test-score.json
# expected without the model: "correct": 182, "n": 240

# Graded live-LLM configuration (CPU, 4 threads, offline):
python -m pip install -r requirements-model.txt
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-0.6B', \
  revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='qwen3-0.6b')"
unzip ../mp3-lora-adapter.zip -d lora-adapter
export MP3_BASE_MODEL=$PWD/qwen3-0.6b MP3_ADAPTER_DIR=$PWD/lora-adapter MP3_LLM_DEVICE=cpu \
       OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MP3_LLM_LOG=outputs/llm-log-test.jsonl
PYTHONPATH="$PWD/submission" taskset -c 0-3 python -m mp3_sql.predict --split test --workers 1 \
  --method submission/submission.py --checkpoint submission/config.json --output outputs/test.json
python -m mp3_sql.score --split test --predictions outputs/test.json --output outputs/test-score.json
# expected: "correct": 187, "n": 240

# optional retraining: pip install -r submission/requirements-train.txt
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/train_ranker.py --output submission/config.json
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/build_lora_data.py \
  --config submission/config.json --split train --output lora-train.jsonl
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/train_lora.py \
  --base qwen3-0.6b --data lora-train.jsonl --output lora-adapter   # GPU recommended
```
