# DASE7506 MP1 - Small Language Model Challenge

Student ID: **3036797441**. This repository contains the supplied GPT baseline,
an independently maintained exponential moving average (EMA) of training weights,
paired experiments, and a reproducible submission package.

Verified full-test CPU FP32 BPB: **2.1012651958438786**. Validation selected the
original seed-17 raw baseline: EMA was slightly worse in both paired seeds.
The [frozen code](https://github.com/bob0612/Advanced-ML/tree/0659d1eb28752ced329b3c94ac460b2628c4b527)
and [matching model bundle](https://raw.githubusercontent.com/bob0612/Advanced-ML/0659d1eb28752ced329b3c94ac460b2628c4b527/MP1_student_starter/submission/model-bundle.zip)
are publicly downloadable. See [delivery status](MP1_student_starter/submission/DELIVERY_STATUS.json)
for course submission and later peer-review status.

See [reproduction instructions](MP1_student_starter/REPRODUCE.md),
the [report](MP1_student_starter/report/REPORT.pdf), and the
[Chinese report source](MP1_student_starter/report/REPORT.md).
Final scores, checkpoint hashes and the release manifest are recorded under
`MP1_student_starter/submission/`.

The fixed evaluator, tokenization, dataset, original model architecture and baseline
configuration are preserved. EMA changes the exported weights; it does not ensemble
predictions or modify the training model. The report distinguishes any candidate
mechanism from the model actually selected for submission, including negative results.

## AI assistance and reused work

Substantive assistance from OpenAI Codex was used for project planning, code and test
implementation, running and checking experiments, interpreting results, drafting and
typesetting the report, preparing reproduction instructions, and packaging/publishing.
The model, optimizer recipe, tokenizer, data loader, evaluator and original contract
tests originate from the supplied course starter. EMA is a standard parameter averaging
method, not a novel algorithm. Its implementation and tests are the coursework addition.
The report states the evidence and limitations; no performance improvement is assumed.

The student should understand and be able to explain the submitted code and evidence;
automated assistance does not establish the student's personal understanding.

## Data attribution

WikiText-2: Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher,
*Pointer Sentinel Mixture Models*, https://arxiv.org/abs/1609.07843.
Text is by Wikipedia contributors. The upstream dataset identifies
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html).
See the preserved [data attribution](MP1_student_starter/code/README.md#6-data-attribution)
for the supplied dataset revision. These notices do not grant a new license to the
surrounding classroom code.
