"""Ablations, uncertainty and error analysis on validation (all groups + test-like hard subset).

Writes runs/analysis/validation_analysis.json and figures. Uses train for fitting and
validation for evaluation only; test references are not read.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_inputs  # noqa: E402
from features import FEATURE_NAMES, bank_features  # noqa: E402
from model import EmbeddingVerifier, HAND_COLS, choose, make_ranker, stack_features  # noqa: E402
from train import labels, oof_verifier_scores, VERIFIER_C  # noqa: E402

OUT = ROOT/'runs'/'analysis'
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    'answer_form': ['ans_none', 'ans_int', 'ans_neg', 'ans_zero', 'ans_log_mag', 'ans_in_question',
                    'ans_is_question_num_count', 'final_num_matches_ans'],
    'consensus': ['vote_share', 'vote_is_max', 'vote_unique_max', 'n_distinct', 'n_none_group', 'vote_rank'],
    'surface': ['log_chars', 'log_lines', 'n_boxed', 'has_hash', 'n_steps', 'hedge_count', 'n_numbers',
                'frac_numbers_from_q', 'rel_len'],
    'arithmetic': ['arith_checked', 'arith_wrong', 'arith_any_wrong', 'arith_wrong_frac'],
}

t0 = time.perf_counter()
data = {}
for split in ('train', 'validation'):
    bank, arrays = load_inputs(split)
    X, answers = bank_features(bank, arrays['embeddings'])
    y = labels(answers, split)
    hard = np.array([len({a for a in r if a is not None}) >= 2 for r in answers])
    data[split] = dict(bank=bank, emb=arrays['embeddings'], X=X, answers=answers, y=y, hard=hard)
tr, va = data['train'], data['validation']
feat_seconds = time.perf_counter() - t0

oof = oof_verifier_scores(tr['emb'], tr['y'])
verifier = EmbeddingVerifier().fit(tr['emb'], tr['y'], C=VERIFIER_C)
p_va = verifier.predict_proba(va['emb'])


def evaluate(scores, d=va):
    pick = np.array(choose(scores))
    hit = d['y'][np.arange(len(pick)), pick]
    return {'all': round(100*hit.mean(), 2), 'hard': round(100*hit[d['hard']].mean(), 2),
            'n_all': int(len(hit)), 'n_hard': int(d['hard'].sum())}, pick, hit


def ranker_scores(hand_cols, use_emb, seeds=(0, 1, 2)):
    Str = stack_features(tr['X'], oof if use_emb else None, tr['answers'], hand_cols)
    Sva = stack_features(va['X'], p_va if use_emb else None, va['answers'], hand_cols)
    tick = time.perf_counter()
    models = [make_ranker(s).fit(Str.reshape(-1, Str.shape[-1]), tr['y'].ravel()) for s in seeds]
    fit_s = time.perf_counter() - tick
    tick = time.perf_counter()
    s = np.mean([m.predict_proba(Sva.reshape(-1, Sva.shape[-1]))[:, 1] for m in models], 0).reshape(-1, 4)
    return s, fit_s, time.perf_counter() - tick


results, picks = {}, {}
n = len(va['y'])
results['first_candidate'], picks['first'], _ = evaluate(np.tile([1., 0, 0, 0], (n, 1)))
results['majority_vote'], picks['majority'], _ = evaluate(va['X'][:, :, FEATURE_NAMES.index('vote_share')])
results['oracle'], _, _ = evaluate(va['y'])
results['emb_lr_only'], picks['emb'], _ = evaluate(p_va)

s_full, fit_full, inf_full = ranker_scores(HAND_COLS, True)
results['full'], picks['full'], hit_full = evaluate(s_full)
results['full']['ranker_fit_seconds'] = round(fit_full, 2)
results['full']['ranker_infer_seconds'] = round(inf_full, 3)
s_one, _, _ = ranker_scores(HAND_COLS, True, seeds=(0,))
results['full_single_ranker'], _, _ = evaluate(s_one)
results['hand_only_no_emb'], picks['hand'], _ = evaluate(ranker_scores(HAND_COLS, False)[0])
for g, cols in GROUPS.items():
    keep = [c for c in HAND_COLS if c not in cols]
    results[f'full_minus_{g}'], _, _ = evaluate(ranker_scores(keep, True)[0])
    results[f'only_{g}'], _, _ = evaluate(ranker_scores(cols, False)[0])
results['emb_plus_surface_only'], _, _ = evaluate(ranker_scores(GROUPS['surface'], True)[0])

# --- uncertainty: confidence = top ranker probability; margin over best competing answer
order = np.argsort(-s_full, 1)
top = s_full[np.arange(n), order[:, 0]]
comp = []
for g in range(n):
    a = va['answers'][g][order[g, 0]]
    others = [s_full[g, j] for j in range(4) if not (a is not None and va['answers'][g][j] == a)]
    comp.append(top[g] - (max(others) if others else 0.0))
comp = np.array(comp)
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
calib = []
for lo, hi in zip(bins, bins[1:]):
    m = va['hard'] & (comp >= lo) & (comp < hi)
    if m.any():
        calib.append({'margin_bin': f'[{lo},{min(hi,1)})', 'n': int(m.sum()),
                      'accuracy': round(100*hit_full[m].mean(), 2)})
results['calibration_hard_by_margin'] = calib
# selective accuracy: keep the most confident fraction of hard questions
cov = []
hard_idx = np.where(va['hard'])[0]
srt = hard_idx[np.argsort(-comp[hard_idx])]
for frac in (0.25, 0.5, 0.75, 1.0):
    k = int(round(frac*len(srt)))
    cov.append({'coverage': frac, 'accuracy': round(100*hit_full[srt[:k]].mean(), 2)})
results['selective_accuracy_hard'] = cov

# --- agreement between majority vote, embedding LR and the full model (hard subset)
ans = lambda pk: [va['answers'][g][pk[g]] for g in range(n)]  # noqa: E731
a_full, a_maj, a_emb = ans(picks['full']), ans(picks['majority']), ans(picks['emb'])
agree = np.array([a_full[g] == a_maj[g] for g in range(n)])
agree3 = np.array([a_full[g] == a_maj[g] == a_emb[g] for g in range(n)])
H = va['hard']
results['agreement_hard'] = {
    'full_vs_majority_agree_frac': round(float(agree[H].mean()), 3),
    'acc_when_agree': round(100*hit_full[H & agree].mean(), 2),
    'acc_when_disagree': round(100*hit_full[H & ~agree].mean(), 2),
    'n_disagree': int((H & ~agree).sum()),
    'all_three_agree_frac': round(float(agree3[H].mean()), 3),
    'acc_all_three_agree': round(100*hit_full[H & agree3].mean(), 2),
    'acc_not_all_agree': round(100*hit_full[H & ~agree3].mean(), 2),
}

# --- error analysis: hard-subset failures by vote pattern of the gold answer
pattern = {}
for g in hard_idx:
    k = int(va['y'][g].sum())
    pattern.setdefault(k, []).append(hit_full[g])
results['hard_acc_by_num_correct'] = {k: {'n': len(v), 'accuracy': round(100*np.mean(v), 2)}
                                     for k, v in sorted(pattern.items())}
# minority-correct wins: gold answer had fewer votes than the majority answer
X = va['X']; vs = FEATURE_NAMES.index('vote_share')
minority = [g for g in hard_idx if (X[g, :, vs]*va['y'][g]).max() < X[g, :, vs].max()]
results['minority_correct'] = {'n': len(minority),
                               'full_acc': round(100*np.mean([hit_full[g] for g in minority]), 2),
                               'majority_acc': 0.0}

# --- what features separate correct/incorrect candidates on hard validation groups
diag = {}
for name in ('arith_any_wrong', 'ans_int', 'ans_none', 'hedge_count', 'log_chars'):
    c = FEATURE_NAMES.index(name)
    xh, yh = X[H][:, :, c].ravel(), va['y'][H].ravel()
    diag[name] = {'mean_correct': round(float(xh[yh == 1].mean()), 3),
                  'mean_wrong': round(float(xh[yh == 0].mean()), 3)}
results['feature_means_hard'] = diag
results['feature_seconds_train_plus_val'] = round(feat_seconds, 2)

# ranker feature importance by permutation on validation-hard (single seed model)
Str = stack_features(tr['X'], oof, tr['answers'])
Sva = stack_features(va['X'], p_va, va['answers'])
m0 = make_ranker(0).fit(Str.reshape(-1, Str.shape[-1]), tr['y'].ravel())
names = HAND_COLS + ['emb_logit', 'emb_rel_max', 'emb_rel_mean', 'emb_answer_sum', 'emb_margin_other']
base = evaluate(m0.predict_proba(Sva.reshape(-1, Sva.shape[-1]))[:, 1].reshape(-1, 4))[0]['hard']
rng = np.random.default_rng(0)
imp = {}
for j, nm in enumerate(names):
    drops = []
    for _ in range(3):
        P = Sva.copy()
        perm = rng.permutation(n)
        P[:, :, j] = P[perm, :, j]  # shuffle across groups, keep within-group structure
        drops.append(base - evaluate(m0.predict_proba(P.reshape(-1, P.shape[-1]))[:, 1].reshape(-1, 4))[0]['hard'])
    imp[nm] = round(float(np.mean(drops)), 2)
results['permutation_importance_hard'] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:12])

(OUT/'validation_analysis.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
