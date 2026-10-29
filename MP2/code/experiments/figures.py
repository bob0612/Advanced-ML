"""Report figures from runs/analysis/validation_analysis.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT/'runs/analysis/validation_analysis.json').read_text())
OUT = ROOT.parent/'report'/'figures'
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})

rows = [('First candidate', 'first_candidate'), ('Majority vote', 'majority_vote'),
        ('Surface features only', 'only_surface'), ('Arithmetic only', 'only_arithmetic'),
        ('Answer form only', 'only_answer_form'), ('Hand features (no emb.)', 'hand_only_no_emb'),
        ('Embedding LR only', 'emb_lr_only'), ('Full − answer form', 'full_minus_answer_form'),
        ('Full − surface', 'full_minus_surface'), ('Full − consensus', 'full_minus_consensus'),
        ('Full − arithmetic', 'full_minus_arithmetic'), ('Full model', 'full')]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
vals = [R[k]['hard'] for _, k in rows]
cols = [ORANGE if k == 'full' else BLUE for _, k in rows]
ax.barh(range(len(rows)), vals, color=cols, height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.6, i, f'{v:.1f}', va='center', color=INK, fontsize=8)
ax.set_yticks(range(len(rows)), [r for r, _ in rows])
ax.set_xlim(50, 90); ax.set_xlabel('Selection accuracy on validation-hard (%, n=355)')
ax.grid(axis='x', color='#e6e5e1', linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT/'ablation.pdf')

cal = R['calibration_hard_by_margin']
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.bar(range(len(cal)), [c['accuracy'] for c in cal], color=BLUE, width=0.62)
for i, c in enumerate(cal):
    ax.text(i, c['accuracy'] + 1.5, f"{c['accuracy']:.0f}%\nn={c['n']}", ha='center', fontsize=7, color=INK)
ax.set_xticks(range(len(cal)), [c['margin_bin'] for c in cal], fontsize=7)
ax.set_ylim(0, 115); ax.set_ylabel('Accuracy (%)'); ax.set_xlabel('Score margin over best competing answer')
ax.grid(axis='y', color='#e6e5e1', linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT/'calibration.pdf')
print('ok')
