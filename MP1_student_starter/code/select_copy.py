"""Select scalar copy settings on validation only; never reads test predictions."""
import argparse
import json
from pathlib import Path
import time
import torch
from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score


# Declared search space; all trials, including the identity, are recorded.
COPY_CANDIDATES = [
    [0., 0., 0.],
    [0., .05, .1],
    [.02, .05, .15],
    [0., .1, .2],
    [.02, .1, .3],
    [.05, .2, .4],
    [0., .1, .2, .4],
    [0., .05, .1, .2],
]
EXTENDED_CANDIDATES = COPY_CANDIDATES + [
    [.05, .2, .4, .6, .7, .8],
    [.05, .15, .3, .5, .65, .8],
    [.1, .25, .45, .6],
    [.1, .3, .5],
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--space', choices=['initial', 'extended'], default='initial')
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use a new output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device, precision = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Wrong protocol.')
    data = load_data()
    model, implementation_sha = make_model('student', checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    trials = []
    for weights in COPY_CANDIDATES if args.space == 'initial' else EXTENDED_CANDIDATES:
        model.copy_weights = tuple(weights)
        result = score(model, *data['validation'], device, precision)
        result.pop('window_nll_nats')
        trials.append({'copy_weights': weights, **result})
        print(json.dumps(trials[-1]), flush=True)
    best = min(trials, key=lambda row: row['bpb'])
    selected = dict(checkpoint)
    selected.update(implementation='student', config=checkpoint['config'] | {'copy_weights': best['copy_weights']},
                    ancestry={'neural_checkpoint_sha256': sha(args.checkpoint),
                              'neural_checkpoint': str(args.checkpoint),
                              'source_ancestry': checkpoint.get('ancestry', {
                                  'operation': 'direct_training', 'seed': checkpoint['seed'],
                                  'train_tokens': checkpoint['train_tokens']}),
                              'additional_training_targets': 0})
    target = args.output_dir / 'checkpoint.pt'
    torch.save(selected, target)
    output = {'selection_split': 'validation', 'trials': trials, 'selected': best,
              'source_checkpoint': str(args.checkpoint), 'source_sha256': sha(args.checkpoint),
              'checkpoint_sha256': sha(target), 'implementation_sha256': implementation_sha,
              'script_sha256': sha(Path(__file__)), 'train_tokens': checkpoint['train_tokens'],
              'process_seconds': time.perf_counter() - started}
    (args.output_dir / 'selection.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
