"""Average snapshots from one training run without adding training targets."""
import argparse
import json
from pathlib import Path
import time
import torch
from common import sha


def average(paths):
    if len(paths) < 2:
        raise ValueError('Provide at least two snapshots from one run.')
    checkpoints = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    first = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        for key in ('protocol', 'implementation', 'config', 'seed'):
            if checkpoint[key] != first[key]:
                raise ValueError(f'Snapshot {key} mismatch.')
    if len({Path(p).resolve().parent for p in paths}) != 1:
        raise ValueError('Snapshots must come from the same run directory.')
    result = dict(max(checkpoints, key=lambda c: c['train_tokens']))
    result['model'] = {}
    for name, value in first['model'].items():
        if value.is_floating_point():
            result['model'][name] = torch.stack([c['model'][name] for c in checkpoints]).mean(0)
        else:
            if any(not torch.equal(value, c['model'][name]) for c in checkpoints[1:]):
                raise ValueError(f'Non-floating state differs: {name}')
            result['model'][name] = value
    result['ancestry'] = {'operation': 'uniform_snapshot_average', 'additional_training_targets': 0,
                          'sources': [{'path': str(p), 'sha256': sha(p), 'train_tokens': c['train_tokens']}
                                      for p, c in zip(paths, checkpoints)]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists.')
    started = time.perf_counter()
    result = average(args.checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    metrics = {'checkpoint_sha256': sha(args.output), 'ancestry': result['ancestry'],
               'train_tokens': result['train_tokens'], 'process_seconds': time.perf_counter()-started}
    args.output.with_suffix('.average.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
