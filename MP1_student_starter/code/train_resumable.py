"""CPU training in resumable chunks, preserving optimizer and both RNG states.

--steps is the complete cosine schedule, --chunk-steps is only a process boundary.
Chunk boundaries do not restart the optimizer, schedule, sampler or dropout RNG.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import ROOT, PROTOCOL, load_data, make_model, setup, sha
from evaluate import score


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_training(model, optimizer, rng):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'sampler_rng': rng.get_state()}


def restore_training(state, model, optimizer, rng):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    rng.set_state(state['sampler_rng'])


def record_interrupted_work(run_dir, recovery_step, batch_size, prior_seconds):
    """Keep confirmed replay costs before the next update replaces progress.json."""
    progress_path = run_dir/'progress.json'
    if not progress_path.exists():
        return
    progress = json.loads(progress_path.read_text())
    if progress['step'] <= recovery_step:
        return
    path = run_dir/'interrupted_work.json'
    events = json.loads(path.read_text()) if path.exists() else []
    event_id = f'{recovery_step}:{sha(progress_path)}'
    if any(event['event_id'] == event_id for event in events):
        return
    seconds = progress.get('train_seconds')
    events.append({'event_id': event_id, 'recovery_step': recovery_step,
                   'last_confirmed_step': progress['step'],
                   'discarded_completed_targets': (progress['step']-recovery_step)*batch_size*256,
                   'discarded_training_seconds': None if seconds is None else max(0., seconds-prior_seconds),
                   'note': 'Confirmed completed updates only; unrecorded partial work may be additional.'})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(events, indent=2)+'\n')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/wide.json')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--chunk-steps', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=600)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.steps, args.chunk_steps, args.batch_size) < 1 or args.eval_every < 0:
        p.error('Step/batch counts must be positive; evaluation interval nonnegative.')
    process_started = time.perf_counter()
    device, precision = setup('cpu', 'fp32', args.threads)
    config = json.loads(args.config.read_text())
    recipe = {'config': config, 'steps': args.steps, 'batch_size': args.batch_size,
              'seed': args.seed, 'learning_rate': .001, 'weight_decay': .1,
              'threads': args.threads, 'precision': precision}
    resume_path = args.run_dir/'training_state.pt'
    if args.resume and not resume_path.exists():
        p.error('Resume state does not exist.')
    if not args.resume and args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new run directory, or --resume with its complete state.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model, implementation_sha = make_model('student', config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    rng = torch.Generator().manual_seed(args.seed)
    state = None
    start_step, prior_seconds, history, validations, chunks = 0, 0., [], [], []
    if args.resume:
        state = torch.load(resume_path, map_location='cpu', weights_only=True)
        if state['recipe'] != recipe:
            raise ValueError('Resume recipe differs: changing it would break exact continuation.')
        restore_training(state, model, optimizer, rng)
        start_step, prior_seconds = state['step'], state['train_seconds']
        history, validations, chunks = state['history'], state['validation_history'], state['chunks']
        record_interrupted_work(args.run_dir, start_step, args.batch_size, prior_seconds)
    if start_step >= args.steps:
        p.error('This run is already complete.')
    data = load_data()
    tokens = data['train'][0]
    end_step = min(args.steps, start_step + args.chunk_steps)
    preparation_seconds = time.perf_counter() - process_started
    training_started = time.perf_counter()
    validation_seconds = 0.
    for step in range(start_step, end_step):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng)
        batch = tokens[starts[:, None] + torch.arange(257)]
        rate = .001 * min(1., (step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == end_step:
            row = {'step': step+1, 'loss': loss.item(),
                   'seconds': prior_seconds + time.perf_counter()-training_started-validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        # Persist exact completed-update accounting. It is not a recovery state;
        # complete optimizer/RNG recovery is written atomically at chunk end.
        progress = {'step': step+1, 'train_tokens': (step+1)*args.batch_size*256,
                    'last_recoverable_step': start_step,
                    'train_seconds': prior_seconds + time.perf_counter()-training_started-validation_seconds}
        temporary = args.run_dir/'progress.tmp'
        temporary.write_text(json.dumps(progress)+'\n')
        os.replace(temporary, args.run_dir/'progress.json')
        if (args.eval_every and (step+1) % args.eval_every == 0) or step+1 == args.steps:
            validation = score(model, *data['validation'], device, 'fp32')
            validation.pop('window_nll_nats')
            validation_seconds += validation['seconds']
            validations.append({'step': step+1, **validation})
            print(json.dumps({'validation': validations[-1]}), flush=True)
    train_seconds = prior_seconds + time.perf_counter()-training_started-validation_seconds
    checkpoint = {'protocol': PROTOCOL, 'implementation': 'student', 'config': config,
                  'model': model.state_dict(), 'seed': args.seed,
                  'train_tokens': end_step*args.batch_size*256, 'step': end_step,
                  'planned_steps': args.steps, 'learning_rate': .001, 'weight_decay': .1}
    path = args.run_dir/f'step-{end_step:05d}.pt'
    atomic_save(checkpoint, path)
    atomic_save(checkpoint, args.run_dir/'checkpoint.pt')
    chunk = {'start_step': start_step, 'end_step': end_step,
             'executed_training_targets': (end_step-start_step)*args.batch_size*256,
             'training_seconds': train_seconds-prior_seconds,
             'validation_seconds': validation_seconds, 'preparation_seconds': preparation_seconds,
             'process_seconds_before_resume_save': time.perf_counter()-process_started}
    chunks.append(chunk)
    recovery = capture_training(model, optimizer, rng) | {
        'recipe': recipe, 'step': end_step, 'train_seconds': train_seconds,
        'history': history, 'validation_history': validations, 'chunks': chunks}
    atomic_save(recovery, resume_path)
    (args.run_dir/'progress.json').write_text(json.dumps({
        'step': end_step, 'train_tokens': end_step*args.batch_size*256,
        'last_recoverable_step': end_step, 'train_seconds': train_seconds})+'\n')
    result = {'protocol': PROTOCOL, 'implementation': 'student', 'config': config,
              'seed': args.seed, 'precision': precision, 'threads': args.threads,
              'parameters': sum(p.numel() for p in model.parameters()),
              'train_tokens': end_step*args.batch_size*256, 'train_seconds': train_seconds,
              'step': end_step, 'planned_steps': args.steps, 'complete': end_step == args.steps,
              'history': history, 'validation_history': validations, 'chunks': chunks,
              'validation': validations[-1] if validations else None,
              'checkpoint_sha256': sha(args.run_dir/'checkpoint.pt'),
              'implementation_sha256': implementation_sha, 'trainer_sha256': sha(Path(__file__)),
              'torch_version': str(torch.__version__), 'device': 'cpu'}
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'chunk_complete': chunk, 'step': end_step, 'complete': result['complete']}), flush=True)


if __name__ == '__main__':
    main()
