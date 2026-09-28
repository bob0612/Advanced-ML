"""Behavioral checks for independent weight averaging and trainer integration."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from common import PROTOCOL
from model import GPT
import train
import test_contract


class EMATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.assertIsNotNone(importlib.util.find_spec('ema'), 'EMA has not been implemented')
        from ema import EMA
        self.EMA = EMA

    @staticmethod
    def scalar(value):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(value)
        return model

    def test_ema_arithmetic(self):
        model = self.scalar(2.)
        ema = self.EMA(model, .9)
        with torch.no_grad():
            model.weight.fill_(4.)
        ema.update(model)
        torch.testing.assert_close(ema.state_dict()['weight'], torch.tensor([[2.2]]))

    def test_ema_does_not_mutate_training_model(self):
        model = self.scalar(2.)
        ema = self.EMA(model, .9)
        with torch.no_grad():
            model.weight.fill_(4.)
        torch.testing.assert_close(ema.state_dict()['weight'], torch.tensor([[2.]]))
        ema.update(model)
        torch.testing.assert_close(model.weight, torch.tensor([[4.]]), rtol=0, atol=0)
        exported = ema.state_dict()
        exported['weight'].zero_()
        torch.testing.assert_close(ema.state_dict()['weight'], torch.tensor([[2.2]]))
        self.assertFalse(ema.state_dict()['weight'].requires_grad)

    def test_ema_copies_integer_buffers_and_averages_float_buffers(self):
        model = self.scalar(2.)
        model.register_buffer('counter', torch.tensor(2, dtype=torch.int64))
        model.register_buffer('running_value', torch.tensor(2.))
        ema = self.EMA(model, .9)
        model.counter.fill_(4)
        model.running_value.fill_(4.)
        ema.update(model)
        state = ema.state_dict()
        self.assertEqual(state['counter'].dtype, torch.int64)
        self.assertEqual(state['counter'].item(), 4)
        torch.testing.assert_close(state['running_value'], torch.tensor(2.2))

    def test_invalid_decay_is_rejected(self):
        for decay in (0., 1., -.1, 1.1, float('nan'), float('inf')):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                self.EMA(self.scalar(2.), decay)

    def test_ema_initialization_uses_designated_updated_weights(self):
        model = self.scalar(1.)
        ema = train.update_ema(None, model, 1, decay=.9, start=2)
        self.assertIsNone(ema)
        with torch.no_grad():
            model.weight.fill_(2.)
        ema = train.update_ema(ema, model, 2, decay=.9, start=2)
        torch.testing.assert_close(ema.state_dict()['weight'], torch.tensor([[2.]]))
        with torch.no_grad():
            model.weight.fill_(4.)
        ema = train.update_ema(ema, model, 3, decay=.9, start=2)
        torch.testing.assert_close(ema.state_dict()['weight'], torch.tensor([[2.2]]))

    def test_cli_bounds_and_disabled_defaults(self):
        defaults = train.parse_args(['--steps', '10'])
        self.assertEqual(defaults.ema_decay, 0.)
        self.assertEqual(defaults.ema_start, 600)
        valid = train.parse_args(['--steps', '10', '--ema-decay', '.99', '--ema-start', '5'])
        self.assertEqual((valid.ema_decay, valid.ema_start), (.99, 5))
        invalid = [
            ['--ema-decay', '-.1'], ['--ema-decay', '1'],
            ['--ema-decay', 'nan'], ['--ema-decay', 'inf'],
            ['--steps', '10', '--ema-decay', '.99', '--ema-start', '0'],
            ['--steps', '10', '--ema-decay', '.99', '--ema-start', '10'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_ema_checkpoint_round_trip_and_tied_weights(self):
        torch.manual_seed(17)
        config = dict(vocab=2048, width=16, heads=2, depth=1, context=256)
        model = GPT(config).eval()
        ema = self.EMA(model, .9)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.01)
        ema.update(model)
        state = ema.state_dict()
        torch.testing.assert_close(state['head.weight'], state['token.weight'], atol=0, rtol=0)
        model.load_state_dict(state)
        ids = torch.tensor([[0, 7, 23, 81]])
        with torch.no_grad():
            expected = model.predict_log_probs(ids)
        stream = io.BytesIO()
        torch.save({'protocol': PROTOCOL, 'implementation': 'model', 'config': config,
                    'model': state, 'seed': 17, 'train_tokens': 8192}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        loaded = GPT(checkpoint['config']).eval()
        loaded.load_state_dict(checkpoint['model'])
        self.assertIs(loaded.token.weight, loaded.head.weight)
        with torch.no_grad():
            actual = loaded.predict_log_probs(ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(1, 4), atol=1e-6, rtol=1e-6)

    def test_ema_tracking_preserves_raw_training_trajectory(self):
        # Replace only slow corpus I/O; use the real GPT, trainer, optimizer and scorer.
        data = {'train': (torch.arange(600) % 2048, 600),
                'validation': (torch.arange(18), 18)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'small.json'
            config.write_text(json.dumps(dict(vocab=2048, width=8, heads=2, depth=1, context=256)))
            for name, extra in [('raw', []), ('ema', ['--ema-decay', '.9', '--ema-start', '2'])]:
                argv = ['train.py', '--implementation', 'model', '--config', str(config),
                        '--steps', '3', '--batch-size', '1', '--threads', '2',
                        '--seed', '17', '--run-dir', str(root/name), *extra]
                with patch('sys.argv', argv), patch('train.load_data', return_value=data), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main()
            raw = torch.load(root/'raw/checkpoint.pt', weights_only=True)
            paired = torch.load(root/'ema/checkpoint.pt', weights_only=True)
            averaged = torch.load(root/'ema/checkpoint-ema.pt', weights_only=True)
            for key, value in raw['model'].items():
                torch.testing.assert_close(value, paired['model'][key], atol=0, rtol=0)
            self.assertTrue(any(not torch.equal(value, averaged['model'][key])
                                for key, value in paired['model'].items()))
            self.assertEqual(averaged['train_tokens'], 768)
            self.assertEqual(averaged['protocol'], PROTOCOL)
            self.assertEqual(averaged['recipe']['ema_start'], 2)
            self.assertEqual(averaged['recipe']['ema_updates'], 1)
            metrics = json.loads((root/'ema/metrics.json').read_text())
            self.assertTrue(torch.isfinite(torch.tensor(metrics['ema']['validation']['bpb'])))
            self.assertEqual(metrics['ema']['checkpoint'], 'checkpoint-ema.pt')
            self.assertFalse((root/'raw/checkpoint-ema.pt').exists())


class EMAContractTests(test_contract.ContractTests):
    """Apply the provided causal/window/gradient checks to loaded EMA weights."""
    def setUp(self):
        super().setUp()
        self.assertIsNotNone(importlib.util.find_spec('ema'), 'EMA has not been implemented')
        from ema import EMA
        ema = EMA(self.model, .9)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(.01)
        ema.update(self.model)
        self.model.load_state_dict(ema.state_dict())


if __name__ == '__main__':
    unittest.main()
