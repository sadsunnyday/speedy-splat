import argparse
import contextlib
import io
import json
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from arguments import OptimizationParams
from utils.general_utils import safe_state, validate_seed
from utils.prune_score_logger import collect_metadata


def parse_options(*args):
    parser = argparse.ArgumentParser()
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args(list(args)))


class SeedParameterTests(unittest.TestCase):
    def test_cli(self):
        for args, expected in [((), 0), (('--seed', '1'), 1), (('--seed', '2'), 2)]:
            with self.subTest(args=args):
                seed = parse_options(*args).seed
                self.assertIs(type(seed), int)
                self.assertEqual(seed, expected)
        opt = parse_options('--seed', '1', '--log_prune_scores', '--hard_prune_ratios', '.4,.35,.3,.25,.18')
        self.assertEqual((opt.seed, opt.log_prune_scores, opt.hard_prune_ratios), (1, True, '.4,.35,.3,.25,.18'))
        for value in ('abc', '1.5'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_options('--seed', value)

    def test_validation(self):
        for value in (0, 1, 2, 4294967295):
            self.assertEqual(validate_seed(value), value)
        for value in (-1, 4294967296, True, False, 1.5, '1', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'seed=.*0..4294967295'):
                    validate_seed(value)


class SafeStateTests(unittest.TestCase):
    def test_seed_calls_and_default(self):
        for explicit, expected in [(True, 1), (False, 0)]:
            with self.subTest(explicit=explicit), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch('utils.general_utils.random.seed') as py_seed, \
                     mock.patch('utils.general_utils.np.random.seed') as np_seed, \
                     mock.patch('utils.general_utils.torch.manual_seed') as torch_seed, \
                     mock.patch('utils.general_utils.torch.cuda.set_device') as set_device:
                    if explicit:
                        safe_state(False, seed=1)
                    else:
                        safe_state(False)
                    py_seed.assert_called_once_with(expected)
                    np_seed.assert_called_once_with(expected)
                    torch_seed.assert_called_once_with(expected)
                    set_device.assert_called_once_with(torch.device('cuda:0'))

    def test_invalid_has_no_side_effects(self):
        with contextlib.redirect_stdout(io.StringIO()):
            original_stdout = sys.stdout
            with mock.patch('utils.general_utils.random.seed') as py_seed, \
                 mock.patch('utils.general_utils.np.random.seed') as np_seed, \
                 mock.patch('utils.general_utils.torch.manual_seed') as torch_seed, \
                 mock.patch('utils.general_utils.torch.cuda.set_device') as set_device:
                with self.assertRaises(ValueError):
                    safe_state(False, seed=-1)
                self.assertIs(sys.stdout, original_stdout)
                py_seed.assert_not_called()
                np_seed.assert_not_called()
                torch_seed.assert_not_called()
                set_device.assert_not_called()

    def test_cpu_random_sequences(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            results = []
            for seed in (1, 1, 2):
                with contextlib.redirect_stdout(io.StringIO()), mock.patch('torch.cuda.set_device'):
                    safe_state(False, seed=seed)
                    results.append((random.random(), np.random.rand(3), torch.rand(3, device='cpu')))
            self.assertEqual(results[0][0], results[1][0])
            np.testing.assert_array_equal(results[0][1], results[1][1])
            self.assertTrue(torch.equal(results[0][2], results[1][2]))
            self.assertNotEqual(results[0][0], results[2][0])
            self.assertFalse(np.array_equal(results[0][1], results[2][1]))
            self.assertFalse(torch.equal(results[0][2], results[2][2]))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class MetadataSeedTests(unittest.TestCase):
    def test_metadata(self):
        fake_torch = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda=None),
                                     cuda=SimpleNamespace(get_device_name=lambda: 'test GPU'))
        dataset = SimpleNamespace(model_path='model', source_path='source')
        for seed in (0, 1):
            with self.subTest(seed=seed), mock.patch('utils.prune_score_logger.git_info', return_value={}):
                metadata = collect_metadata(dataset, SimpleNamespace(seed=seed), SimpleNamespace(),
                                            None, 0, [], 'repo', fake_torch)
                self.assertEqual(metadata['seed'], seed)
                self.assertEqual(metadata['optimization_params']['seed'], seed)
                json.dumps(metadata, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
