import argparse
import json
from types import SimpleNamespace
import unittest

from arguments import OptimizationParams
from utils.prune_schedule import build_hard_prune_schedule


def options(*args):
    parser = argparse.ArgumentParser()
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args(list(args)))


class HardPruneScheduleTests(unittest.TestCase):
    def test_default_and_experiment(self):
        opt = options()
        original = vars(opt).copy()
        config = build_hard_prune_schedule(opt)
        self.assertEqual(vars(opt), original)
        self.assertFalse(opt.log_prune_scores)
        self.assertEqual(config['mode'], 'constant')
        self.assertEqual([x['iteration'] for x in config['schedule']], [15000, 18000, 21000, 24000, 27000])
        self.assertEqual([x['ratio'] for x in config['schedule']], [.30] * 5)
        self.assertAlmostEqual(config['planned_retention_fraction'], .16807)
        opt = options('--hard_prune_ratios', '0.40,0.35,0.30,0.25,0.18', '--after_densify_prune_ratio', '0.25')
        original = vars(opt).copy()
        config = build_hard_prune_schedule(opt)
        self.assertEqual(vars(opt), original)
        self.assertEqual(config['mode'], 'scheduled')
        self.assertEqual([x['ratio'] for x in config['schedule']], [.4, .35, .3, .25, .18])
        self.assertAlmostEqual(config['planned_retention_fraction'], .167895)
        self.assertEqual(opt.densify_prune_ratio, .8)
        self.assertFalse(opt.log_prune_scores)
        table = {x['iteration']: x['ratio'] for x in config['schedule']}
        self.assertEqual([table[i] for i in (21000, 24000, 27000)], [.3, .25, .18])
        json.dumps(config, allow_nan=False)

    def test_constant_and_whitespace(self):
        opt = options('--after_densify_prune_ratio', '.25', '--hard_prune_ratios', '  ')
        result = build_hard_prune_schedule(opt)
        self.assertEqual(result['mode'], 'constant')
        self.assertEqual([x['ratio'] for x in result['schedule']], [.25] * 5)
        self.assertAlmostEqual(result['planned_retention_fraction'], .2373046875)
        opt.hard_prune_ratios = ' .4, .35 , .3, .25, .18 '
        self.assertEqual([x['ratio'] for x in build_hard_prune_schedule(opt)['schedule']], [.4, .35, .3, .25, .18])

    def test_iteration_boundaries(self):
        cases = [({'iterations': 21000}, [15000, 18000, 21000]),
                 ({'densify_until_iter': 15500}, [18000, 21000, 24000, 27000]),
                 ({'prune_from_iter': 16000}, [18000, 21000, 24000, 27000]),
                 ({'prune_until_iter': 24000}, [15000, 18000, 21000]),
                 ({'iterations': 14000}, [])]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                opt = options()
                for name, value in changes.items(): setattr(opt, name, value)
                config = build_hard_prune_schedule(opt)
                self.assertEqual([x['iteration'] for x in config['schedule']], expected)
                if not expected:
                    self.assertEqual(config['planned_retention_fraction'], 1.0)
                    opt.hard_prune_ratios = '.4'
                    with self.assertRaisesRegex(ValueError, 'expected 0 ratios'): build_hard_prune_schedule(opt)
        opt = options('--iterations', '21000', '--hard_prune_ratios', '.4,.35,.3')
        self.assertEqual(len(build_hard_prune_schedule(opt)['schedule']), 3)
        opt.hard_prune_ratios = '.4,.35,.3,.25,.18'
        with self.assertRaisesRegex(ValueError, r'expected 3 ratios.*15000.*18000.*21000'):
            build_hard_prune_schedule(opt)

    def test_invalid_integer_parameters(self):
        for name, values in [('iterations', [0, -1, True]), ('prune_interval', [0, -1, 1.5, False]),
                             ('densify_until_iter', [-1, True]), ('prune_from_iter', [-1, False]),
                             ('prune_until_iter', [-1, True])]:
            for value in values:
                with self.subTest(name=name, value=value):
                    opt = options()
                    setattr(opt, name, value)
                    with self.assertRaisesRegex(ValueError, name): build_hard_prune_schedule(opt)

    def test_invalid_ratios_and_lengths(self):
        for value in ['0', '1', '-0.1', '40', 'NaN', 'Inf', '40%', 'abc', '[0.4]',
                      '0.4,,0.3', '0.4,', '.4,.3', '.4,.3,.2,.1,.2,.3']:
            with self.subTest(value=value):
                opt = options('--hard_prune_ratios', value)
                with self.assertRaisesRegex(ValueError, 'hard_prune_ratios'): build_hard_prune_schedule(opt)
        for value in [0, 1, -0.1, 40, float('nan'), float('inf')]:
            with self.subTest(constant=value):
                opt = options()
                opt.after_densify_prune_ratio = value
                with self.assertRaisesRegex(ValueError, 'after_densify_prune_ratio'): build_hard_prune_schedule(opt)


if __name__ == '__main__':
    unittest.main()
