import argparse
import json
import unittest

from arguments import OptimizationParams
from utils.soft_prune_schedule import build_soft_prune_schedule


def options(*args):
    parser = argparse.ArgumentParser()
    group = OptimizationParams(parser)
    return group.extract(parser.parse_args(list(args)))


class SoftPruneScheduleTests(unittest.TestCase):
    def test_default_and_scheduled(self):
        opt = options()
        before = vars(opt).copy()
        result = build_soft_prune_schedule(opt)
        self.assertEqual(vars(opt), before)
        self.assertEqual(result['mode'], 'constant')
        self.assertEqual(result['schedule'], [{'iteration': 6000, 'ratio': .8}, {'iteration': 9000, 'ratio': .8}, {'iteration': 12000, 'ratio': .8}])
        self.assertAlmostEqual(result['retention_product_without_densification'], .008)
        opt = options('--soft_prune_ratios', ' .84, .80 , .75 ', '--densify_prune_ratio', '.75')
        before = vars(opt).copy()
        result = build_soft_prune_schedule(opt)
        self.assertEqual(vars(opt), before)
        self.assertEqual(result['mode'], 'scheduled')
        table = {item['iteration']: item['ratio'] for item in result['schedule']}
        self.assertEqual(table, {6000: .84, 9000: .8, 12000: .75})
        self.assertEqual((table[9000], table[12000]), (.8, .75))
        self.assertAlmostEqual(result['retention_product_without_densification'], .008)
        self.assertEqual(opt.after_densify_prune_ratio, .3)
        self.assertFalse(opt.log_prune_scores)
        json.dumps(result, allow_nan=False)

    def test_constant(self):
        for raw in ('', '   '):
            opt = options('--densify_prune_ratio', '.75', '--soft_prune_ratios', raw)
            result = build_soft_prune_schedule(opt)
            self.assertEqual(result['mode'], 'constant')
            self.assertEqual([x['ratio'] for x in result['schedule']], [.75] * 3)
            self.assertAlmostEqual(result['retention_product_without_densification'], .015625)

    def test_boundaries(self):
        cases = [({'iterations': 9000}, [6000, 9000]), ({'iterations': 12000}, [6000, 9000, 12000]),
                 ({'densify_until_iter': 12000}, [6000, 9000]),
                 ({'densify_until_iter': 15500}, [6000, 9000, 12000, 15000]),
                 ({'prune_from_iter': 6500}, [9000, 12000]),
                 ({'prune_until_iter': 12000}, [6000, 9000]), ({'iterations': 5000}, [])]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                opt = options()
                for key, value in changes.items(): setattr(opt, key, value)
                result = build_soft_prune_schedule(opt)
                self.assertEqual([x['iteration'] for x in result['schedule']], expected)
                if not expected:
                    self.assertEqual(result['retention_product_without_densification'], 1.0)
                    opt.soft_prune_ratios = '.8'
                    with self.assertRaisesRegex(ValueError, 'expected 0 ratios'): build_soft_prune_schedule(opt)
        opt = options('--iterations', '9000', '--soft_prune_ratios', '.84,.8')
        self.assertEqual(len(build_soft_prune_schedule(opt)['schedule']), 2)
        opt.soft_prune_ratios = '.84,.8,.75'
        with self.assertRaisesRegex(ValueError, r'expected 2 ratios.*6000.*9000.*got 3'):
            build_soft_prune_schedule(opt)
        opt = options('--densify_until_iter', '15500', '--soft_prune_ratios', '.84,.8,.75,.7')
        self.assertEqual(len(build_soft_prune_schedule(opt)['schedule']), 4)

    def test_invalid(self):
        for name, values in [('iterations', [0, -1, True]), ('prune_interval', [0, -1, 1.5, False]),
                             ('densify_until_iter', [-1, True]), ('prune_from_iter', [-1, False]),
                             ('prune_until_iter', [-1, True])]:
            for value in values:
                with self.subTest(name=name, value=value):
                    opt = options()
                    setattr(opt, name, value)
                    with self.assertRaisesRegex(ValueError, name): build_soft_prune_schedule(opt)
        for value in ('0', '1', '-.1', '84', 'NaN', 'Inf', 'abc', '84%', '[0.84]', '.84,,.75', '.84,', '.84,.8', '.84,.8,.75,.7'):
            with self.subTest(value=value):
                opt = options('--soft_prune_ratios', value)
                with self.assertRaisesRegex(ValueError, 'soft_prune_ratios'): build_soft_prune_schedule(opt)
        for value in (0, 1, -1, 84, float('nan'), float('inf')):
            opt = options()
            opt.densify_prune_ratio = value
            with self.assertRaisesRegex(ValueError, 'densify_prune_ratio'): build_soft_prune_schedule(opt)


if __name__ == '__main__':
    unittest.main()
