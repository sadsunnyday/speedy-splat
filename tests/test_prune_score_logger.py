import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from utils.prune_score_logger import PruneScoreLogger, summarize_scores


class SummaryTests(unittest.TestCase):
    def check(self, values, ratio, after):
        scores = np.array(values, dtype=np.float32)
        original = scores.copy()
        result = summarize_scores(scores, ratio, len(scores), after)
        np.testing.assert_array_equal(scores, original)
        self.assertEqual(sum(result['counts'].values()), len(scores))
        self.assertEqual(sum(result['log10_positive']['hist_counts']), result['counts']['finite_positive'])
        for name in ('finite_summary', 'positive_summary'):
            q = list(result[name]['quantiles'].values())
            if q[0] is not None: self.assertEqual(q, sorted(q))
        json.dumps(result, allow_nan=False)
        return result

    def test_ties_and_threshold(self):
        r = self.check([0, 0, 1, 1, 2], .5, 1)
        t = r['threshold']
        self.assertEqual((t['index'], t['value_reconstructed'], t['count_below'], t['count_equal'], t['count_le']), (2, 1, 2, 2, 4))
        self.assertEqual((r['n_pruned'], r['actual_ratio'], t['pruned_score_fraction']), (4, .8, .5))
        self.assertEqual(self.check([1, 2, 3, 4], .5, 2)['threshold']['value_reconstructed'], 2)

    def test_zero_constant_empty_singleton(self):
        r = self.check([0, 0, 0], .3, 0)
        self.assertEqual((r['actual_ratio'], r['positive_summary']['count'], r['threshold']['pruned_score_fraction']), (1, 0, None))
        self.assertEqual(r['log10_positive']['hist_edges'], [])
        r = self.check([2, 2, 2], .3, 0)
        self.assertEqual((len(r['log10_positive']['hist_edges']), len(r['log10_positive']['hist_counts']), r['log10_positive']['iqr']), (33, 32, 0))
        r = self.check([], .3, 0)
        self.assertIsNone(r['actual_ratio'])
        self.assertEqual(r['threshold']['status'], 'empty')
        r = self.check([5], .3, 0)
        self.assertEqual((r['threshold']['index'], r['threshold']['value_reconstructed'], r['actual_ratio']), (0, 5, 1))

    def test_nonfinite_and_negative(self):
        r = self.check([np.nan, np.inf, -np.inf, -1, 0, 2], .5, 3)
        self.assertEqual(list(r['counts'].values()), [1] * 6)
        self.assertEqual(r['threshold']['status'], 'nonfinite')
        r = self.check([-1, 0, 2], .5, 1)
        self.assertEqual((r['threshold']['value_reconstructed'], r['threshold']['count_le']), (0, 2))
        self.assertIn('negative_scores', r['issues'])
        self.assertIsNone(r['threshold']['pruned_score_fraction'])
        r = self.check([1, 2, 3], .5, 0)
        self.assertIn('threshold_count_mismatch', r['issues'])
        self.assertFalse(r['threshold']['count_matches_actual'])

    def test_validation_and_overflow(self):
        for args in [(np.array([1]), .5, 1, 0), (np.ones((1, 1)), .5, 1, 0),
                     (np.ones(1), np.nan, 1, 0), (np.ones(1), 2, 1, 0),
                     (np.ones(1), .5, 2, 0), (np.ones(1), .5, 1, 2)]:
            with self.assertRaises(ValueError): summarize_scores(*args)
        r = summarize_scores(np.array([1e308, 1e308]), .5, 2, 0)
        self.assertIn('nonfinite_derived_stat', r['issues'])
        json.dumps(r, allow_nan=False)


class LoggerTests(unittest.TestCase):
    def test_sessions_and_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = PruneScoreLogger.create(tmp, {}), PruneScoreLogger.create(tmp, {})
            self.assertNotEqual(a.output_dir, b.output_dir)
            context = dict(iteration=10, stage='soft', n_before=2, n_after=1, requested_ratio=.5)
            for _ in range(2): a.record(np.array([1., 2.]), context, 1.0)
            with open(os.path.join(a.output_dir, 'rounds.jsonl'), encoding='utf-8') as f:
                rows = [json.loads(line, parse_constant=lambda x: self.fail(x)) for line in f]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['run_id'], a.run_id)
            self.assertTrue(os.path.exists(os.path.join(b.output_dir, 'metadata.json')))

    def test_io_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = PruneScoreLogger.create(tmp, {})
            with mock.patch('builtins.open', side_effect=OSError('no space')), contextlib.redirect_stderr(io.StringIO()) as err:
                logger.record(np.array([1.]), dict(iteration=7, n_before=1, n_after=0, requested_ratio=.5), 0)
            self.assertFalse(logger.enabled)
            self.assertIn('incomplete', err.getvalue())
            with mock.patch('os.makedirs', side_effect=OSError('denied')), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertIsNone(PruneScoreLogger.create(tmp, {}))
            self.assertIn('incomplete', err.getvalue())


if __name__ == '__main__': unittest.main()
