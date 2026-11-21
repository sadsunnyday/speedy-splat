import unittest
from types import SimpleNamespace
from utils.adaptive_pruning import AdaptivePruning


def scheduler():
    return AdaptivePruning(SimpleNamespace(densify_until_iter=15000, prune_from_iter=6000,
        prune_until_iter=30000, iterations=40000, adaptive_check_interval=500, adaptive_max_wait=2000,
        adaptive_beta=.95, adaptive_target_fraction=.1))


class AdaptiveTests(unittest.TestCase):
    def test_trigger_budget_and_resume(self):
        s = scheduler()
        self.assertTrue(s.should_check(15000, 1000))
        self.assertAlmostEqual(s.decide(15000, 1000, .1), .45)
        s.record_prune(15000, .1)
        self.assertEqual(s.decide(15500, 550, .096), 0)
        self.assertGreater(s.decide(15500, 550, .095), 0)
        self.assertGreater(s.decide(17000, 550, .2), 0)
        restored = scheduler()
        restored.load_state_dict(s.state_dict())
        self.assertEqual(restored.state_dict(), s.state_dict())
        self.assertEqual(restored.decide(15500, 550, .096), 0)

    def test_unlimited_rounds_and_target(self):
        s = scheduler()
        s.decide(15000, 1000, .1)
        s.round = 4
        s.last_iteration, s.last_loss = 15000, .1
        self.assertAlmostEqual(s.decide(17000, 200, .2), 28 / 200)
        s.round = 30
        self.assertTrue(s.should_check(17000, 101))
        self.assertAlmostEqual(s.decide(17000, 101, .2), 1 / 101)
        self.assertEqual(s.decide(17000, 100, .2), 0)
        self.assertFalse(s.should_check(17500, 100))
        self.assertTrue(s.should_check(30000, 101))
        self.assertTrue(s.should_check(40000, 101))
        self.assertFalse(s.should_check(40500, 101))

if __name__ == '__main__':
    unittest.main()
