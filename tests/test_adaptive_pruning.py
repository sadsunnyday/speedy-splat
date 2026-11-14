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

class RecoveryTests(unittest.TestCase):
    def make(self):
        s = scheduler()
        s.mode = "recover"
        return s

    def test_recovery_and_timeout(self):
        s = self.make()
        self.assertEqual(s.trigger_reason(15000, .015), "initial")
        self.assertAlmostEqual(s.decide(15000, 1000, .015), .45)
        s.record_prune(15000, .015, .016)
        self.assertEqual(s.trigger_reason(15500, .0152), "wait")
        self.assertEqual(s.decide(15500, 550, .0152), 0)
        self.assertEqual(s.trigger_reason(15500, .015), "recovered")
        self.assertGreater(s.decide(15500, 550, .015), 0)
        self.assertEqual(s.trigger_reason(17000, .0152), "timeout")
        self.assertEqual(s.trigger_reason(17000, .015), "recovered")
        for post in (.015, .014):
            s.last_post_loss = post
            self.assertEqual(s.trigger_reason(15500, .0151), "wait")
            self.assertEqual(s.trigger_reason(15500, .015), "recovered")
        self.assertFalse(s.should_check(15500, 100))
        self.assertFalse(s.should_check(40000, 550))

    def test_checkpoint_compatibility(self):
        s = self.make()
        s.decide(15000, 1000, .015)
        s.record_prune(15000, .015, .016)
        other = self.make()
        other.load_state_dict(s.state_dict())
        self.assertEqual(other.state_dict(), s.state_dict())
        self.assertEqual(other.trigger_reason(15500, .0152), s.trigger_reason(15500, .0152))
        with self.assertRaisesRegex(ValueError, "mode"):
            scheduler().load_state_dict(s.state_dict())
        old = scheduler().state_dict()
        for key in ('mode', 'last_pre_loss', 'last_post_loss'):
            del old[key]
        scheduler().load_state_dict(old)
        with self.assertRaisesRegex(ValueError, "mode"):
            self.make().load_state_dict(old)
        state = s.state_dict()
        del state['last_post_loss']
        with self.assertRaisesRegex(ValueError, "missing"):
            self.make().load_state_dict(state)
        other.max_wait += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            other.load_state_dict(s.state_dict())


if __name__ == '__main__':
    unittest.main()
