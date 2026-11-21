"""Reconstruction-aware hard-pruning schedule, independent of PyTorch."""
import math


class AdaptivePruning:
    def __init__(self, opt):
        self.start = max(opt.densify_until_iter, opt.prune_from_iter)
        self.stop = opt.iterations
        self.interval = opt.adaptive_check_interval
        self.max_wait = opt.adaptive_max_wait
        self.beta = opt.adaptive_beta
        self.fraction = opt.adaptive_target_fraction
        if (self.interval <= 0 or self.max_wait <= 0 or not 0 < self.beta <= 1
                or not 0 < self.fraction < 1):
            raise ValueError('Invalid adaptive pruning parameters')
        self.target = None
        self.budget = None
        self.last_iteration = None
        self.last_loss = None
        self.round = 0

    def should_check(self, iteration, count):
        return (self.start <= iteration <= self.stop and
                (self.target is None or count > self.target) and
                (iteration - self.start) % self.interval == 0)

    def trigger_reason(self, iteration, loss):
        if self.round == 0:
            return "initial"
        if loss <= self.beta * self.last_loss:
            return "loss_recovered"
        if iteration - self.last_iteration >= self.max_wait:
            return "timeout"
        return None

    def decide(self, iteration, count, loss):
        if not math.isfinite(loss) or count <= 0:
            raise ValueError('Invalid reconstruction loss or Gaussian count')
        if self.target is None:
            self.target = max(1, int(count * self.fraction))
            self.budget = count - self.target
        if count <= self.target:
            return 0.0
        if self.trigger_reason(iteration, loss) is None:
            return 0.0
        remove = min(count - self.target, max(1, self.budget >> (self.round + 1)))
        # The existing percentile deletion removes at least one point for positive ratios.
        return remove / count if remove > 0 else 0.0

    def record_prune(self, iteration, loss):
        self.last_iteration = iteration
        self.last_loss = loss
        self.round += 1
