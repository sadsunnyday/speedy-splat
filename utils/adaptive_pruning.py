"""Reconstruction-aware hard-pruning schedule, independent of PyTorch."""
import math


class AdaptivePruning:
    def __init__(self, opt):
        self.mode = getattr(opt, "adaptive_trigger_mode", "legacy")
        if self.mode not in ("legacy", "recover"):
            raise ValueError("Invalid adaptive trigger mode")
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
        self.last_pre_loss = None
        self.last_post_loss = None
        self.round = 0

    def should_check(self, iteration, count):
        return (self.start <= iteration and
                (iteration < self.stop if self.mode == "recover" else iteration <= self.stop) and
                (self.target is None or count > self.target) and
                (iteration - self.start) % self.interval == 0)

    def decide(self, iteration, count, loss):
        if not math.isfinite(loss) or count <= 0:
            raise ValueError('Invalid reconstruction loss or Gaussian count')
        if self.target is None:
            self.target = max(1, int(count * self.fraction))
            self.budget = count - self.target
        if count <= self.target:
            return 0.0
        if self.trigger_reason(iteration, loss) == "wait":
            return 0.0
        remove = min(count - self.target, max(1, self.budget >> (self.round + 1)))
        # The existing percentile deletion removes at least one point for positive ratios.
        return remove / count if remove > 0 else 0.0

    def trigger_reason(self, iteration, loss):
        if not self.round:
            return "initial"
        threshold = self.last_pre_loss if self.mode == "recover" else self.beta * self.last_loss
        if loss <= threshold:
            return "recovered"
        if iteration - self.last_iteration >= self.max_wait:
            return "timeout"
        return "wait"

    def record_prune(self, iteration, loss, post_loss=None):
        if self.mode == "recover" and (post_loss is None or not math.isfinite(post_loss)):
            raise ValueError("Recover mode requires a finite post-pruning loss")
        self.last_pre_loss = loss
        self.last_post_loss = post_loss
        self.last_iteration = iteration
        self.last_loss = loss
        self.round += 1

    def state_dict(self):
        return dict(mode=self.mode, last_pre_loss=self.last_pre_loss, last_post_loss=self.last_post_loss, target=self.target, budget=self.budget, last_iteration=self.last_iteration,
                    last_loss=self.last_loss, round=self.round,
                    config=(self.start, self.stop, self.interval, self.max_wait, self.beta, self.fraction))

    def load_state_dict(self, state):
        saved_mode = state.get('mode', 'legacy')
        if saved_mode != self.mode:
            raise ValueError('Adaptive pruning trigger mode differs from checkpoint')
        if self.mode == 'recover' and (any(k not in state for k in ('last_pre_loss', 'last_post_loss', 'round')) or
                (state['round'] and (state['last_pre_loss'] is None or state['last_post_loss'] is None))):
            raise ValueError('Recover checkpoint is missing pruning loss state')
        if tuple(state['config']) != (self.start, self.stop, self.interval, self.max_wait, self.beta, self.fraction):
            raise ValueError('Adaptive pruning configuration differs from checkpoint')
        self.last_pre_loss = state.get('last_pre_loss')
        self.last_post_loss = state.get('last_post_loss')
        for key in ('target', 'budget', 'last_iteration', 'last_loss', 'round'):
            setattr(self, key, state[key])
