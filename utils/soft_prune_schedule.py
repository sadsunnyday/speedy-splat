"""Build the full soft-pruning schedule without training dependencies."""

import math


def _integer(opt, name, minimum):
    value = getattr(opt, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError('{}={!r}: must be an integer >= {}'.format(name, value, minimum))
    return value


def _ratio(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('{}={!r}: must be finite and strictly between 0 and 1'.format(name, value))
    if not math.isfinite(result) or not 0 < result < 1:
        raise ValueError('{}={!r}: must be finite and strictly between 0 and 1'.format(name, value))
    return result


def build_soft_prune_schedule(opt):
    iterations = _integer(opt, 'iterations', 1)
    interval = _integer(opt, 'prune_interval', 1)
    densify_until = _integer(opt, 'densify_until_iter', 0)
    prune_from = _integer(opt, 'prune_from_iter', 0)
    prune_until = _integer(opt, 'prune_until_iter', 0)
    rounds = [i for i in range(interval, iterations + 1, interval)
              if i >= prune_from and i < prune_until and i < densify_until]
    raw = opt.soft_prune_ratios
    if not isinstance(raw, str):
        raise ValueError('soft_prune_ratios={!r}: must be a comma-separated string'.format(raw))
    if not raw.strip():
        mode = 'constant'
        ratios = [_ratio(opt.densify_prune_ratio, 'densify_prune_ratio')] * len(rounds) if rounds else []
    else:
        mode = 'scheduled'
        parts = raw.split(',')
        if any(not part.strip() for part in parts):
            raise ValueError('soft_prune_ratios={!r}: empty ratio element'.format(raw))
        ratios = [_ratio(part.strip(), 'soft_prune_ratios') for part in parts]
        if len(ratios) != len(rounds):
            raise ValueError('soft_prune_ratios={!r}: expected {} ratios for iterations {}, got {}'.format(
                raw, len(rounds), rounds, len(ratios)))
    schedule = []
    retention = 1.0
    for iteration, ratio in zip(rounds, ratios):
        schedule.append({'iteration': iteration, 'ratio': ratio})
        retention *= 1 - ratio
    return {'mode': mode, 'schedule': schedule, 'retention_product_without_densification': retention}
