"""Build the complete hard-pruning schedule from training options."""

import math


def _integer_parameter(opt, name, minimum):
    value = getattr(opt, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError('{}={!r}: must be an integer >= {}'.format(name, value, minimum))
    return value


def _ratio(value, name):
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('{}={!r}: must be a finite number strictly between 0 and 1'.format(name, value))
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('{}={!r}: must be a finite number strictly between 0 and 1'.format(name, value))
    return ratio


def build_hard_prune_schedule(opt):
    """Return the mode, full iteration schedule, and planned retention fraction."""
    iterations = _integer_parameter(opt, 'iterations', 1)
    interval = _integer_parameter(opt, 'prune_interval', 1)
    densify_until = _integer_parameter(opt, 'densify_until_iter', 0)
    prune_from = _integer_parameter(opt, 'prune_from_iter', 0)
    prune_until = _integer_parameter(opt, 'prune_until_iter', 0)
    rounds = [i for i in range(interval, iterations + 1, interval)
              if i >= densify_until and i >= prune_from and i < prune_until]

    raw = opt.hard_prune_ratios
    if not isinstance(raw, str):
        raise ValueError('hard_prune_ratios={!r}: must be a comma-separated string'.format(raw))
    if not raw.strip():
        mode = 'constant'
        ratios = [_ratio(opt.after_densify_prune_ratio, 'after_densify_prune_ratio')] * len(rounds) if rounds else []
    else:
        mode = 'scheduled'
        parts = raw.split(',')
        if any(not part.strip() for part in parts):
            raise ValueError('hard_prune_ratios={!r}: empty ratio element'.format(raw))
        ratios = [_ratio(part.strip(), 'hard_prune_ratios') for part in parts]
        if len(ratios) != len(rounds):
            raise ValueError('hard_prune_ratios={!r}: expected {} ratios for iterations {}, got {}'.format(
                raw, len(rounds), rounds, len(ratios)))

    schedule = []
    retention = 1.0
    for iteration, ratio in zip(rounds, ratios):
        schedule.append({'iteration': iteration, 'ratio': ratio})
        retention *= 1 - ratio
    return {'mode': mode, 'schedule': schedule, 'planned_retention_fraction': retention}
