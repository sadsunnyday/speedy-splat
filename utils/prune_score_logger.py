import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

_PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 80, 90, 95, 99, 100)
_KEYS = tuple('q%02d' % x for x in _PERCENTILES)


def summarize_scores(scores, requested_ratio, n_before, n_after):
    if (not isinstance(scores, np.ndarray) or scores.ndim != 1 or
            not np.issubdtype(scores.dtype, np.floating)):
        raise ValueError('scores must be a one-dimensional floating array')
    if (not isinstance(n_before, (int, np.integer)) or not isinstance(n_after, (int, np.integer)) or
            n_before != len(scores) or not 0 <= n_after <= n_before):
        raise ValueError('invalid counts')
    try:
        ratio = float(requested_ratio)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('invalid requested ratio')
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('invalid requested ratio')
    issues = []
    finite_mask = np.isfinite(scores)
    finite = scores[finite_mask].astype(np.float64)
    positive = finite[finite > 0]
    counts = {'nan': int(np.isnan(scores).sum()), 'posinf': int(np.isposinf(scores).sum()),
              'neginf': int(np.isneginf(scores).sum()), 'finite_negative': int((finite < 0).sum()),
              'zero': int((finite == 0).sum()), 'finite_positive': int(len(positive))}
    if not finite_mask.all(): issues.append('nonfinite_scores')
    if counts['finite_negative']: issues.append('negative_scores')
    if not n_before: issues.append('empty_scores')

    def clean(value):
        value = float(value)
        if not np.isfinite(value):
            if 'nonfinite_derived_stat' not in issues: issues.append('nonfinite_derived_stat')
            return None
        return value

    def quantiles(values):
        if not len(values): return dict.fromkeys(_KEYS)
        with np.errstate(all='ignore'):
            return dict(zip(_KEYS, [clean(v) for v in np.percentile(values, _PERCENTILES)]))

    def summary(values):
        if not len(values):
            return dict(count=0, min=None, max=None, mean=None, std=None, sum=0.0, quantiles=quantiles(values))
        with np.errstate(all='ignore'):
            return dict(count=int(len(values)), min=clean(np.min(values)), max=clean(np.max(values)),
                        mean=clean(np.mean(values)), std=clean(np.std(values)), sum=clean(np.sum(values)),
                        quantiles=quantiles(values))

    finite_summary, positive_summary = summary(finite), summary(positive)
    logs = np.log10(positive)
    log_q = quantiles(logs)
    if len(logs):
        low, high = float(np.min(logs)), float(np.max(logs))
        if low == high: low, high = low - 0.5, high + 0.5
        hist_counts, edges = np.histogram(logs, bins=32, range=(low, high))
        log_summary = dict(quantiles=log_q, iqr=clean(log_q['q75'] - log_q['q25']),
                           hist_edges=edges.tolist(), hist_counts=hist_counts.tolist())
    else:
        log_summary = dict(quantiles=log_q, iqr=None, hist_edges=[], hist_counts=[])
    threshold = dict(status='empty' if not n_before else 'nonfinite', index=None,
                     value_reconstructed=None, count_below=None, count_equal=None, count_le=None,
                     count_matches_actual=None, pruned_score_fraction=None)
    n_pruned = int(n_before - n_after)
    if n_before and finite_mask.all():
        index = int(ratio * (n_before - 1))
        value = np.sort(scores)[index]
        below, equal, le = int((scores < value).sum()), int((scores == value).sum()), int((scores <= value).sum())
        matches = le == n_pruned
        threshold.update(status='ok', index=index, value_reconstructed=float(value), count_below=below,
                         count_equal=equal, count_le=le, count_matches_actual=matches)
        if not matches: issues.append('threshold_count_mismatch')
        if not counts['finite_negative']:
            with np.errstate(all='ignore'):
                total = np.sum(finite)
                if total > 0:
                    threshold['pruned_score_fraction'] = clean(np.sum(scores[scores <= value].astype(np.float64)) / total)
    return dict(n_pruned=n_pruned, actual_ratio=n_pruned / n_before if n_before else None,
                counts=counts, finite_summary=finite_summary, positive_summary=positive_summary,
                log10_positive=log_summary, threshold=threshold, issues=issues)


def git_info(path):
    try:
        def run(*args):
            return subprocess.check_output(['git', '-C', path] + list(args), stderr=subprocess.STDOUT,
                                           timeout=2).decode('utf-8', errors='replace').strip()
        # A missing submodule must not silently report the parent repository's HEAD.
        root = run('rev-parse', '--show-toplevel')
        if os.path.normcase(os.path.realpath(root)) != os.path.normcase(os.path.realpath(path)):
            return {'head': None, 'dirty': None, 'reason': 'not a repository root'}
        return {'head': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain')), 'reason': None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'head': None, 'dirty': None, 'reason': type(exc).__name__}


def collect_metadata(dataset, opt, pipe, checkpoint, resume_iteration, cameras, repo_root, torch):
    sizes = {}
    for camera in cameras:
        key = (int(camera.image_width), int(camera.image_height))
        sizes[key] = sizes.get(key, 0) + 1
    return dict(model_path=dataset.model_path, source_path=dataset.source_path, checkpoint=checkpoint,
                resume_iteration=resume_iteration, model_params=vars(dataset), optimization_params=vars(opt),
                pipeline_params=vars(pipe), seed=0, python_version=sys.version, numpy_version=np.__version__,
                pytorch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(), train_view_count=len(cameras),
                image_sizes=[dict(width=k[0], height=k[1], count=v) for k, v in sorted(sizes.items())],
                git_main=git_info(repo_root), git_renderer=git_info(os.path.join(repo_root, 'submodules', 'diff-gaussian-rasterization')),
                statistics_notes={'timezone': 'UTC', 'histogram': '32 equal-width log10 bins; edges vary by round',
                                  'score': 'renderer accumulated squared score, not image quality loss',
                                  'memory': 'pruning-stage process peak including resident allocations',
                                  'overhead': 'copy and CPU summary exclude CUDA event wait and JSON writing'})


class PruneScoreLogger:
    def __init__(self, run_id, output_dir):
        self.run_id, self.output_dir, self.enabled = run_id, output_dir, True

    @classmethod
    def create(cls, model_path, metadata):
        base = os.path.join(model_path, 'prune_score_logs')
        now = datetime.datetime.utcnow()
        stem = now.strftime('%Y%m%dT%H%M%S.%fZ') + '-' + str(os.getpid())
        output_dir = base
        try:
            os.makedirs(base, exist_ok=True)
            suffix = 0
            while True:
                run_id = stem + ('-' + str(suffix) if suffix else '')
                output_dir = os.path.join(base, run_id)
                try:
                    os.makedirs(output_dir, exist_ok=False)
                    break
                except FileExistsError:
                    suffix += 1
            data = dict(metadata, schema_version=1, run_id=run_id, created_utc=now.isoformat() + 'Z')
            serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, default=_json_default)
            with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as handle:
                handle.write(serialized + '\n')
            return cls(run_id, output_dir)
        except OSError as exc:
            print('Warning: prune score logging initialization failed at {}: {}; logs incomplete'.format(output_dir, exc), file=sys.stderr)
            return None

    def record(self, scores, context, copy_ms):
        if not self.enabled: return
        start = time.perf_counter()
        stats = summarize_scores(scores, context['requested_ratio'], context['n_before'], context['n_after'])
        summary_ms = (time.perf_counter() - start) * 1000
        data = dict(context, schema_version=1, run_id=self.run_id, score_copy_ms=copy_ms, cpu_summary_ms=summary_ms)
        data.update(stats)
        if 'threshold_count_mismatch' in stats['issues']:
            print('Warning: prune threshold count mismatch at iteration {}'.format(context.get('iteration')), file=sys.stderr)
        line = json.dumps(data, ensure_ascii=False, allow_nan=False, default=_json_default) + '\n'
        path = os.path.join(self.output_dir, 'rounds.jsonl')
        try:
            with open(path, 'a', encoding='utf-8') as handle:
                handle.write(line)
                handle.flush()
        except OSError as exc:
            self.enabled = False
            print('Warning: prune score logging failed at {} iteration {}: {}; logs incomplete and disabled'.format(path, context.get('iteration'), exc), file=sys.stderr)


def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError('Object of type {} is not JSON serializable'.format(type(value).__name__))
