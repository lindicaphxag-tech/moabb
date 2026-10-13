"""Controlled counterfactual witness for post-IIR CV leakage beyond epoch overlap.

Scope: deterministic manufactured signals, not a clinical EEG accuracy or
independent population study.  Genuine scipy SOS zero-phase bandpass is used.
A raw-extraction purging scheme is not a complete independence certificate
when whole-recording filtering precedes fold partitioning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class FilterFoldWitness:
    seed: int
    guard_samples: int
    event_window_samples: tuple[int, int]
    training_window_samples: tuple[int, int]
    direct_raw_overlap_samples: int
    whole_record_training_l2_change: float
    fold_masked_training_l2_change: float
    source: str = "scipy.signal.sosfiltfilt on frozen synthetic signal"
    task_accuracy_evidence: bool = False


def counterfactual_filter_witness(
    seed: int, *,
    guard_samples: int,
    sample_rate_hz: float = 250.0,
    record_samples: int = 4096,
    train_samples: int = 320,
    test_interval: tuple[int, int] = (1800, 2120),
    burst_amplitude: float = 3.0,
) -> FilterFoldWitness:
    """Measure a held-out raw perturbation's influence on nonoverlap train data.

    Both worlds share the same continuous recording, EEG background and split;
    only the held-out test interval's values differ.  The fold-masked negative
    control zeros the WHOLE held-out interval *before* global filtering in both
    worlds.  This proves counterfactual invariance for this control only, not
    that fold-specific zero masking is a scientifically fair evaluation design.

    No iid assumption about the filtered samples is used. Seeds index distinct
    synthetic response perturbations, not independent human participants.
    """
    if type(seed) is not int or type(guard_samples) is not int or guard_samples < 0:
        raise ValueError("integer seed and nonnegative integer guard required")
    if (not np.isfinite(sample_rate_hz) or sample_rate_hz <= 60
            or record_samples < 1024 or train_samples <= 0
            or not np.isfinite(burst_amplitude) or burst_amplitude <= 0):
        raise ValueError("invalid source/filter/perturbation settings")
    a, b = test_interval
    if (type(a) is not int or type(b) is not int
            or not 0 <= a < b <= record_samples):
        raise ValueError("invalid contiguous held-out test interval")
    lo, hi = a - guard_samples - train_samples, a - guard_samples
    if lo < 0 or hi <= lo:
        raise ValueError("training interval is empty or outside record")
    rng = np.random.default_rng(seed)
    t = np.arange(record_samples) / sample_rate_hz
    x = (np.sin(2 * np.pi * 7.3 * t) + .3 * np.sin(2 * np.pi * 18 * t)
         + rng.normal(scale=.1, size=record_samples))
    delta = np.zeros_like(x)
    # Randomized signal frequencies and phases prevent 32 seeds from merely
    # duplicating the SAME linear IIR impulse transfer response.
    f1, f2 = rng.uniform(2.5, 12), rng.uniform(3, 20)
    p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
    delta[a:b] = burst_amplitude * (
        np.sin(2 * np.pi * f1 * t[a:b] + p1)
        + .3 * np.sin(2 * np.pi * f2 * t[a:b] + p2)
    )
    sos = butter(4, [1, 30], btype="bandpass", fs=sample_rate_hz, output="sos")
    full_first = sosfiltfilt(sos, x)
    full_second = sosfiltfilt(sos, x + delta)
    train = slice(lo, hi)
    effect = float(np.linalg.norm(full_second[train] - full_first[train]))
    # Negative control prevents the counterfactual test information from
    # reaching the filtering operator; it is NOT claimed deployable as-is.
    masked_a, masked_b = x.copy(), (x + delta).copy()
    masked_a[a:b] = masked_b[a:b] = 0.0
    folded_a = sosfiltfilt(sos, masked_a)
    folded_b = sosfiltfilt(sos, masked_b)
    null_effect = float(np.linalg.norm(folded_b[train] - folded_a[train]))
    return FilterFoldWitness(
        seed=seed, guard_samples=guard_samples,
        event_window_samples=(a, b), training_window_samples=(lo, hi),
        direct_raw_overlap_samples=max(0, min(hi, b) - max(lo, a)),
        whole_record_training_l2_change=effect,
        fold_masked_training_l2_change=null_effect,
    )


def run_frozen_control(
    *, seeds: range = range(2026101300, 2026101332),
    guards: tuple[int, ...] = (0, 80, 320, 640),
) -> dict:
    if len(seeds) != 32 or tuple(guards) != (0, 80, 320, 640):
        raise ValueError("do not silently alter prospective denominator/guards")
    results = [
        counterfactual_filter_witness(seed, guard_samples=guard)
        for seed in seeds for guard in guards
    ]
    assert len(results) == 128
    summary = {}
    for guard in guards:
        rows = [r for r in results if r.guard_samples == guard]
        full = np.array([r.whole_record_training_l2_change for r in rows])
        masked = np.array([r.fold_masked_training_l2_change for r in rows])
        summary[str(guard)] = {
            "synthetic_signal_pairs": len(rows),
            "whole_record_influence_l2_median": float(np.median(full)),
            "whole_record_influence_l2_min": float(full.min()),
            "nonzero_whole_record_instances": int(np.sum(full > 1e-8)),
            "fold_masked_influence_l2_max": float(masked.max()),
            "direct_raw_epoch_overlap_instances": sum(
                r.direct_raw_overlap_samples > 0 for r in rows),
        }
    return {
        "schema": "synthetic_counterfactual_filter_fold_leakage_v1",
        "source": "SciPy SOS filtfilt; no clinical datasets; no model training",
        "precommitted_seed_count": 32,
        "guard_register": list(guards),
        "record_sample_count": 4096,
        "filter": "Butterworth 4th-order 1–30 Hz zero-phase bandpass at 250 Hz",
        "heldout_interval": [1800, 2120],
        "decision": (
            "Counterfactual held-out effects on train features can survive "
            "direct raw epoch disjointness when filtering is global."
        ),
        "negative_control": "zero held-out source values before filtering",
        "not_claimed": [
            "post-IIR independence from PurgedEpochKFold",
            "unbiased classifier accuracy",
            "real EEG data or independent human participants",
            "that fold-masking is a usable MOABB default",
            "formal zero-tail IIR response at any finite guard",
        ],
        "summary": summary,
        "paired_detailed_rows": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="filter_fold_dependency_witness.json")
    args = parser.parse_args()
    result = run_frozen_control()
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("FILTER_DEPENDENCE_AUDIT", json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
