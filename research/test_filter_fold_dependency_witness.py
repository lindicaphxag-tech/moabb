"""Source-matched counterfactual filter leakage tests; no human EEG claims."""
import json

import numpy as np
import pytest

from research.filter_fold_dependency_witness import (
    counterfactual_filter_witness,
    run_frozen_control,
)


@pytest.mark.parametrize("guard", [0, 80, 320, 640])
def test_direct_interval_purge_is_not_post_iir_independence(guard):
    # Real SOS zero-phase IIR operation: no shared raw train/test samples
    # but altered heldout data still affect an earlier, nonoverlap train epoch.
    for seed in (1, 4, 9):
        row = counterfactual_filter_witness(seed, guard_samples=guard)
        assert row.direct_raw_overlap_samples == 0
        assert row.whole_record_training_l2_change > 1e-8
        assert row.fold_masked_training_l2_change == 0
        assert row.training_window_samples[1] <= row.event_window_samples[0]


def test_sensitivity_decreases_with_guard_in_frozen_population():
    summary = run_frozen_control()["summary"]
    median = [summary[str(g)]["whole_record_influence_l2_median"]
              for g in (0, 80, 320, 640)]
    assert all(0 < y < x for x, y in zip(median, median[1:]))
    assert all(summary[str(g)]["nonzero_whole_record_instances"] == 32
               for g in (0, 80, 320, 640))
    assert all(summary[str(g)]["direct_raw_epoch_overlap_instances"] == 0
               and summary[str(g)]["fold_masked_influence_l2_max"] == 0
               for g in (0, 80, 320, 640))


def test_model_does_not_mislabel_synthetic_pairs_as_independent_patients():
    result = run_frozen_control()
    assert result["precommitted_seed_count"] == 32
    assert len(result["paired_detailed_rows"]) == 128
    assert "no clinical datasets" in result["source"]
    assert "real EEG data or independent human participants" in result["not_claimed"]
    # Deterministic replay contract; no post-hoc sample dropping.
    assert result == run_frozen_control()
    json.dumps(result, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"guard_samples": -1},
    {"guard_samples": 1700},
    {"guard_samples": 1.5},
    {"guard_samples": 0, "test_interval": (1800, 1800)},
    {"guard_samples": 0, "test_interval": (2000, 5000)},
    {"guard_samples": 0, "burst_amplitude": 0},
    {"guard_samples": 0, "sample_rate_hz": np.nan},
])
def test_malformed_counterfactual_design_fails_closed(kwargs):
    with pytest.raises(ValueError):
        counterfactual_filter_witness(0, **kwargs)


def test_prospective_denominator_rejects_posthoc_subsetting():
    with pytest.raises(ValueError, match="prospective"):
        run_frozen_control(seeds=range(5))
    with pytest.raises(ValueError, match="prospective"):
        run_frozen_control(guards=(0, 80))
