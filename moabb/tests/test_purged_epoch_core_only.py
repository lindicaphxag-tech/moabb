"""Independent raw-interval CV core contract tests; no paradigm/evaluator edits.

Important: this detects only direct overlap of declared RAW epoch windows.
Filtering continuous MNE Raw before the split can still leak across windows.
"""
import numpy as np
import pandas as pd
import pytest

from moabb.evaluations.splitters import PurgedEpochKFold, epoch_interval_groups


def _metadata(starts, *, runs=None, width=500):
    return pd.DataFrame({
        "run": ["only"] * len(starts) if runs is None else runs,
        "event_sample": starts,
        "epoch_n_samples": [width] * len(starts),
    })


def _has_direct_overlap(train, test, md):
    run = md["run"].to_numpy()
    start = md["event_sample"].to_numpy()
    end = start + md["epoch_n_samples"].to_numpy()
    return any(
        run[i] == run[j] and start[i] < end[j] and start[j] < end[i]
        for i in train for j in test
    )


def test_scikit_learn_compatible_raw_epoch_purge_and_complete_test_partition():
    md = _metadata(np.arange(30) * 100)
    y = np.tile([0, 1], 15)
    splitter = PurgedEpochKFold(n_splits=5)
    folds = list(splitter.split(np.zeros(len(md)), y=y,
                                groups=epoch_interval_groups(md)))
    assert len(folds) == 5
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(30))
    for train, test in folds:
        assert not set(train).intersection(test)
        assert not _has_direct_overlap(train, test, md)
        assert set(y[train]) == {0, 1}
        assert set(y[test]) == {0, 1}
        assert 0 < len(train) < 30
    assert splitter.get_n_splits() == 5
    metadata = splitter.get_metadata()
    assert metadata["n_purged_overlap"] > 0
    assert 0 < metadata["purge_fraction"] < 1


def test_exact_half_open_boundary_must_be_retained_in_training():
    # A window [0, 100) does NOT overlap [100, 200).
    mask = PurgedEpochKFold._overlap_mask(
        np.array([0, 100, 199]), np.array([100, 200, 300]),
        np.array([100]), np.array([200])
    )
    assert mask.tolist() == [False, True, True]


def test_run_identity_prevents_cross_recording_false_purge():
    md = _metadata(np.r_[np.arange(16) * 800, np.arange(16) * 800],
                   runs=["A"] * 16 + ["B"] * 16,
                   width=500)
    splitter = PurgedEpochKFold(n_splits=4)
    folds = list(splitter.split(np.zeros(len(md)), groups=epoch_interval_groups(md)))
    assert len(folds) == 4
    # The independent runs reuse the SAME numeric sample coordinates.
    # With no intra-run overlap, every train is n minus test (no false purge).
    for train, test in folds:
        assert not _has_direct_overlap(train, test, md)
        assert len(train) == 24
        assert len(test) == 8


def test_chronological_blocks_not_dataframe_row_order():
    md = _metadata([1100, 300, 1500, 700, 0, 1900, 400, 1600, 800,
                    1200, 100, 2000, 1700, 500, 900, 1300, 200, 1400,
                    1800, 600], width=300)
    folds = list(PurgedEpochKFold(n_splits=4).split(
        np.zeros(len(md)), groups=epoch_interval_groups(md)))
    order = np.argsort(md["event_sample"].to_numpy())
    test_blocks = []
    for _, test in folds:
        positions = sorted(np.searchsorted(np.sort(md.event_sample), md.event_sample.iloc[test]))
        assert positions == list(range(positions[0], positions[-1] + 1))
        test_blocks.extend(positions)
    assert sorted(test_blocks) == list(range(20))


def test_absence_or_malformation_of_true_epoch_timing_refused():
    splitter = PurgedEpochKFold(n_splits=2)
    with pytest.raises(ValueError, match="true run/event timing"):
        list(splitter.split(np.zeros(8)))
    with pytest.raises(ValueError, match="shape"):
        list(splitter.split(np.zeros(8), groups=np.ones((8, 2))))
    md = _metadata(np.arange(8) * 100, width=100)
    with pytest.raises(ValueError, match="missing"):
        epoch_interval_groups(md.drop(columns="event_sample"))
    groups = epoch_interval_groups(md)
    groups[0, 1] = 0.5
    with pytest.raises(ValueError, match="finite integers"):
        list(splitter.split(np.zeros(8), groups=groups))
    groups = epoch_interval_groups(md)
    groups[0, 2] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        list(splitter.split(np.zeros(8), groups=groups))


def test_fails_closed_when_stratification_cannot_support_every_fold():
    md = _metadata(np.arange(10) * 600, width=500)
    y = np.array([0] * 8 + [1] * 2)
    with pytest.raises(ValueError, match="does not contain every class"):
        list(PurgedEpochKFold(n_splits=5).split(
            np.zeros(10), y, groups=epoch_interval_groups(md)))


def test_explicitly_out_of_scope_continuous_filter_leakage():
    doc = PurgedEpochKFold.__doc__
    assert "does NOT guarantee independence of already-filtered" in doc
    assert "IIR filter" in doc
