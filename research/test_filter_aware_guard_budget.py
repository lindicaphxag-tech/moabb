"""Independent real SciPy counterfactual oracle for source-dependent guard."""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from research.filter_aware_guard_budget import (
    cross_fold_filter_operator,
    select_filter_guard,
)


def _sos():
    return butter(4, [1, 30], fs=250, btype="bandpass", output="sos")


def test_discrete_cross_fold_operator_matches_actual_filtered_counterfactual():
    sos = _sos()
    n, a, b, lo, hi = 600, 270, 290, 190, 210
    operator = cross_fold_filter_operator(sos, n, (lo, hi), (a, b))
    assert operator.shape == (hi-lo, b-a)
    rng = np.random.default_rng(20261013)
    for _ in range(20):
        background = rng.standard_normal(n)
        delta = rng.standard_normal(b-a)
        second = background.copy()
        second[a:b] += delta
        direct = sosfiltfilt(sos, second)[lo:hi] - sosfiltfilt(
            sos, background
        )[lo:hi]
        oracle = operator @ delta
        np.testing.assert_allclose(direct, oracle, atol=3e-13, rtol=3e-11)
        assert np.linalg.norm(direct) <= (
            np.linalg.norm(operator, ord="fro") * np.linalg.norm(delta) + 3e-12
        )
    assert np.linalg.norm(operator, ord=2) <= (
        np.linalg.norm(operator, ord="fro") + 1e-13
    )


def test_filter_aware_guard_selects_first_admissible_amplification_budget():
    chosen, rows = select_filter_guard(
        _sos(), record_samples=4096, test_interval=(1800, 2120),
        training_epoch_samples=320, candidate_guards=(0, 80, 320, 640),
        max_amplification=.001,
    )
    assert [row.guard_samples for row in rows] == [0, 80, 320, 640]
    assert chosen is not None and chosen.guard_samples == 640
    assert all(row.spectral_gain_diagnostic <=
               row.frobenius_gain_bound + 1e-12 for row in rows)
    assert rows[-1].frobenius_gain_bound == pytest.approx(
        .0006676653535, rel=1e-4
    )
    assert all(rows[j].frobenius_gain_bound >
               rows[j+1].frobenius_gain_bound
               for j in range(3))
    assert not chosen.physical_independence_certified


def test_inadequate_guard_register_fails_closed_instead_of_claiming_no_leakage():
    choice, rows = select_filter_guard(
        _sos(), record_samples=4096, test_interval=(1800, 2120),
        training_epoch_samples=320, candidate_guards=(0, 80, 320),
        max_amplification=1e-4,
    )
    assert choice is None
    assert len(rows) == 3
    assert all(not row.admissible_for_declared_tolerance for row in rows)


@pytest.mark.parametrize("train,test", [
    ((2, 2), (10, 30)),
    ((0, 20), (10, 30)),
    ((50, 80), (50, 60)),
    ((30, 40), (30, 40)),
    ((-1, 10), (40, 50)),
    ((0, 10), (60, 80)),
])
def test_invalid_or_overlapping_windows_are_rejected(train, test):
    with pytest.raises(ValueError):
        cross_fold_filter_operator(_sos(), 64, train, test)


def test_guard_register_cannot_be_changed_silently_or_exceed_record_bounds():
    opts = dict(
        sos=_sos(), record_samples=4096, test_interval=(1800, 2120),
        training_epoch_samples=320, max_amplification=.001,
    )
    for invalid in ((80, 0), (0, 0), (-1, 80), (0, 80.5), ()):
        with pytest.raises(ValueError):
            select_filter_guard(candidate_guards=invalid, **opts)
    with pytest.raises(ValueError):
        select_filter_guard(candidate_guards=(1600,), **opts)
