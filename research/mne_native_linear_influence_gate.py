"""Native-MNE fixed-filter train/test influence-budget gate.

Independent finite-dimensional linear operator probe of MNE zero-phase FIR/IIR.
The fixed transform is measured using single-sample impulses at held-out test
locations, and independently checked against a random test-only intervention.
The matrix Frobenius bound is conservative for every test intervention with
unit L2 norm under this fixed linear transform and selected train positions.
Not an EEG accuracy result, an all-MOABB-pipelines theorem, or an upstream PR.
"""

import json
import warnings

import mne
import numpy as np


SFREQ = 250
DURATION_S = 14
TEST_START_S = 4.0
TEST_END_S = 4.04
TRAIN_WINDOW_S = 2
GAPS_S = (0.04, 0.3, 0.8, 1.5, 2.5, 4.0)
INFLUENCE_BUDGET = 0.01


def filter_signal(data, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(
            data, sfreq=SFREQ, l_freq=1.0, h_freq=40.0,
            method=method, phase="zero", verbose=False,
        )


def finite_operator(method, n_samples, test_indices):
    column_list = []
    for sample_index in test_indices:
        impulse = np.zeros(n_samples)
        impulse[sample_index] = 1.0
        column_list.append(filter_signal(impulse, method))
    return np.column_stack(column_list)


def evaluate():
    n = DURATION_S * SFREQ
    rng = np.random.default_rng(20261026)
    t = np.arange(n) / SFREQ
    before = 0.04 * rng.standard_normal(n) + 0.2 * np.sin(14 * np.pi * t)
    test = np.arange(int(TEST_START_S * SFREQ), int(TEST_END_S * SFREQ))
    delta = rng.standard_normal(len(test))
    delta /= np.linalg.norm(delta)
    changed = before.copy()
    changed[test] += delta
    receipts = {}

    for method in ("fir", "iir"):
        actual_whole_change = (
            filter_signal(changed, method) - filter_signal(before, method)
        )
        operator = finite_operator(method, n, test)
        predicted = operator @ delta
        rel_error = float(
            np.linalg.norm(predicted - actual_whole_change)
            / np.linalg.norm(actual_whole_change)
        )
        assert rel_error < 1e-9
        entries = []
        for gap in GAPS_S:
            train_start = int(round((TEST_END_S + gap) * SFREQ))
            train = slice(train_start, train_start + TRAIN_WINDOW_S * SFREQ)
            assert train.start > test[-1]
            np.testing.assert_array_equal(before[train], changed[train])
            cross = operator[train, :]
            upper = float(np.linalg.norm(cross, "fro") * np.linalg.norm(delta))
            effect = float(np.linalg.norm(actual_whole_change[train]))
            assert effect <= upper + 1e-12  # allow floating roundoff
            entries.append({
                "gap_s": gap,
                "cross_operator_frobenius": upper,
                "observed_train_l2_change": effect,
                "passes_budget": upper <= INFLUENCE_BUDGET,
            })
        feasible = [e["gap_s"] for e in entries if e["passes_budget"]]
        assert feasible, "Candidate gap grid contains no budget-feasible case"
        receipts[method] = {
            "independent_linear_replay_relative_error": rel_error,
            "minimum_budget_feasible_gap_s": min(feasible),
            "candidates": entries,
        }

    assert receipts["fir"]["minimum_budget_feasible_gap_s"] == 0.8
    assert receipts["iir"]["minimum_budget_feasible_gap_s"] == 1.5
    return {
        "scope": "fixed finite-recording native MNE zero-phase filter",
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "sfreq": SFREQ,
        "test_samples": len(test),
        "train_window_s": TRAIN_WINDOW_S,
        "max_unit_norm_intervention_l2_budget": INFLUENCE_BUDGET,
        "results": receipts,
        "limitations": (
            "Budget applies only to the fixed linear filter and specified "
            "finite recording/windows, not to nonlinear or learned pipelines, "
            "real EEG bias, or clinical utility"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
