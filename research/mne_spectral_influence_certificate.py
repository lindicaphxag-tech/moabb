"""Research-only native MNE tight train/test influence certificate.

For a fixed finite-length linear preprocessing operator H and held-out raw
samples T, the worst-case L2 train-output change under ||delta_T||_2 <= rho
is rho * ||P_A H P_T||_2. The spectral norm is exact for this linear operator;
the Frobenius norm is only a conservative upper bound.

Not a clinical EEG result, all-pipelines certificate, or upstream PR.
"""
import json
import warnings
import mne
import numpy as np

SAMPLE_RATE = 250
N = 14 * SAMPLE_RATE
TEST = np.arange(int(4.0 * SAMPLE_RATE), int(4.04 * SAMPLE_RATE))
TRAIN_SAMPLES = 2 * SAMPLE_RATE
GAPS = (0.04, 0.3, 0.8, 1.5, 2.5, 4.0)
BUDGETS = (0.01, 0.12)


def run_filter(x, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(
            x, SAMPLE_RATE, 1, 40, method=method, phase="zero", verbose=False
        )


def main():
    results = {}
    for method in ("fir", "iir"):
        columns = []
        for index in TEST:
            impulse = np.zeros(N)
            impulse[index] = 1.0
            columns.append(run_filter(impulse, method))
        influence = np.column_stack(columns)
        conditions = []
        rng = np.random.default_rng(20261026)
        signal = 0.04 * rng.normal(size=N)
        baseline = run_filter(signal, method)

        for gap in GAPS:
            begin = int(round((4.04 + gap) * SAMPLE_RATE))
            train = slice(begin, begin + TRAIN_SAMPLES)
            assert begin > max(TEST)
            cross = influence[train]
            _, singular_values, right_vectors = np.linalg.svd(
                cross, full_matrices=False
            )
            exact = float(singular_values[0])
            conservative = float(np.linalg.norm(cross, "fro"))
            assert exact <= conservative + 1e-14
            delta = right_vectors[0].copy()
            changed = signal.copy()
            changed[TEST] += delta
            np.testing.assert_array_equal(signal[train], changed[train])
            observed = float(
                np.linalg.norm(run_filter(changed, method)[train] - baseline[train])
            )
            assert abs(observed - exact) <= max(1e-11, exact * 1e-9)
            conditions.append(dict(
                gap_seconds=gap,
                spectral_l2_worst_case=exact,
                frobenius_l2_upper_bound=conservative,
                tightness_ratio=exact / conservative if conservative else 0.0,
                adversarial_observed_l2=observed,
            ))

        feasible = {}
        for budget in BUDGETS:
            for selector, key in (
                ("tight_spectral", "spectral_l2_worst_case"),
                ("conservative_frobenius", "frobenius_l2_upper_bound"),
            ):
                valid = [
                    x["gap_seconds"] for x in conditions if x[key] <= budget
                ]
                feasible[f"{selector}_budget_{budget}"] = min(valid) if valid else None
        results[method] = dict(conditions=conditions, feasible_grid_gaps=feasible)

    assert results["fir"]["feasible_grid_gaps"]["tight_spectral_budget_0.12"] == 0.04
    assert results["fir"]["feasible_grid_gaps"]["conservative_frobenius_budget_0.12"] == 0.3
    assert results["fir"]["feasible_grid_gaps"]["tight_spectral_budget_0.01"] == 0.8
    assert results["iir"]["feasible_grid_gaps"]["tight_spectral_budget_0.01"] == 1.5
    print(json.dumps({
        "experiment": "native fixed-MNE spectral norm certificate",
        "numpy": np.__version__,
        "mne": mne.__version__,
        "test_sample_count": len(TEST),
        "train_window_seconds": 2,
        "budgets_exploratory": BUDGETS,
        "results": results,
        "scope": "fixed linear finite-record filtering; no EEG performance conclusion",
    }, indent=2))


if __name__ == "__main__":
    main()
