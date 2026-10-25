"""Reproducible MNE finite-signal interference control (research only).

Compare one-shot full-recording zero-phase band-pass preprocessing with an
independently preprocessed training-only segment. Intervene ONLY on a held-out
raw interval; verify raw-input identity for training and trace post-filter
influence. This is not an EEG accuracy benchmark or a proof for all pipelines.

Requires: numpy, mne. Tested on MNE 1.11.0, NumPy 2.3.5 (CPU).
"""

import json
import warnings

import mne
import numpy as np


SFREQ = 250
TEST_START = 5.0
TEST_END = 5.2
TRAIN_DURATION = 6.0
FILTER_MODES = ("fir", "iir")
GAPS = (0.3, 0.8, 1.5, 2.5)


def make_signal():
    n_samples = SFREQ * 24
    times = np.arange(n_samples) / SFREQ
    rng = np.random.default_rng(20261026)
    return 0.04 * rng.standard_normal(n_samples) + 0.2 * np.sin(
        2 * np.pi * 7 * times
    )


def filtered(data, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(
            data, sfreq=SFREQ, l_freq=1.0, h_freq=40.0,
            method=mode, phase="zero", verbose=False,
        )


def main():
    control = make_signal()
    perturbed = control.copy()
    test = slice(int(TEST_START * SFREQ), int(TEST_END * SFREQ))
    pulse = 3.0 * np.hanning(test.stop - test.start)
    perturbed[test] += pulse
    assert np.count_nonzero(control != perturbed) < test.stop - test.start

    receipts = []
    for mode in FILTER_MODES:
        whole_before = filtered(control, mode)
        whole_after = filtered(perturbed, mode)
        for gap in GAPS:
            train_start = int(round((TEST_END + gap) * SFREQ))
            train_stop = train_start + int(TRAIN_DURATION * SFREQ)
            train = slice(train_start, train_stop)
            assert test.stop < train.start
            np.testing.assert_array_equal(control[train], perturbed[train])
            whole_effect = whole_after[train] - whole_before[train]
            isolated_before = filtered(control[train], mode)
            isolated_after = filtered(perturbed[train], mode)
            isolated_effect = isolated_after - isolated_before
            global_max = float(np.max(np.abs(whole_effect)))
            local_max = float(np.max(np.abs(isolated_effect)))
            assert local_max == 0.0
            if gap == 0.3:
                assert global_max > 1e-5, (
                    "Expected MNE full-recording filtering to couple disjoint "
                    "raw intervals at 300 ms gap"
                )
            receipts.append({
                "filter_mode": mode,
                "heldout_to_training_gap_seconds": gap,
                "raw_train_inputs_exactly_identical": True,
                "full_record_train_max_abs_change": global_max,
                "full_record_train_l2_change": float(np.linalg.norm(whole_effect)),
                "isolated_train_max_abs_change": local_max,
            })
    print(json.dumps({
        "scope": "synthetic intervention, native MNE.filter.filter_data",
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "sample_rate_hz": SFREQ,
        "bandpass_hz": [1.0, 40.0],
        "test_interval_seconds": [TEST_START, TEST_END],
        "train_duration_seconds": TRAIN_DURATION,
        "records": receipts,
        "caveat": "Not proof of EEG predictive leakage magnitude or clinical impact",
    }, indent=2))


if __name__ == "__main__":
    main()
