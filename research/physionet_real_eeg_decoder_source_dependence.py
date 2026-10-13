"""Source-counterfactual sensitivity of REAL motor-imagery EEG decoding.

The prospective protocol fixes event/feature/split/classifier choices. This
assesses whether a heldout EEG source perturbation contaminates train features
through whole-record IIR and can change out-of-sample decoder decisions.
NO natural label leakage, new clinical accuracy improvement or external
human cohort is asserted. All raw EDF content remains on the CI runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from urllib.request import urlopen

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROTOCOL = Path("research/PHYSIONET_EEG_DECODER_CONTRACT_PREREG_20261013.json")
PROTOCOL_BLOB = "25501f7fa3eacc460f4baa116878ca1ec3db947f"


def frozen_protocol():
    actual = subprocess.check_output(
        ["git", "hash-object", str(PROTOCOL)], text=True
    ).strip()
    if actual != PROTOCOL_BLOB:
        raise RuntimeError("Classifier source protocol was altered after freeze")
    obj = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (
        obj["status"] != "FROZEN_BEFORE_CLASSIFIER_OUTCOME"
        or obj["participant_ids"] != [1, 2, 3, 4]
        or obj["run_ids"] != [4, 8]
        or obj["channels"] != ["C3", "C4"]
        or obj["labels"] != {"T1": 0, "T2": 1}
        or obj["epoch_after_event_seconds"] != [0.5, 2.5]
    ):
        raise RuntimeError("Unexpected data, class or timing contract")
    return obj


def get_original_edf(participant, run, cache):
    subdir = f"S{participant:03d}"
    name = f"{subdir}R{run:02d}.edf"
    path = cache / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = (f"https://physionet.org/files/eegmmidb/1.0.0/"
               f"{subdir}/{name}")
        temp = path.with_suffix(".tmp")
        try:
            with urlopen(url, timeout=90) as src, open(temp, "wb") as dest:
                while data := src.read(1024 * 1024):
                    dest.write(data)
            if temp.stat().st_size < 2048:
                raise RuntimeError("Downloaded EEG file is truncated")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def event_windows(raw, spec):
    # Explicit names -> standard MNE event codes; diagnostic class is code-1.
    mne.datasets.eegbci.standardize(raw)
    sfreq = float(raw.info["sfreq"])
    if sfreq != spec["raw_sfreq_hz"]:
        raise RuntimeError("EEG file deviated from registered 160 Hz raw source")
    ev, _ = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR"
    )
    if ev.shape[0] < 9:
        raise RuntimeError("Too few actual EEG motor imagery epochs")
    ev = ev[np.argsort(ev[:, 0], kind="stable")]
    starts = ev[:, 0] + int(round(.5 * sfreq))
    stops = ev[:, 0] + int(round(2.5 * sfreq))
    windows = [(int(a), int(b)) for a, b in zip(starts, stops)]
    if any(a < 0 or b > raw.n_times or b-a != 320
           for a, b in windows):
        raise RuntimeError("Out-of-bounds original motor imagery epoch")
    if len({a for a,b in windows}) != len(windows):
        raise RuntimeError("Non-unique raw EEG event timestamps")
    y = (ev[:, 2] - 1).astype(int)
    n_train = len(y) - math.ceil(len(y) / 3)
    if n_train < 5 or len(set(y[:n_train])) != 2 or len(set(y[n_train:])) != 2:
        raise RuntimeError("Class-balanced prospective temporal split impossible")
    tr, te = windows[:n_train], windows[n_train:]
    if any(max(a, c) < min(b, d) for a, b in tr for c, d in te):
        raise RuntimeError("Direct raw train/test epoch window overlap")
    return windows, y, n_train


def band_features(filtered, windows, fs=160):
    # Input shape channels x recording samples; compute 4 fixed log-power
    # features per event from ORIGINAL registered 320-sample EEG crops.
    n = 320
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    bands = [(8., 12.), (13., 30.)]
    taper = np.hanning(n)
    output = []
    for lo, hi in windows:
        signal = filtered[:, lo:hi]
        if signal.shape != (2, n):
            raise RuntimeError("Wrong original event EEG feature dimensions")
        ft = np.fft.rfft(signal * taper, axis=-1) / taper.sum()
        power = np.abs(ft) ** 2
        output.append([
            float(np.log(max(float(np.mean(power[k, (frequencies >= a)
                                               & (frequencies <= b)])), 1e-20)))
            for k in range(2) for a, b in bands
        ])
    features = np.asarray(output, dtype=np.float64)
    if not np.isfinite(features).all() or features.shape != (len(windows), 4):
        raise RuntimeError("Nonfinite true EEG spectral features")
    return features


def fit_predict(Xtrain, ytrain, Xtest):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=500,
            tol=1e-8, random_state=0
        ),
    )
    clf.fit(Xtrain, ytrain)
    return clf.predict_proba(Xtest)[:, 1]


def one_original_edf(spec, participant, run, cache):
    original, sha = get_original_edf(participant, run, cache)
    raw = mne.io.read_raw_edf(original, preload=True, verbose="ERROR")
    try:
        windows, label, ntr = event_windows(raw, spec)
        signal = raw.get_data(picks=spec["channels"])
        if signal.shape != (2, raw.n_times) or not np.isfinite(signal).all():
            raise RuntimeError("Original EDF does not contain valid C3/C4 EEG")
        sos = butter(4, [1., 30.], fs=160, output="sos", btype="bandpass")
        full_clean = sosfiltfilt(sos, signal, axis=-1)
        train_windows, test_windows = windows[:ntr], windows[ntr:]
        # Counterfactual "test-source changes, train labels fixed".
        corrupted = signal.copy()
        for lo, hi in test_windows:
            corrupted[:, lo:hi] *= -1
        for lo, hi in train_windows:
            if not np.array_equal(signal[:, lo:hi], corrupted[:, lo:hi]):
                raise RuntimeError("Intervention directly changed training raw sample")
        perturbed_filter = sosfiltfilt(sos, corrupted, axis=-1)
        # The heldout evaluation features remain ORIGINAL clean test data,
        # never the counterfactually perturbed test recording.
        Xclean = band_features(full_clean, windows)
        Xchanged = band_features(perturbed_filter, windows)
        ytrain, ytest = label[:ntr], label[ntr:]
        pclean = fit_predict(Xclean[:ntr], ytrain, Xclean[ntr:])
        pchanged = fit_predict(Xchanged[:ntr], ytrain, Xclean[ntr:])
        changed_pred = (pchanged >= .5).astype(int)
        clean_pred = (pclean >= .5).astype(int)
        deltas = np.abs(pclean-pchanged)

        # A negative-control only: no deployable classifier claim is made
        # for masking all test EEG segments to zeros in the train pipeline.
        masked0, masked1 = signal.copy(), corrupted.copy()
        for lo, hi in test_windows:
            masked0[:, lo:hi] = 0
            masked1[:, lo:hi] = 0
        if not np.array_equal(masked0, masked1):
            raise RuntimeError("Negative control counterfactual source mismatch")
        negative0 = sosfiltfilt(sos, masked0, axis=-1)
        negative1 = sosfiltfilt(sos, masked1, axis=-1)
        control_drift = float(np.linalg.norm(
            band_features(negative1, train_windows)
            - band_features(negative0, train_windows)
        ))
        if control_drift != 0.0:
            raise RuntimeError("Masked source control did not cancel exactly")

        train_feature_drift = np.linalg.norm(
            Xchanged[:ntr]-Xclean[:ntr], axis=1
        )
        if not np.isfinite(deltas).all():
            raise RuntimeError("Invalid supervised result / unstable probabilities")
        return {
            "participant": int(participant), "run": int(run),
            "edf_sha256": sha,
            "source_samples": int(raw.n_times),
            "train_event_count": int(ntr),
            "test_event_count": int(len(label)-ntr),
            "training_class_counts": [int(sum(ytrain == i)) for i in (0, 1)],
            "testing_class_counts": [int(sum(ytest == i)) for i in (0, 1)],
            "direct_train_test_epoch_raw_overlaps": 0,
            "source_perturbation_changes_train_raw_epochs": False,
            "masked_test_source_control_train_feature_l2": control_drift,
            "train_log_bandpower_feature_l2": float(np.linalg.norm(
                Xclean[:ntr]
            )),
            "train_log_bandpower_delta_l2": float(np.linalg.norm(
                Xchanged[:ntr]-Xclean[:ntr]
            )),
            "train_epochs_feature_l2_above_1e_12": int(sum(
                train_feature_drift > 1e-12
            )),
            "mean_heldout_probability_absolute_delta": float(np.mean(deltas)),
            "max_heldout_probability_absolute_delta": float(np.max(deltas)),
            "prediction_flips": int(sum(clean_pred != changed_pred)),
            "accuracy_clean": float(np.mean(clean_pred == ytest)),
            "accuracy_corrupt_test_source_only": float(
                np.mean(changed_pred == ytest)
            ),
            "genuine_motor_imagery_event_annotations": True,
            "adversarial_synthetic_heldout_signal_flip": True,
        }
    finally:
        raw.close()


def run(cache, output):
    spec = frozen_protocol()
    results = []
    for subject in spec["participant_ids"]:
        for run_id in spec["run_ids"]:
            row = one_original_edf(spec, subject, run_id, cache)
            print("REAL_EEG_DECODER_PAIRED_SOURCE",json.dumps(row,sort_keys=True),
                  flush=True)
            results.append(row)
    keys = {(r["participant"], r["run"]) for r in results}
    expected = {(s, r) for s in spec["participant_ids"] for r in spec["run_ids"]}
    if keys != expected or len(results) != 8:
        raise RuntimeError("Prospective 8-run supervised EEG denominator incomplete")
    summary = {
        "schema": spec["schema"],
        "preoutcome_preregister_git_blob": PROTOCOL_BLOB,
        "independent_actual_participants": 4,
        "original_public_edf_recordings": 8,
        "registered_within_record_comparisons": 8,
        "number_of_true_heldout_event_trials": sum(
            r["test_event_count"] for r in results
        ),
        "original_median_mean_heldout_probability_delta": float(
            np.median([r["mean_heldout_probability_absolute_delta"]
                       for r in results])
        ),
        "records_with_any_prediction_flip": sum(
            r["prediction_flips"] > 0 for r in results
        ),
        "total_predicted_label_flips": sum(r["prediction_flips"]
                                          for r in results),
        "mean_clean_accuracy_record_level_not_independent_subject": float(
            np.mean([r["accuracy_clean"] for r in results])
        ),
        "mean_perturbed_accuracy_record_level_not_independent_subject": float(
            np.mean([r["accuracy_corrupt_test_source_only"] for r in results])
        ),
        "no_classifier_accuracy_improvement_claim": True,
        "not_label_leakage_or_clinical_validation": True,
        "original_edf_sha256_and_all_results": results,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n",
                      encoding="utf-8")
    print("REAL_EEG_DECODER_COMPLETE",json.dumps({
        k:v for k,v in summary.items()
        if k!="original_edf_sha256_and_all_results"
    },sort_keys=True),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("/tmp/eegbci_decoder_original"))
    ap.add_argument("--out", type=Path,
                    default=Path("real_eegbci_decoder_source_dependence.json"))
    args=ap.parse_args()
    run(args.cache,args.out)
