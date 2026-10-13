# Public EDF validation of zero-phase IIR cross-fold source dependence

**Date:** 2026-10-13. Frozen design + completed original source-data experiment. An entirely separate original-file SHA-256/redownload audit is an additional reproducibility gate and must be marked independent only when successful. This is a real public EEG signal *counterfactual interference* test, not a new clinical result, classifier accuracy test or proof of label leakage.

## Reproducible frozen design

- The unaltered prospective protocol is [PHYSIONET_EDF_FOLD_FILTER_PREREG_20261013.json](../PHYSIONET_EDF_FOLD_FILTER_PREREG_20261013.json), Git blob `8dca2a3c9f287967812dd9a5bec2a9d1ba90c120`. Frozen on Oct 13 before data outcomes.
- [Source implementation](../physionet_real_edf_filter_dependence.py), modified only to cast a NumPy `int64` to Python `int` when serializing the file's sample count. First [run](https://github.com/lindicaphxag-tech/moabb/actions/runs/38336698968) failed due JSON serialization after reading the first actual EDF; the succeeding second run preserves every original dataset/protocol/seed/filter parameter.
- [Complete original successful workflow](https://github.com/lindicaphxag-tech/moabb/actions/runs/38336887369), head `7656ded966dabb473d9118a25441868b84e9648f`, actual 8 EDF recordings, original file SHA-256 fingerprints and 64 registered nonoverlapping same-recording EEG counterfactuals.
- Raw PhysioNet [EEGBCI (eegmmidb/1.0.0)](https://physionet.org/content/eegmmidb/1.0.0/): subject IDs S001–S004, each runs R04 and R08, C3 and C4 channels. Four **independent participant clusters**, eight EDF files, sixteen subject-run-channel measurements at each of four different guards. No training, labels, classifiers, optimization or hyperparameter tuning on these subjects.
- Per record, for each C3 and C4: select source samples [9600,9920), create a *synthetic polarity inversion* only inside that held-out source interval, apply fixed fourth-order 1–30 Hz Butterworth zero-phase SOS filtering to the WHOLE recording both before and after perturbation. Compare filtered training window of 320 samples ending at offsets 0,80,320,640 before held-out region. The raw train and heldout windows are disjoint in all 64 cells. Negative-control copies zero the intervened source interval prior to the filtering operation in both worlds.

## Original results

| Guard samples | Absolute train-filter L2 disturbance median | Relative L2 transfer gain median | Nonzero influence observations | Nonzero masked control |
|---:|---:|---:|---:|---:|
| 0 | 8.0513138314e-05 | 0.06608612005 | 16 / 16 | 0 / 16 |
| 80 | 1.6136919595e-05 | 0.01381811035 | 16 / 16 | 0 / 16 |
| 320 | 5.5106793282e-07 | 0.00042380443 | 16 / 16 | 0 / 16 |
| 640 | 5.3218958406e-09 | 0.00000426660 | 16 / 16 | 0 / 16 |

Unit of absolute L2 is the filtered EEG signal's native MNE volts under the exact C3/C4 320-sample window, no independent calibration to physiological artifact severity. Relative L2 gain is `||filtered_test_intervention_effect_on_train||_2 / ||raw_test_intervention||_2`. These are not classification accuracy percentages or probability guarantees.

The sensitivity values are **conditional on a synthetic intervention**, signal recording, filter and window geometry; they must not be extrapolated to naturally occurring task leakage, causal label leakage, average EEG artifact prevalence or classification assessment bias. The effect being nonzero is structurally expected for IIR filtering; it is not a mathematical surprise or theorem of first discovery.

## Next exact scientific falsification gate

1. [Independent audit code](../audit_physionet_real_edf_filter_dependence.py): redownload all original EDFs, check SHA-256 versus original complete artifacts, independently derive filtered perturbation as `F(delta)` using linearity (instead of subtraction `F(x+delta)-F(x)`), and recompute all 64 original condition values. [Independent workflow](https://github.com/lindicaphxag-tech/moabb/actions/workflows/physionet-independent-real-edf-audit.yml) is a separate gate. The first attempt had only a download-artifact *path mismatch*, not a measurement error; a corrected workflow sets `merge-multiple: true`. **Do not label it passed before checking its successful run.**
2. Test actual fold-level EEG decoding under the same public data and *leakage-safe* within-record preprocessing only where there is a defensible causal train/test policy. Compare baseline models fairly and report changes in fold assignment, retained sample numbers and participant-wise uncertainty. Do not use original test labels or future test samples during training preprocessing.
3. Test representative realistic source perturbation amplitudes and filter choices, disclose negative cases, compare against the source-specific maximum-singular-value guard, and measure wall-clock/GPU/CPU cost.
4. Seek MOABB maintainer agreement for a **narrow, honestly named, opt-in** API only if the real classifier decision benefit is supported. Existing [MOABB #1235](https://github.com/NeuroTechX/moabb/pull/1235) protects extraction-window overlap, not arbitrary whole-record IIR dependence; mixing its contract with post-filter independence would be scientifically false.

**Outcome stage:** credible public original-data measurement (raw-source hashes and all 64 frozen conditions); pending external replication, downstream classifier utility, peer-reviewed originality and official upstream adoption.
