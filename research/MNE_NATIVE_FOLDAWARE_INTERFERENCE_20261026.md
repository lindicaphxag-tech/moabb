# Native MNE preprocessing counterfactual (2026-10-26)

**Status:** fully executed on synthetic data with native MNE 1.11.0 + NumPy 2.3.5, CPU. This is a research-only witness of signal dependence, **not** a new upstream PR, a proof of EEG predictive bias, or an accuracy benchmark.

Executable: [mne_foldaware_causal_probe.py](./mne_foldaware_causal_probe.py). Strictly separate from opt-in minimal splitter [NeuroTechX/moabb #1329](https://github.com/NeuroTechX/moabb/pull/1329), which upstream maintainers explicitly requested to keep small.

## Preregisterable intervention

- Make a fixed 24 s, 250 Hz synthetic signal consisting of seeded white noise and a 7 Hz sinusoid.
- Add a Hanning pulse of amplitude 3 **only in the held-out raw test interval [5.0, 5.2) seconds**.
- Test starts are always earlier than training: candidate train interval is [5.2 + gap, 11.2 + gap), 6 s long; gaps 0.3, 0.8, 1.5 and 2.5 seconds. Raw sample sets never intersect, and train-region raw arrays are **exactly identical** before/after intervention.
- Process with `mne.filter.filter_data`, zero-phase `FIR` and `IIR` bandpass 1–40 Hz. Compare the training outputs when the **whole continuous recording** is filtered, versus when the train segment is filtered separately without test raw samples. Same coefficients/options/inputs are used within each arm.
- A control fails if local training-alone output changes from the held-out intervention; this remained exactly zero for all eight combinations.

## Observed maximum train-output change from test-only intervention

| Held-out-to-train gap (s) | Full-record zero-phase FIR | Full-record zero-phase IIR | Train-only FIR/IIR |
|---:|---:|---:|---:|
| 0.3 | 0.1940795666634191 | 0.11336302122121972 | 0 |
| 0.8 | 0.015426693949763516 | 0.033588385383311964 | 0 |
| 1.5 | 0.004221045786469632 | 0.007016157767425937 | 0 |
| 2.5 | 3.3306690738754696e-16 (roundoff) | 0.000584179941755178 | 0 |

Values are in arbitrary synthetic signal amplitude units, **not classifier accuracy, clinical EEG amplitudes or a proven population effect**. Different preprocessing filters, time windows and boundary policies can change them.

### Mechanism: raw overlap vs transformation support

For a fixed linear full-record transform `H`, raw index nonoverlap `A ∩ T = ∅` does not imply zero dependency `P_A H P_T = 0`. For any test-only intervention `δx_T`, the relevant influence is `P_A H P_T δx_T`. Independent train-only filtering excludes this input by construction, though it changes edge/boundary conditions and **is not guaranteed to retain the same spectral response near boundaries**.

This experiment demonstrates that a raw-epoch-purging splitter alone cannot certify full post-preprocessing isolation in the native MNE zero-phase filtering path. It does **not** establish which preprocessing policy is optimal, quantify bias in standard MOABB evaluation, or automatically imply that no raw-window purge has value.

## Higher-grade next gates (not yet met)

1. Identify the exact MOABB continuous-Raw preprocessing/filtering configuration for each target paradigm and ensure signals are produced with the native `BaseParadigm` pathway, not an approximate stand-in.
2. Separate test-only interventions from training-fit preprocessors (ICA, scaling, learned transforms), fold partitions and hidden-state leakage; derive a transformation-support or influence-budget contract.
3. Compare on real held-out EEG records using a predetermined primary endpoint, multiple subject/session recordings, stratified bootstrap CIs and fully independent baseline pipelines.
4. Prove and verify computational/coverage costs; quantify leakage mitigation without claiming EEG performance gains when none are observed.

**Upstream scope boundary:** keep this out of #1329 unless the maintainer explicitly requests a larger end-to-end preprocessing change.

## Stronger native influence-budget gate (2026-10-26, subsequent experiment)

New independent executable: [mne_native_linear_influence_gate.py](./mne_native_linear_influence_gate.py).

For a **fixed native MNE zero-phase filtering configuration**, construct the
finite-recording cross-dependence matrix `C=P_train H P_test` by applying one
unit impulse to each of 10 held-out test samples, and observing the resulting
full-recording filter output. An independently seeded, **unit-L2 test-only
intervention** checks that `C δx` matches the actual full-recording change.

At 250 Hz, [4.00,4.04) s test input, and 2 s training output windows, we
obtained linear-replay relative errors **4.36e-15 for FIR** and **6.43e-13 for
IIR**. The Frobenius bound `||C δx||_2 <= ||C||_F ||δx||_2`
held for all 12 tested filter/gap cases, up to `1e-12` numerical tolerance.

| Gap (s) | FIR `||C||_F` | IIR `||C||_F` |
|---:|---:|---:|
| 0.04 | 0.13813777 | 0.16224108 |
| 0.30 | 0.06653585 | 0.04736701 |
| 0.80 | 0.00674182 | 0.01351959 |
| 1.50 | 0.00119128 | 0.00257838 |
| 2.50 | below 1e-15 | 0.00022849 |
| 4.00 | below 1e-15 | 0.00000802 |

For an **exploratory**, unit-norm intervention budget of `0.01`, the smallest
gap **within the tested candidate grid** was **0.8 s FIR** and **1.5 s IIR**.
This is a concrete example of filter-specific influence-budget selection; it
is **not** a global optimum, a pre-registered threshold, nor a measured
increase in real EEG training coverage or downstream classification quality.

### Generalization limits

- The matrix is constructed for *one finite-length* MNE filter operator and
  the specified test/train positions. Padding, recording length, channel
  mixing, time-varying processing, learned transforms, and metadata-dependent
  preprocessing can alter this operator.
- The Frobenius norm is a conservative matrix upper bound for this *fixed
  linear* transform. Measurement error and floating-point residuals near zero
  are not exact proofs of mathematical zero dependence.
- To claim a new research algorithm, compare this selector against strong
  gap-based, embargo, support-aware and fold-aware preprocessing baselines on
  held-out real EEG before proposing it upstream. Keep #1329 minimal.
