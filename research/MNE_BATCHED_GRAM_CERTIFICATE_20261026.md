# Native MNE batched Gram certificate — source-verified research receipt (2026-10-26)

**Status:** exact-source locally executed, committed to research branch only. **Not** an external upstream contribution or a proven EEG classification improvement. Runnable: [mne_batched_gram_gate.py](./mne_batched_gram_gate.py).

## Mechanics

For a **fixed, finite-dimensional linear** continuous-signal transform \(y=Hx\), define train output indices \(A\), held-out raw input indices \(T\), and \(C=P_A H P_T\). An \(\ell_2\) bound on unseen test-only perturbations is

\[
\sup_{\|\delta x_T\|_2\le\rho}\|C\delta x_T\|_2
= \rho\|C\|_2
= \rho\sqrt{\lambda_{\max}(C^\top C)}.
\]

This identity is standard linear algebra, **not** a newly proved theorem. Its executable application here batches ten native MNE impulse probes as ten separate synthetic channels in **one** `mne.filter.filter_data` call, then builds the small Gram matrix \(C^\top C\), avoiding ten independent filter initialization calls. The independence of the synthetic channels is intentional; this does **not** model multichannel learned preprocessing.

### Direct numerical verification

- MNE **1.11.0**, NumPy **2.3.5**, CPU, 250 Hz, 14 s continuous record, 1–40 Hz zero-phase FIR/IIR bandpass.
- Held-out raw inputs: **ten samples** at 4.00–4.04 s; training output: **2 s** after one of six gaps (0.04, 0.30, 0.80, 1.50, 2.50, 4.00 s).
- The batched filter output is **elementwise identical** to ten separately filtered impulse inputs in this configuration (max absolute error **0.0**, both FIR and IIR).
- The top Gram eigenvalue produces exactly the same spectral norm as an **independent direct SVD oracle** to numerical tolerance in all 12 filter-gap cases.
- A separate seeded test-only intervention along the top right singular vector reproduces the predicted worst-case train-output L2 change (within \(\max(10^{-11},10^{-9}\|C\|_2)\)), without changing the train raw samples.
- The certified spectral bound never exceeds the Frobenius bound beyond numerical tolerance.

| Gap (s) | FIR spectral | FIR Frobenius | IIR spectral | IIR Frobenius |
|---:|---:|---:|---:|---:|
| 0.04 | 0.11391827 | 0.13813777 | 0.16189096 | 0.16224108 |
| 0.30 | 0.06652906 | 0.06653585 | 0.04720989 | 0.04736701 |
| 0.80 | 0.00671884 | 0.00674182 | 0.01348070 | 0.01351959 |
| 1.50 | 0.00117089 | 0.00119128 | 0.00256996 | 0.00257838 |

### Carefully warmed CPU timing

We **excluded first-run filter setup cost** by warming both implementations before 12 *alternating-order* comparisons. Single-machine median wall times:

| Filter | 10 sequential native-MNE calls | One batched native-MNE call | Observed speedup |
|---|---:|---:|---:|
| FIR | 22.45 ms | 3.54 ms | **6.34×** |
| IIR | 29.82 ms | 7.04 ms | **4.24×** |

These are run-specific setup/amortization numbers for ten impulses on one CPU; **not** a machine-independent algorithmic complexity breakthrough, a larger-cohort scaling law, or evidence of improved BCI performance. For many held-out samples, storing the batched probe matrix costs \(O(|T|N)\); the Gram matrix costs \(O(|T|^2)\). Memory-bounded blocking and realistic MNE preprocessing pipelines require independent development and validation.

## Advancement gate

Before an external methodological claim: perform hold-out runs over multiple native MNE filters and actual MOABB paradigms; specify an influence budget *before* evaluation; compare time/coverage/predictive endpoints against fixed embargo, raw-window purge, strong fold-aware preprocessing, and an efficient implicit-operator baseline; report confidence intervals and null/negative conditions. Do **not** modify the existing 3-file [MOABB #1329](https://github.com/NeuroTechX/moabb/pull/1329) while its maintainer-requested minimal feature is undergoing review.
