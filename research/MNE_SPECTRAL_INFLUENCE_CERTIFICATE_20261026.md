# Tight native-MNE preprocessing influence certificate — research note

**2026-10-26. Status:** independently executed fixed-filter synthetic experiment; code is in [mne_spectral_influence_certificate.py](./mne_spectral_influence_certificate.py). This is **not** merged external upstream code, a general EEG leakage theorem or a real-data SOTA claim. Keep separate from [NeuroTechX/moabb#1329](https://github.com/NeuroTechX/moabb/pull/1329).

## Correctness statement

For fixed *linear* finite-recording filtering `y=Hx`, held-out raw index set `T`, train-output positions `A`, and intervention `delta` with `||delta||_2<=rho`:

```text
C = P_A H P_T
sup_{||delta||_2 <= rho} ||C delta||_2 = rho * ||C||_2
                                         <= rho * ||C||_F
```

The first equality is exact in finite-dimensional linear algebra; its maximizer is the top right singular vector. The Frobenius norm is a conservative bound. Crucially, the guarantee is bounded *signal influence*, **not** formal independence of arbitrary labels, learned transform state or clinician-level EEG predictions.

## Executed native MNE probe

- MNE 1.11.0, NumPy 2.3.5, sample rate 250 Hz.
- Zero-phase 1–40 Hz FIR and IIR filters, fixed 14-second recording.
- Ten held-out input samples [4.00,4.04) s; 2-second training output window starting after one of six candidate gaps.
- Obtain `C` column-by-column via native `mne.filter.filter_data` unit impulses. Compare its largest singular value to `||C||_F`.
- Independently apply its maximizing right singular vector to a seeded synthetic record **only in the held-out indices**. The observed train-output L2 change matches the predicted spectral norm within `max(1e-11, 1e-9*||C||_2)`. Training raw input samples remain bitwise identical.

| Gap (s) | FIR spectral | FIR Frobenius | IIR spectral | IIR Frobenius |
|---:|---:|---:|---:|---:|
| 0.04 | 0.11391827 | 0.13813777 | 0.16189096 | 0.16224108 |
| 0.30 | 0.06652906 | 0.06653585 | 0.04720989 | 0.04736701 |
| 0.80 | 0.00671884 | 0.00674182 | 0.01348070 | 0.01351959 |
| 1.50 | 0.00117089 | 0.00119128 | 0.00256996 | 0.00257838 |
| 2.50 | ~0 | ~0 | 0.00022737 | 0.00022849 |
| 4.00 | ~0 | ~0 | 0.00000815 | 0.00000817 |

**Exploratory thresholds, not pre-registered primary endpoints:** at intervention-L2 budget `0.12`, the earliest FIR gap in the candidate grid is **0.04 s** under the tight spectral certificate versus **0.30 s** under the Frobenius bound: **0.26 seconds of nominal raw time is retained**. IIR has **no difference**, both choose 0.30 s. For budget `0.01`, neither filter gains anything (FIR chooses 0.8 s, IIR 1.5 s using either bound). This is a selective, mechanism-limited positive result with important null cases, **not** a measured gain in EEG trials, task AUC or study acceptance probability.

## Why this may be useful but is not yet a high-tier algorithm result

1. The exact spectral certificate is ordinary finite-dimensional operator theory; originality would need to come from its **new protocol/algorithm** for real fold-aware EEG preprocessing with independently measured utility, not from presenting the SVD identity as novel.
2. Numerical zero below floating-point precision is not a proof of true strict isolation. Finite recording length, padding, filter coefficients, nonlinear transforms, learned normalization, train/test labels, and different MNE versions can all change the influence.
3. The source directly computes the dense influence matrix, and is appropriate for mechanistic small-window confirmation, **not** efficient 100k-epoch deployment. A scalable implicit Jacobian-vector / block operator norm estimator requires separate proof, stopping rule and cost analysis.
4. Next hard gate: run actual MOABB paradigm/source preprocessing, compare embargo/purge/fold-aware baselines on independent EEG cohorts, report train coverage and independent task metrics with uncertainty and runtime. Do not merge this experimental feature into the maintainers' requested small #1329.
