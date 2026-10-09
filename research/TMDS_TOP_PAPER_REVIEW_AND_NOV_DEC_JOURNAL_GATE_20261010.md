# Research upgrade gate: Transferability-Matched Discriminant Shrinkage (TMDS)

**As of 2026-10-10. Candidate EEG method, not a proven top-journal contribution. No empirical AUC improvement is claimed until the public EEG first cohort is fully executed, audited and published.**

## 1. Technical upgrade versus the old EEG 'regularization mismatch' observation

The prior Default Regularisation research line reported a potential mismatch between pooled covariance-optimal shrinkage and cross-subject discriminability. Such an observation is not, by itself, a new learning method. We implement a falsifiable candidate method **TMDS** in [transfer_consistency_shrinkage.py](transfer_consistency_shrinkage.py).

For each labeled source participant s, estimate its class difference vector δ_s and within-class covariance Σ_s on source-standardized log-spectral EEG features. To prevent participants with many epochs dominating, use equal-subject contributions. Form the average within-subject covariance Σ and diagonalize it as U diag(σ_j) Uᵀ. Project δ_s into each spectral coordinate u_j; define a *chance-corrected source consistency* score

    q_j = clip_0^1{ [K*(mean_s δ_sj)^2 - mean_s δ_sj²] / [(K-1)*mean_s δ_sj² + eps] }

The zero-shot discriminant is

    w = U diag(1/[σ_j + λ(1+α(1-q_j))]) Uᵀ mean_s δ_s.

Here λ>0 is the baseline ridge regularizer, and α≥0 penalizes directions whose discriminative sign/magnitude do not transfer between training subjects. α=0 is an identical-representation, identical-base-λ ablation. K counts **independent source subjects**, not EEG epochs. The coherent direction score is a heuristic estimate, **not** a proven prediction interval, Bayes posterior or mathematically optimal OOD regularizer.

### Nested zero-leakage selection
For each completely heldout target participant, select λ∈{0.03, 0.3, 3} and α∈{0,1,4} by leave-*another source subject*-out AUC inside the remaining training participants. Refit the model only on those training subjects. Never use target test labels in scaling, covariance, q_j, hyperparameter selection or calibration. The test labels are used *only* for one final AUC report per independently heldout participant.

The first implementation has **8 adversarial leakage tests**, including flipping the outer target's entire label vector and verifying that the selected source-only parameters and source performance do not change. It refuses source populations too small for nested validation and prevents weighting one subject more merely by duplicating their trials.

## 2. Genuine real-EEG pilot must pass before any statement of improvement

[Exact registered protocol](EEGBCI_FIRST8_PREOUTCOME_20261010.json) freezes PhysioNet EEGBCI subjects 1–8, left/right motor-imagery runs [4,8,12] (NOT hands-versus-feet runs), 8–30 Hz bandpass, 0.5–2.5 sec after cue, channel log-Welch bandpower features [8,12], [12,20], [20,30] Hz and all eight outer heldout subjects.

The initial pre-outcome CI attempt [#38003097900](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/38003097900) FAILED before EEG data execution on a test error-message capitalization assertion (7/8 tests passed). This was repaired *before any EEG outcomes*, leaving the prereg and method untouched. [New exact-code physical-data run #38003196642](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/38003196642) is registered with source SHA frozen and first-stage no-leakage tests passed. Do **not** report data-stage successes until its actual complete run and all eight source EDF SHA256 values appear.

Primary comparisons, all on whole heldout **subject** AUC:
1. TMDS with source-only nested tuning.
2. Same log-PSD representation and same selected base ridge, with α=0 (within-model ablation).
3. Ledoit–Wolf automatic shrinkage LDA on the **same log-PSD features**.
4. CSP + regularized LDA on the **same filtered EEG epochs** but a different spatial representation (standard stronger EEG baseline; never imply equal representation).

Report all eight subject AUC triplets/quartets, average, paired subject-level difference, exact exploratory subject sign-flip, fail cases and actual dataset files' SHA256. No epoch-level IID claims. Eight subjects on **one** PhysioNet cohort do not support a paper-scale cross-dataset superiority claim.

## 3. Against actual published research, not fictional 'L8/L9 researchers'

| Work | Published evidence standard and technique | Gap to close |
|---|---|---|
| Barachant et al., IEEE TBME 2012, *Multiclass BCI Classification by Riemannian Geometry* (DOI: 10.1109/TBME.2011.2172210) | Spatial covariance, Riemannian MDM and tangent-space LDA; tested against CSP+LDA in BCI IV 2a | Implement exact spatial-covariance MDM/tangent-space and/or reliable MOABB pipelines as true strong rivals |
| An et al., MICCAI 2024, *Subject-Adaptive Transfer Learning Using Resting-State EEG Signals* | Cross-subject calibration using resting-state target EEG; public code, three benchmarks | Distinguish the zero-target-label/no-resting-state setting, still compare total sensing requirements fairly |
| Wang et al., IEEE TBME 2025, *TFTL: Task-Free Transfer Learning Strategy for Cross-Subject and Cross-Dataset MI-BCI* | Cross-subject AND cross-dataset approach published in an established biomedical-engineering venue | A single source dataset and eight heldout people fall short |
| *Multiple Regularized Knowledge Transfer Learning* (EAAI, 2026, DOI: 10.1016/j.engappai.2025.113397) | Riemannian tangent-space/MMD, confidence-aware pseudo-labels, three MI datasets and eight SOTA baselines | More evidence and stronger algorithms required before claiming novelty or major performance jump |

These are *documented research outcomes*. An invented L8/L9 score or a count of green CI jobs is not a meaningful substitute.

## 4. 2026 November–December journal decision and kill-switches

**Primary if the actual data justify it:** Biomedical Signal Processing and Control (BSPC). Independent CAS 2025 information identifies it as **medicine category 2区**, subject to final institutional CAS edition verification. Publisher's historical 141-day submission-to-acceptance makes **Nov/Dec 2026 submission realistic as a TARGET, acceptance by Dec highly uncertain**.

**CCF-labeled alternative for a more general algorithm:** Neurocomputing, **CCF C** in the currently available CCF international A/B/C catalog, but publisher's displayed historical time submission-to-acceptance is ~218 days. Not a fast guaranteed CCF success. Empirical Software Engineering (CCF B) is appropriate to the distinct proof-carrying review project, NOT to EEG bandpower LDA. IP&M (CCF B) would require a real information-retrieval/processing problem and cannot be used solely to obtain a B classification for arbitrary biomedical EEG.

For either EEG journal, the **submission gate** is not eight-subject pilot:
- at least 3–4 distinct publicly documented EEG task/domain datasets, independently defined labels/electrode crosswalk and fully source-locked LOSO; new test subjects unseen during method revision;
- strong covariance/riemannian, CSP/FBCSP, shrinkage-LDA and published transfer methods with equal target supervision, preprocessing and tuning budgets;
- nontrivial consistently positive per-subject paired AUC and calibration robustness, or a mathematically documented negative-transfer avoidance mechanism with appropriate robust bounds;
- explicit absolute runtime/feature size, parameter sensitivity, subject-blocked uncertainty intervals, prior art differentiation, ethics/licensing, trained code/seed hash and original dataset provenance;
- if the pilot shows α=0 selected or CSP beats TMDS, retain that result and redesign on a **separate future development cohort**. Do not silently retune after seeing the heldout results and call the same eight subjects confirmatory.

**Recommendation:** Aim for a defensible Nov/Dec submission of Default Regularisation or Two Laws when this gate is actually met. Preserve the dynamic controller proof-carrying recovery as the longer-horizon flagship, rather than delaying the EEG near-term manuscript for an unfinished robot policy-level claim.

This branch is author-controlled experimental research, not official MOABB upstream merged software, independent use, accepted paper or an L8/L9 graduation event.
