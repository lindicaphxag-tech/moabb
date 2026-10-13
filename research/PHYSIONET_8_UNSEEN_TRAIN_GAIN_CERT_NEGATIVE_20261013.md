# Registered blind eight-person EEG decision-certificate study: the go/no-go gate FAILED

**Original study and independent audit date:** 2026-10-13  
**Consequence:** retain the mathematically valid implementation, **STOP** treating globally bounded Hessian curvature refinement as a demonstrated PhD-level algorithmic improvement on real EEG.

## Frozen-before-outcomes source and cohort

- [Immutable prospective preregistration](UNSEEN_EEG_TRAIN_GAIN_DECISION_CERT_PREREG_20261013.json): Git blob `b5b8b92f53f8588c937446193c5c641c5aca09e8` originally authored before any outcomes from this experiment.
- The original complete study [Actions #38355519800](https://github.com/lindicaphxag-tech/moabb/actions/runs/38355519800): **4/4 successful original independently split shards**, S029–S036 (8 genuinely new people, original R04/R08 EDF per person, 16 EDF files) with SHA-256s for each EDF. Original 80 clean heldout motor-imagery T1/T2 events. Three *predeclared synthetic* raw **C3 TRAIN-epoch-only** gains of +5%, +20%, +50%; the raw heldout source windows remain unchanged. This is 240 repeatedly compared decision instances, **NOT 240 independent humans or a clinical EEG fault trial**.
- The simple λ-only certificate and six-step globally bounded Hessian refinement use exactly the **same** fixed lambda=0.5, frozen train-only normalization, explicit penalized intercept, C3/C4 original EEG spectral features, solver, original label semantics and clean evaluation events.
- [Independent original-artifact audit #38355642342](https://github.com/lindicaphxag-tech/moabb/actions/runs/38355642342): first attempt FAILED because the audit started while the four original jobs had not uploaded their artifacts, found 0 artifacts and thus no S029 input. We **reran exactly the failed audit job** *after* the original four jobs finished, without changing the study code, protocol or original stored results: attempt 2 **SUCCESS**. Original EDF SHA256 identity, all 16 real recordings, full 48 run×gain cells, 80 clean original event trials and all 240 repeated policy/certificate comparisons checked. First audit failure was a workflow ordering issue, not a mathematical assertion failure.
- Original complete worker summaries (all four original logs):
  - A, S029–S030: **2** extra Hessian-covered decision instances; **7** observed refit prediction flips
  - B, S031–S032: **0** extra; **5** flips
  - C, S033–S034: **0** extra; **1** flip
  - D, S035–S036: **0** extra; **1** flip
  - total **2/240** additional certificate coverages under 3 related gain conditions, **14/240** observed class flips under perturbation, and **zero unsafe certified flips**. Counts are nested repeated decisions, not independent trials.

## The frozen hypothesis has been tested, not shifted after the result

| Frozen decision rule | Original and independently recomputed observation | Pass? |
|---|---|---|
| >=5 percentage points of mean person-level Hessian-vs-λ coverage benefit | **0.8333333333 percentage points** | **NO** |
| >=2 distinct independent people with strictly positive benefit | **1 of 8 people** | **NO** |
| Exactly zero falsely certified class invariances | **0 unsafe certificates** | YES |
| Complete original denominator, no result-based exclusion | All 8 people, all 16 source EDF, all 3 perturbations | YES |

**Independent full cohort audit's precommitted result: `precommitted_go_no_go_condition_met = false`.**

The descriptive participant-cluster bootstrap percentile interval for the primary benefit, with 2000 resamples and frozen seed 20261013, was **[0.0, 0.025]** in fractions = **[0, 2.5] percentage points**. Do not confuse this descriptive interval with an independently verified population effect under a randomized clinical trial.

## What was learned (narrow, honest)

1. A **preregistered train-only source intervention** can produce genuinely different predictions after retraining a regularized EEG decoder on publicly recorded event-labelled EEG: 14 repeated flip instances across 8 genuine human participants, under synthetic training C3 gain. This establishes a *conditional empirical sensitivity*, not naturally occurring harm/label leakage.
2. A mathematically more informative parameter-radius certificate is not automatically a practically meaningful scientific algorithm; under the frozen workload only **two** more decisions can be certified, in one independent person, and the preregistered aggregate improvement gate **FAILED**.
3. The earlier 20 fresh people with **0/200** flips had a *different intervention*: **test-only** sign inversion, not **train-only** C3 gain. These results are not inconsistent; they are controlled contrasts across distinct perturbation questions, classifiers and objectives. Do not pool them or quietly select whichever looked favorable.
4. The globally bounded Hessian contraction and λ gradient bound use well-known strong convexity. [Sosnin et al. (2026 JMLR), Abstract Gradient Training](https://www.jmlr.org/papers/v27/25-2206.html) and [Sosnin et al. (2025 TMLR), Certified Robustness to Data Poisoning in Gradient-Based Training](https://openreview.net/forum?id=9WHifn9ZVX) are substantive prior art. No new mathematical-priority claim or world-class original theorem follows from this implementation.
5. Source adaptation / risk certification is **not** yet a high-impact accepted paper method, nor a merged public MOABB feature. No demonstrated medical diagnostic gain or clinical deployment relevance from artificial channel gain.

## Doctoral go/no-go consequence

**Mechanism claim that must be WITHDRAWN on current evidence:** "Our data-curvature Hessian certificate substantially improves EEG retraining decision robustness certification on heldout real EEG relative to the ordinary λ-only convex certificate."

**Safe surviving statements:** correct conditional certificate math under exact model; 3-OS synthetic refit tests passed; one narrowly defined observed increase of 2/240 repeated certified decisions and 0 unsafe cases on registered public EEG; a preregistered negative comparison to the stronger new certificate.

**Next scientifically legitimate options (new registration required before new results):**
- Turn the negative into a *scientific measurement paper* only if linked to a distinctive, consequential, independently replicated characterization across **different source datasets**, rather than just 8 subjects in one corpus.
- For a new ML method, derive source-specific uncertainty sets and meaningful decision/retraining risk **before** selecting another cohort, compare against AGT, full fold-isolated preprocessing, relevant stability/robustness certificates, and measured compute + abstention cost. The new mathematical ingredient must not reduce to a relabeled Hessian bound.
- Or STOP this direction and concentrate resources on core Braindecode #1320 engineering and a robotics/LLM case whose strongest baseline is not yet already equivalent.

## External acceptance status

- Original EEG code and fork CI are author-executed, not an outside laboratory adopting the mechanism.
- [MOABB #1235](https://github.com/NeuroTechX/moabb/pull/1235) is open; its original scope is raw extraction-window overlap, **not** the new decision certificate. Avoid misrepresenting this as official MOABB adoption.
- A doctoral-grade "new discovery → new mechanism → independent evidence → external recognition" outcome has **not** yet been established for this particular certificate.

This full negative report is the scientifically required output of the frozen experiment, not a reason to invent a new gain or hide the failed gate.
