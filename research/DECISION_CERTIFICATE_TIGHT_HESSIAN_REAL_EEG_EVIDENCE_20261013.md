# Source-conditioned logistic EEG decision certificates: complete evidence and tightened bound

**2026-10-13 | Prospective, auditable algorithm-development note.**
This is a *known optimization-theory* inequality adapted to a specific real PhysioNet EEG counterfactual. Do NOT claim new theorem priority, 2026 OpenAI/Claude-level original mathematics, proven practical deployment value or official MOABB adoption.

## Preregistered real human EEG evidence (completed)

- Exact [S025–S028 cohort/source/model preregistration](PHYSIONET_PROSPECTIVE_DECISION_CERT_4_NEW_PREREG_20261013.json), SHA-1 Git blob `a28c932fa2f990a3f621b49cac224183ac2d64b0`. Subjects S025–S028, two EDF runs each, 40 true T1/T2 heldout event windows (four independent humans), true PhysioNet source EEG (C3/C4). None belongs to exploratory S001–S024. Fully regularized logistic model **penalizes the intercept** and uses the original CLEAN training split only to freeze standardization.
- [Successful true original prospective EDF experiment](https://github.com/lindicaphxag-tech/moabb/actions/runs/38351450509), original EEG SHA-256 fingerprints for all 8 EDF recordings, no raw EEG uploaded: **40/40** test decisions received the sign-invariance certificate, **0/40** were actually flipped by the exact synthetic heldout-only source polarity reversal, **0/40** abstentions.
- [Separate full-source-data ARTIFACT auditor success](https://github.com/lindicaphxag-tech/moabb/actions/runs/38353240808): independently checked the eight SHA-256-recorded original EDF identities, all original 40 cells and 4 subject clusters, reference/refit numerical residual, strong convex parameter enclosure and per-person certificate counts: **40/40** verified, 0 false positives. This is independent CODE verification using the same underlying human recordings, **NOT external-laboratory replication**.
- At larger earlier prospective-scale, [20 unseen real public EEG people/200 genuine clean heldout event trials](https://github.com/lindicaphxag-tech/moabb/actions/runs/38341544307), [independent 20-person artifacts and participant audit](https://github.com/lindicaphxag-tech/moabb/actions/runs/38342691099), showed **0/200 actual class flips** using the PRIOR sklearn model with unpenalized bias. This is a different model and thus is **not automatically covered** by this λ-strong-convex theorem. That earlier negative result motivated the certificate question and is not a positive classifier performance gain.

## Registered original mathematical model

For a known test-origin, raw-source perturbation, retain the original train labels, original clean-test features, and exact fixed preprocessing/transformation. Only the affected **train feature vectors** change. The fully regularized finite-dimensional logistic loss is

    F_delta(w) = (1/n) Σ_i log(1 + exp(-y_i z_delta_i^T w))
               + λ/2 ||w||_2², λ>0, y_i∈{-1,+1}.

For ANY finite reference weights w0 (not required to be a global clean optimum),
the exact changed-data minimizer w_delta* obeys

    ||w_delta* - w0|| <= ||∇F_delta(w0)|| / λ = R0.

Thus each frozen original clean holdout feature q has unchanged classification
sign whenever

    |qᵀ w0| > ||q|| * R0.

Otherwise **abstain**. The original 3-OS tested implementation and proof live at [decision_aware_convex_eeg_certificate.py](decision_aware_convex_eeg_certificate.py) and [DECISION_AWARE_EEG_CONVEX_CERTIFICATE_PROOF_20261013.md](DECISION_AWARE_EEG_CONVEX_CERTIFICATE_PROOF_20261013.md).

The inequality follows from previously known gradient strong monotonicity and Cauchy–Schwarz; this is **not** novel mathematical priority.

## Sharper globally bounded Hessian refinement (implemented + tested)

A useful next question: when the λ-only radius is too loose, can information
about the registered changed training matrix certify more true unchanged
decisions, **without** asserting label leakage or changing the model?

For each changed training feature z_i and any w in the already certified
ball B(w0,R), let

    A_i(R) = |y_i z_iᵀ w0| + ||z_i|| R,
    k_i(R) = sigmoid(-A_i(R))*sigmoid(A_i(R)).

Because the logistic Hessian's scalar weight decreases with absolute
margin, over the ENTIRE ball (not only at the fitted training solution),
the changed objective Hessian satisfies

    ∇² F_delta(w) >= λI + (1/n)Σ_i k_i(R) z_i z_iᵀ.

Write

    μ(R) = λ + λ_min((1/n) Z_deltaᵀ diag(k_i(R)) Z_delta)
    ≥ λ.

Since the unknown w_delta* lies within B(w0,R), integrate the Hessian
along the segment to that minimizer. This gives the new, smaller VALID
enclosure

    R_next = min(R, ||∇F_delta(w0)|| / μ(R)).

Repeat the finite contraction a declared number of times. For heldout q,
strictly compare |qᵀ w0| to ||q|| R_next; the new certificate can cover
a weak-margin q that the λ-only bound must abstain on.

This formula is an elementary curvature refinement of established
strong-convexity inequalities, not a first-discovered theorem.
Numerical SVD/eigenvalue computations are **float64 with conservative
programmatic slacks, NOT mathematically verified interval enclosures**;
actual safety-critical claims require interval arithmetic/Lean and a
verified numerical model.

- [Code](decision_aware_eeg_curvature_refinement.py) and [independent refit tests](test_decision_aware_eeg_curvature_refinement.py).
- [Successful 3-OS adversarial CI](https://github.com/lindicaphxag-tech/moabb/actions/runs/38353387922): Ubuntu, Windows and macOS all pass; tests include a constructed candidate whose test margin lies **strictly between** the λ-only and new curvature radius, so baseline abstains while tighter certificate proves invariance; multiple seeded exact numerical refits verify that covered decisions do not flip. Such constructed witnesses are NOT real human EEG deployment outcomes.

### What still blocks a claim of original flagship research

1. **Practical nonvacuity**: in the S025–S028 study the old λ-only bound already certified all 40 test decisions. It is **not truthful** to market the tighter Hessian bound as a measured improvement on that cohort. The nontrivial coverage advantage so far exists only in a synthetic adversarial witness.
2. **Prior art**: exact optimizer-distance bounds and robustness certificates are broadly known; see [Abstract Gradient Training, JMLR 2026](https://www.jmlr.org/papers/v27/25-2206.html). Literature review against stronger poisoning/unlearning certificate baselines, reachability analysis, sample removal and class margin screening is indispensable.
3. **New real-data go/no-go**: a truly fresh and pre-registered population with meaningful near-boundary source influence, more than one classifier, subject-level paired confidence intervals, and a demonstrable runtime/sample-retention/accuracy or decision-risk gain.
4. **External adoption**: community maintainer acceptance of a small safe optional EEG evaluation API, and independent external lab reuse/review. No official merge for this research branch.
5. **Formal rigor**: no interval arithmetic or Lean proof has yet checked the actual floating implementation. It is an exact-real-number mathematical bound, not a verified floating-point safety certificate.

**Scientific claim supported now:** public EEG heldout-to-train signal dependence exists but was operationally negligible for previously studied MI decoders; a conditioned certificate mathematically rules out one specific source-retraining prediction flip in 40/40 fresh decision events, and an established curvature refinement can tighten the conservatism in synthetic strong-baseline comparisons.
