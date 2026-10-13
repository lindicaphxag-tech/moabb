# Decision-aware EEG fold source risk: a verified convex-optimality certificate

**Research code (2026-10-13), NOT an original new mathematical theorem.**

The motivation is a **falsification discovered from real recordings**: in our originally frozen PhysioNet EEGBCI signal-source experiment, holding raw train and test windows disjoint did NOT force zero whole-record `sosfiltfilt` source influence. A separate 4-pilot and 20-*new*-person real EEG motor-imagery classification trial then showed 0/40 and **0/200** heldout class flips respectively, on the declared event-feature/model protocol. Thus the existence of even exact measurable source dependence is *not sufficient* for a meaningful error or decision-level intervention claim.

Original empirical evidence: [8 original EDF source experiment](https://github.com/lindicaphxag-tech/moabb/actions/runs/38336887369), [separate SHA256/linearity replay](https://github.com/lindicaphxag-tech/moabb/actions/runs/38337809594), [20 new-person true heldout EEGMI study](https://github.com/lindicaphxag-tech/moabb/actions/runs/38341544307), [independent four-artifact and participant-cluster audit](https://github.com/lindicaphxag-tech/moabb/actions/runs/38342691099).

## Question

**When is a raw-test-origin perturbation genuinely capable of changing a test decision AFTER the entire trained regularized decoder is refit, rather than merely altering a signal by a numerically nonzero amount?** Can an inexpensive certificate let an evaluation system abstain from aggressive purge/extra filtering when the exact known counterfactual cannot change a heldout class? This is application of existing theory; establishing added MOABB practical value is an OPEN question.

## Fixed mathematical model

Use *exactly* the same fixed train labels `y_i ∈ {-1,+1}`, clean-training-only standardized feature vectors `z_i`, perturbed-source/refiltered training feature vectors `z_i'`, fixed clean heldout vectors `q_j`, and `λ > 0`. All augmented features include a **regularized bias** (unlike sklearn's default unregularized intercept).

Let

`F_delta(theta) = (1/n) sum_i log(1 + exp(-y_i z_i'^T theta)) + (lambda/2) ||theta||_2^2`.

This objective is `lambda`-strongly convex. Let `theta_delta_star` be its **exact** optimizer, and `theta_0` any fixed finite reference weights, including a numerically trained clean model. The standard strong-monotonicity inequality is

`<grad F_delta(theta_0) - grad F_delta(theta_delta_star), theta_0 - theta_delta_star> >= lambda ||theta_0 - theta_delta_star||^2`.

Since `grad F_delta(theta_delta_star)=0`, Cauchy-Schwarz yields

**`||theta_delta_star - theta_0|| <= ||grad F_delta(theta_0)|| / lambda`.**

For any unmodified original clean heldout feature `q`,

`|q^T(theta_delta_star - theta_0)| <= ||q|| * ||grad F_delta(theta_0)||/lambda`.

Therefore, if

**`|q^T theta_0| > ||q|| * ||grad F_delta(theta_0)||/lambda`,**

the final predicted sign for this heldout input **CANNOT change** when one refits the exact, fixed regularized convex objective on this *specific registered source-counterfactual* dataset. If the test margin does not exceed the upper bound, the method returns **ABSTAIN**; abstention is not evidence a prediction actually flipped. This establishes the mathematical guarantee even if `theta_0` is not an exact clean optimum, because the bound uses the changed objective gradient at that candidate.

A practical numerical refit has nonzero gradient residual `eps`, so comparing the two *numerically approximated* models requires an extra `eps/lambda` parameter slack. The test suite enforces this in all empirical comparisons.

## Executed implementation and deliberate failure modes

- [Pure mathematical implementation](../decision_aware_convex_eeg_certificate.py) includes deterministic strongly regularized logistic optimization, fixed-bias semantics, exact changed-objective gradients, per-test margins, per-case bound, abstention and numerical refit audit. Nonfinite inputs, mismatched labels and nonpositive `λ` fail closed.
- [Adversarial unit tests](../test_decision_aware_convex_eeg_certificate.py) challenge the guarantee with seven perturbation scales, six independently seeded synthetic changes, finite-difference gradient checks, regularization invalidity, zero-margin abstention, optimizer stationarity and alternate feature shape. A first run had **one test harness regex expectation error** after the program already correctly rejected a mismatched number of samples. This was transparently corrected WITHOUT relaxing any decision bound. [Three-OS full test workflow SUCCESS](https://github.com/lindicaphxag-tech/moabb/actions/runs/38351231688), Windows/macOS/Ubuntu Python 3.12.
- [True-original real EDF new-cohort preregistration](../PHYSIONET_PROSPECTIVE_DECISION_CERT_4_NEW_PREREG_20261013.json) freezes S025–S028, R04/R08, 40 original heldout events, ordinary T1/T2 MI event labels, original prior feature extraction, the same known source-only polarity flip, and new `λ=0.5` **fully regularized model**. It explicitly forbids claiming it is the exact same logistic regression as the earlier sklearn decoder.
- [Prospective new-human experiment code](../physionet_prospective_decision_certificate.py) is pinned by Git blob and computes original source SHA256 and per-person certificate coverage, actual flips, all abstentions, refit residual and model-vs-bound correctness. [Live separate original-EDF CI](https://github.com/lindicaphxag-tech/moabb/actions/workflows/physionet-prospective-decision-certificate.yml) must be checked before any claimed human evidence. Initial run lacked an **indirect scikit-learn dependency of the frozen event reader**; fixed only the CI dependency, not the scientific protocol.
- The earlier observational outcome 20 people/200 test events with no flips CANNOT itself be retroactively certified, because its sklearn intercept was unregularized while this proof explicitly requires globally positive strong convexity. Claims from these are kept separate.

## Prior-art / originality boundary

Strongly convex gradient residual bounds have long been standard optimization, and training-point poisoning/uncertainty certificates already include broader methods:

- Sosnin et al., [*Certified Robustness to Data Poisoning in Gradient-Based Training*, TMLR 2025](https://openreview.net/forum?id=9WHifn9ZVX).
- Sosnin et al., [*Abstract Gradient Training: A Unified Certification Framework for Data Poisoning, Unlearning, and Differential Privacy*, JMLR 2026](https://www.jmlr.org/papers/v27/25-2206.html).
- Brookshire et al., [*Data leakage in deep learning studies of translational EEG*, Frontiers 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/).

Accordingly, the mathematical inequality and poison-stability technique must **never** be presented as an OpenAI/Anthropic-level new mathematics breakthrough. A potentially scientific EEG algorithm contribution requires validating that source-measured intervention actually changes model-selection behavior or enables a *material, conservative reduction* in sample purging/runtime at adequate subject-level evidence, relative to full strong AGT/boundary controls and correctly isolated preprocessing. No such new performance effect has yet been shown.

## Strict claim levels

1. **Achieved:** a correct and independently CI-tested finite-source convex prediction certificate with sound synthetic refit comparisons (3 operating systems).
2. **Historical negative:** public EEG cross-fold source influence can exist while clean heldout predictions remain unchanged in prior 20-unseen-person experiment.
3. **Pending:** actual 4-new-person prospective certificate coverage and any evidence of nonvacuity on real EDFs; no success presumed until exact-head workflow succeeds.
4. **Not achieved:** original unrecognized theorem, model accuracy improvement, new official MOABB Merge, accepted journal publication, independent mathematical priority validation.

This structure is deliberately evidence-bound: we gain the right to write down a **new applied hypothesis**, not the right to label existing convex theory a new proof.
