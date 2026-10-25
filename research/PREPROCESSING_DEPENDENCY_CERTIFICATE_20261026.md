# Preprocessing-dependency certificate: a bounded next research step (2026-10-26)

**Status:** independent, reproducible NumPy counterexample and linear-algebra certificate. **Not** an externally merged PR, a new MOABB integration feature, a general theorem for nonlinear pipelines, or proof of new scientific novelty.

Official minimal raw-window PR: [NeuroTechX/moabb#1329](https://github.com/NeuroTechX/moabb/pull/1329). This research is **deliberately isolated** from its three-file maintainer-requested scope.

## Issue: raw-window disjointness is not preprocessed-output independence

Let `x` be a raw time series, `y = H x` the output of a **fixed linear filter**, `A` the training output sample positions and `T` the held-out test raw sample positions. Even if `A ∩ T = ∅`, there can be information flow into training outputs through the cross-dependency operator

```text
C_train<-test = P_A H P_T.
```

For any perturbation `δx` supported only on `T`,

```text
||P_A H P_T δx||_2 <= ||P_A H P_T||_2 ||δx||_2
                       <= ||P_A H P_T||_F ||δx||_2.
```

Thus strict nondependence in a *known, fixed linear* pipeline needs `C = 0`, not merely raw extraction intervals with empty intersection. A prescribed `ε` bound on `||C||_F` is an **approximate influence certificate**, not a strict guarantee of independence. A bidirectional check additionally examines `P_T H P_A`.

## Executed controlled evidence

Run `python research/preprocessing_dependency_certificate.py` with NumPy.

- Training output indices: 90..129 inclusive; held-out test raw indices: 210..249 inclusive. Both have 40 samples, no raw index overlap.
- Finite-support triangular FIR with radius 10: both cross Frobenius norms **0**; strict zero cross-dependency.
- Symmetric exponential infinite-support convolution (idealized zero-phase IIR proxy) with `h[d]=((1-alpha)/(1+alpha))*alpha^abs(d)`, `alpha=0.97`: both cross norms **0.019945122682946972** despite zero raw overlap. A unit held-out impulse yields **0.005075760340662572** L2 in training outputs, bounded by **0.019945122682946972**.
- The IIR cross norm decreases with increasing inter-interval gap but stays strictly positive in the chosen analytic model: gap 0→**0.22808565**; 5→**0.19586491**; 20→**0.12403169**; 50→**0.04973758**; 100→**0.01084604**.

These values are for the specified *toy kernel*, not MOABB's actual IIR coefficients or published EEG effects.

## Next proof and native-integration gates

1. For each actual preprocessing stage, derive or empirically identify dependence support, including zero-phase FIR/IIR, resampling, normalization fit statistics, learned transforms, and padding/boundary behavior.
2. Prove exact zero cross-dependence when support is finite and the relevant train/test raw support sets are disjoint. For infinite support, pre-register a bounded influence budget and sample-domain assumptions; never equate small coupling with no leakage.
3. Implement **fold-aware preprocessing** as a separate, optional proposal and compare against (a) standard MOABB, (b) raw-window-only purge and (c) support-aware purge on actual EEG data. Include a synthetic injected-impulse ground truth and independently held-out recordings.
4. Quantify computational/label coverage and predictive utility; report failures and null results. Do not claim SOTA improvements before completing the native experiments.

**Why kept separate:** A theoretically stronger preprocessing protocol is larger and riskier than the narrowly scoped #1329 that the maintainer explicitly offered to review. Both can progress without jeopardizing the near-term upstream merge.
