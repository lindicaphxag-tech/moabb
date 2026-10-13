# Real EEGBCI motor-imagery classifier under heldout-source IIR dependence

**2026-10-13, preregistered four-participant pilot — NEGATIVE decoder-level outcome.**

## Evidence and provenance

- Original code+source [frozen protocol](PHYSIONET_EEG_DECODER_CONTRACT_PREREG_20261013.json), blob `25501f7fa3eacc460f4baa116878ca1ec3db947f`, registered **before** this classifier run.
- [Original GitHub CI, SUCCESS](https://github.com/lindicaphxag-tech/moabb/actions/runs/38341234033), `82c60e3b9c644c546e8d77d3417f934123ce238a`. Every raw EDF SHA-256 is emitted with the original worker artifact. The source is four true people, original 8 EDF recordings (R04/R08) and 40 heldout real T1/T2 event trials.
- Raw EEG C3/C4 + MNE's actual T1/T2 motor imagery annotation; event crop 0.5–2.5 sec, chronological early 2/3 for train and final 1/3 for test; no direct overlap of train and test raw source windows. Within each run, filtering uses SciPy order-4 Butterworth 1–30 Hz zero-phase SOS on the **whole original continuous recording**.
- The *counterfactual* multiplies the original raw source EEG by -1 exclusively inside ALL test-event crops, BEFORE recomputing whole-record filtering. Training labels/times are fixed. Logistic Regression(C=1) with standardized four Hann-windowed log bandpower features (8–12 and 13–30 Hz for C3/C4) is refit from those changed training features. It is **always scored on exactly the original clean test features**, never the altered test signal. All eight recorded negative controls that mask the same counterfactual test intervals before filtering have train-feature difference exactly zero.
- This manipulation is synthetic on genuine public EDFs, **not naturally occurring brain activity, label leakage, a paired independent patient recruitment study or clinical deployment**.

## Full registered pilot, including negative findings

| Original metric | Observed |
|---|---:|
| Independent participants | 4 |
| Original public EDF source recordings | 8 |
| True heldout original event trials (test labels never used for training) | 40 |
| Actual holdout classification flips after altering ONLY the heldout raw source | **0/40** |
| Mean original clean test accuracy across 8 recordings | **0.55** |
| Mean counterfactual-test-source train / original clean test accuracy | **0.55** |
| Median *within-record mean* absolute predicted class-1 probability change | **6.910888528111058e-13** |
| Original recording pairs with at least one classification flip | **0/8** |

The result **falsifies** the naive extrapolation `nonzero whole-record IIR transfer => material change in EEG classifier predictions` in these sparse, temporally separated real event epochs. Even though direct earlier PhysioNet source-window experiments measured nonzero raw train influence for gaps <=640 samples, actual training event windows here are often much further from heldout events: an IIR's long nonzero tail does not imply practically harmful decoding bias.

The result **does not prove universal harmlessness**: only two EEG channels, 4 individuals, 8 records, one fixed feature/classifier, restricted class labels and chronological split. There is no benchmark-scale accuracy gain, no state-of-the-art EEG decoding and no external laboratory review.

## Scientifically necessary next controls

1. The [20 newly heldout subjects S005–S024 study](https://github.com/lindicaphxag-tech/moabb/tree/research/physionet-eeg-decoder-20-unseen-20261013) was preregistered before new-cohort outcomes, and retains the same immutable original program Git blob `67977ba9a58cdcd13f58b801a15c920874609f7b`. [Original four-shard study](https://github.com/lindicaphxag-tech/moabb/actions/runs/38341544307) runs 40 new EDFs; all 20 people must be retained or failures accounted explicitly. **No favorable unseen-data result is assumed here.**
2. Vary *prospectively* event proximity, finite response kernel vs IIR, frequency features and classifier decision margins; separate actual test-source dependence from label/source misuse by a learner. Stratify by subject, recording, and true raw guard distance.
3. For a deployable optional fold preprocessing utility, compare fair fold-specific fit with standard MOABB preprocessing, compute exact sample retention and runtime, and require that actual holdout participant-level conclusions change. A small `F(delta)` numerical effect is not a biomedical diagnostic or proven risk reduction.
4. External review of [MOABB #1235](https://github.com/NeuroTechX/moabb/pull/1235) still pending; it guarantees only raw epoch source overlap purging, not post-IIR independence.

**Bottom line:** credible reproducible public data at signal level, genuinely negative at pilot classification level. A novel method must solve an important measured consequence rather than relabel ordinary spectral norms or inflate synthetic signal effects.
