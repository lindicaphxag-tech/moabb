# Fully prospective 20-unseen-subject PhysioNet EEG decoder counterfactual: NEGATIVE

**Result date 2026-10-13.** The four worker shards of the actual heldout experiment all finished SUCCESS. A separate original-artifact auditor is running; keep that status separate and do not call it passed unless its CI succeeds.

## Original frozen design and data provenance

- [Prespecified 20-person experiment](PHYSIONET_20_NEW_SUBJECTS_DECODER_PREREG_20261013.json), git blob `c2e141134bbff9abfa0fae0edc73a9c2535d5aa9`, registered **before** the first 20-person experiment. All subjects S005 through S024 are disjoint from exploratory subjects S001–S004.
- The tested program is the exact preexisting [original pilot source](physionet_real_eeg_decoder_source_dependence.py), Git blob `67977ba9a58cdcd13f58b801a15c920874609f7b`, verified in every shard. No source, event, hyperparameter, filter, label, model, or split changes were made after seeing pilot results.
- [True original 20-person/40-EDF native study](https://github.com/lindicaphxag-tech/moabb/actions/runs/38341544307), four simultaneous independent five-person partitions, **four of four SUCCESS**. Actual original PhysioNet EEGBCI recordings R04 and R08 for every subject; no raw EEG files committed.
- For each original recording, class labels come from original T1/T2 motor-imagery annotations, channels C3/C4, 0.5–2.5 s event windows, first two thirds chronological for training and final third for clean evaluation. The original complete EDF is preprocessed with fixed 4th-order 1–30Hz zero-phase SOS IIR; features are fixed 8–12Hz and 13–30Hz log bandpower; decoder is C=1 scaled logistic regression.
- The controlled non-natural counterfactual **flips the original raw EEG source samples only in the heldout test event windows**, then re-filters the entire recording BEFORE recomputing train features. The model is trained on identical original train event labels and class-tested on the same **unaltered CLEAN test features**. The procedure is expressly a train-dependence probe, not legitimate evidence that physically natural EEG changes corrupt labels.

## Actual original four-shard results

| Shard | Subject IDs | Independent people | Original EDFs | Clean heldout original events | Predicted-label flips |
|---|---|---:|---:|---:|---:|
| A | S005–S009 | 5 | 10 | 50 | **0** |
| B | S010–S014 | 5 | 10 | 50 | **0** |
| C | S015–S019 | 5 | 10 | 50 | **0** |
| D | S020–S024 | 5 | 10 | 50 | **0** |
| **TOTAL** | **S005–S024** | **20** | **40** | **200** | **0** |

- No person had any changed predicted class, even when training within that original run was re-filtered using a different heldout test signal.
- Arithmetic average of the **20 subject-level means over their two runs** of the mean absolute class-1 test probability drift: **1.3732697271007234e-12**. This is descriptive; its confidence interval and full denominator should be recomputed from original JSON by [separate full-artifact audit](https://github.com/lindicaphxag-tech/moabb/actions/workflows/physionet-20-unseen-artifact-audit.yml).
- The earlier **four pilot participants / eight EDF / 40 original test epochs** similarly produced **0/40** predicted-label changes under the same original source/feature/model design. It was exploratory for the 20-unseen-subject result and therefore MUST NOT be counted in the test 20.

## Interpretation / proof obligations

1. Whole-record zero-phase IIR preprocessing can create a measurable mathematical dependency across nominally non-overlapping source intervals, as separately demonstrated by [64/64 independent public EDF signal-source checks](https://github.com/lindicaphxag-tech/moabb/actions/runs/38337809594).
2. But for the exact original event protocol, real cross-fold dataset and fixed feature-based EEG classifier **this measurable dependence did NOT cross a prediction decision boundary on 200 clean heldout original events from 20 newly held-out subjects**.
3. Consequently, **do not claim** material harmful test leakage, uplift from a newly proposed filter-gap guard, changed EEG decoding accuracy, broad clinical reliability, or new mathematical theorem priority on the basis of these results.
4. A causal provenance diagnostic is distinct from an information-leakage impact or medical generalization claim. Important scientific questions remain about much denser/shorter adjacent event windows, finite response operators, realistic intervention magnitudes and high-margin/low-margin decoding pipelines, but each new protocol must be preregistered on a NEW population before interpreting its gains.
5. All 20 new human participants are distinct participant units; the 40 EDFs, 200 events and multiple synthetic worlds must NOT be treated as additional independent people. We retain the full complete negative denominator and all four shard results.
6. New mathematical results by OpenAI, Anthropic and other groups are evaluated by proof originality, formal checking and independent mathematician review; these source-data audits alone cannot be put in that research category.

**Engineering and scientific status:** full preregistered public real EEG decode *negative* outcome. Fully independent original-artifact audit is a further technical verification step; no new upstream MOABB Merge or academic publication is claimed.
