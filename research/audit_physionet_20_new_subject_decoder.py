"""Independent participant-cluster auditor for frozen 20-person PhysioNet EEG cohort.

Consumes four ORIGINAL GitHub Actions JSON artifacts. Recomputes every
subject/run probability and class-flip summary and a participant-only
descriptive bootstrap (seed preregistered). No synthetic interventions are
counted as independent people, and no results are silently dropped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

PLAN=Path("research/PHYSIONET_20_NEW_SUBJECTS_DECODER_PREREG_20261013.json")
PLAN_BLOB="c2e141134bbff9abfa0fae0edc73a9c2535d5aa9"
PARENT_BLOB="67977ba9a58cdcd13f58b801a15c920874609f7b"


def audit(directory):
    digest=subprocess.check_output(["git","hash-object",str(PLAN)],
                                   text=True).strip()
    if digest!=PLAN_BLOB:
        raise RuntimeError("Original registered 20-subject design no longer pinned")
    spec=json.loads(PLAN.read_text(encoding="utf-8"))
    all_subject_rows=[]
    for shard,expected_subjects in spec["precommitted_per_subject_shards"].items():
        path=directory/f"unseen_eeg_decoder_{shard}.json"
        sample=json.loads(path.read_text(encoding="utf-8"))
        if (
            sample["schema"]!=spec["schema"]
            or sample["shard"]!=shard
            or sample["prospective_preregister_git_blob"]!=PLAN_BLOB
            or sample["parent_classifier_source_git_blob"]!=PARENT_BLOB
            or sample["new_original_participants"]!=5
            or sample["new_original_recordings"]!=10
            or len(sample["subject_rows"])!=5
            or {x["person"] for x in sample["subject_rows"]}!=set(expected_subjects)
        ):
            raise RuntimeError(f"Unregistered, duplicated or missing {shard} shard")
        all_subject_rows+=sample["subject_rows"]
    participants=sorted(all_subject_rows,key=lambda x:x["person"])
    if (
        len(participants)!=20
        or [x["person"] for x in participants]!=
            spec["entire_fresh_subject_ids"]
    ):
        raise RuntimeError("20 genuinely independent subject units incomplete")
    all_runs=[]
    for person in participants:
        runs=person["registered_recordings"]
        if (len(runs)!=2 or {x["run"] for x in runs}!={4,8}
                or any(x["participant"]!=person["person"] for x in runs)):
            raise RuntimeError("Incorrect paired EEG run IDs")
        if person["actual_original_runs"]!=2 or len(
            person["original_edf_sha256"]
        )!=2:
            raise RuntimeError("Subject's raw data evidence missing")
        for run,sha in zip(runs,person["original_edf_sha256"]):
            if (run["edf_sha256"]!=sha or len(sha)!=64
                or run["source_perturbation_changes_train_raw_epochs"]
                or run["direct_train_test_epoch_raw_overlaps"]!=0
                or run["masked_test_source_control_train_feature_l2"]!=0
                or not run["genuine_motor_imagery_event_annotations"]
                or not run["adversarial_synthetic_heldout_signal_flip"]
                or run["train_event_count"]<5
                or run["test_event_count"]<1
                or run["prediction_flips"]<0
                or run["prediction_flips"]>run["test_event_count"]
                or sum(run["training_class_counts"])!=run["train_event_count"]
                or sum(run["testing_class_counts"])!=run["test_event_count"]
            ):
                raise RuntimeError("Source recording/EEG artifact provenance failed")
            all_runs.append(run)
        def avg(name):
            return float(np.mean([r[name] for r in runs]))
        if (
            abs(person["mean_within_person_abs_probability_change"]-
                avg("mean_heldout_probability_absolute_delta"))>1e-15
            or person["total_test_trial_prediction_flips"] !=
                sum(x["prediction_flips"] for x in runs)
            or person["any_predictive_class_flip"] !=
                (sum(x["prediction_flips"] for x in runs)>0)
            or abs(person["per_person_mean_accuracy_change_diagnostic"]-
                np.mean([r["accuracy_corrupt_test_source_only"]-
                         r["accuracy_clean"] for r in runs]))>1e-15
        ):
            raise RuntimeError("Reported per-person statistics not reproducible")
    values=np.asarray([
        x["mean_within_person_abs_probability_change"]
        for x in participants
    ],dtype=float)
    if values.shape!=(20,) or not np.isfinite(values).all():
        raise RuntimeError("Invalid original participant-level metrics")
    rng=np.random.default_rng(20261013)
    subject_bootstrap_means=np.asarray([
        np.mean(values[rng.integers(0,20,size=20)]) for _ in range(2000)
    ],dtype=float)
    results={
        "schema":"frozen_20_subject_eeg_decoder_original_artifact_audit_v1",
        "source_full_original_20_subjects":True,
        "independent_heldout_participant_count":20,
        "original_edf_recordings":40,
        "actual_original_heldout_event_trials":sum(r["test_event_count"]
                                                  for r in all_runs),
        "total_original_prediction_class_flips":sum(r["prediction_flips"]
                                                     for r in all_runs),
        "participants_with_class_flip":sum(
            p["any_predictive_class_flip"] for p in participants
        ),
        "participant_level_mean_absolute_probability_change":float(
            np.mean(values)
        ),
        "participant_level_median_absolute_probability_change":float(
            np.median(values)
        ),
        "participant_mean_change_descriptive_cluster_bootstrap_2p5_97p5":[
            float(t) for t in np.quantile(subject_bootstrap_means,[0.025,0.975])
        ],
        "participant_cluster_bootstrap_seed":20261013,
        "source_is_synthetic_counterfactual_on_real_public_eeg":True,
        "not_independent_clinical_or_label_leakage_proof":True,
        "not_benchmark_accuracy_gain_claim":True,
        "subject_wise":{
            str(p["person"]):{
                "mean_abs_probability_change":
                    p["mean_within_person_abs_probability_change"],
                "class_flips":p["total_test_trial_prediction_flips"],
                "heldout_trials":p["total_clean_heldout_trials"],
                "accuracy_delta":p["per_person_mean_accuracy_change_diagnostic"]
            } for p in participants
        }
    }
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.artifacts)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("UNSEEN_20_INDEPENDENT_SUBJECT_AUDIT",json.dumps({
        k:v for k,v in result.items() if k!="subject_wise"
    },sort_keys=True),flush=True)
