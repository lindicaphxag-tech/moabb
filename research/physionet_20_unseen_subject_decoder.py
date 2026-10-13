"""Prospective truly new 20-person real EDF EEG decoder sensitivity cohort.

This imports the original 4-person classifier source unchanged and verifies
its Git blob. *Each person*, not each event, run or source-perturbation, is a
unit of independent human sampling. Synthetic sign-flip is not clinical error.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from research.physionet_real_eeg_decoder_source_dependence import (
    frozen_protocol, one_original_edf,
)

PLAN=Path("research/PHYSIONET_20_NEW_SUBJECTS_DECODER_PREREG_20261013.json")
PLAN_SHA="c2e141134bbff9abfa0fae0edc73a9c2535d5aa9"
PARENT=Path("research/physionet_real_eeg_decoder_source_dependence.py")
PARENT_SHA="67977ba9a58cdcd13f58b801a15c920874609f7b"


def run_shard(shard, cache, dest):
    for path,want in ((PLAN,PLAN_SHA),(PARENT,PARENT_SHA)):
        sha=subprocess.check_output(
            ["git","hash-object",str(path)],text=True
        ).strip()
        if sha!=want:
            raise RuntimeError(f"Frozen original study bytes changed: {path}")
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    original=frozen_protocol()
    if (plan["status"]!="FROZEN_BEFORE_UNSEEN_COHORT_OUTCOMES"
            or plan["registered_parent_protocol_blob"]!=
               "25501f7fa3eacc460f4baa116878ca1ec3db947f"
            or plan["registered_parent_code_blob"]!=PARENT_SHA
            or plan["independent_new_subject_clusters"]!=20
            or sorted(sum(plan["precommitted_per_subject_shards"].values(),[]))!=
               plan["entire_fresh_subject_ids"]
            or set(plan["entire_fresh_subject_ids"]) &
               set(plan["exploratory_subject_ids"])):
        raise RuntimeError("Original heldout 20-person plan mismatch")
    if shard not in plan["precommitted_per_subject_shards"]:
        raise ValueError("Unregistered independent participant shard")
    people=plan["precommitted_per_subject_shards"][shard]
    output=[]
    for person in people:
        rows=[]
        for run in plan["runs"]:
            row=one_original_edf(original,person,run,cache)
            rows.append(row)
            print("UNSEEN_EEG_DECODER_ORIGINAL_RUN",
                  json.dumps(row,sort_keys=True),flush=True)
        if len(rows)!=2 or set(x["run"] for x in rows)!={4,8}:
            raise RuntimeError("Incomplete paired source recording")
        mean_prob=float(np.mean([
            x["mean_heldout_probability_absolute_delta"] for x in rows
        ]))
        entry={
            "person":person,"actual_original_runs":2,
            "original_edf_sha256":[x["edf_sha256"] for x in rows],
            "total_clean_heldout_trials":sum(x["test_event_count"] for x in rows),
            "mean_within_person_abs_probability_change":mean_prob,
            "any_predictive_class_flip":sum(x["prediction_flips"] for x in rows)>0,
            "total_test_trial_prediction_flips":sum(x["prediction_flips"] for x in rows),
            "per_person_mean_accuracy_change_diagnostic":float(np.mean([
                x["accuracy_corrupt_test_source_only"]-x["accuracy_clean"]
                for x in rows
            ])),
            "per_person_mean_train_feature_delta_l2":float(np.mean([
                x["train_log_bandpower_delta_l2"] for x in rows
            ])),
            "registered_recordings":rows,
        }
        output.append(entry)
        print("UNSEEN_EEG_DECODER_PERSON_COMPLETE",
              json.dumps({k:v for k,v in entry.items()
                          if k!="registered_recordings"},sort_keys=True),
              flush=True)
    if len(output)!=5 or {x["person"] for x in output}!=set(people):
        raise RuntimeError("Sharded original human denominator incomplete")
    result={
        "schema":plan["schema"],"shard":shard,
        "prospective_preregister_git_blob":PLAN_SHA,
        "parent_classifier_source_git_blob":PARENT_SHA,
        "new_original_participants":5,
        "new_original_recordings":10,
        "subject_rows":output,
        "no_test_set_tuning_or_real_clinical_claim":True,
    }
    dest.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("UNSEEN_EEG_DECODER_SHARD_FINISHED",json.dumps({
        "shard":shard,"participants":5,"original_EDFs":10,
        "complete":True
    },sort_keys=True),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--shard",choices=("A","B","C","D"),required=True)
    arg.add_argument("--cache",type=Path,
                     default=Path("/tmp/unseen_eegbci_original_edfs"))
    arg.add_argument("--output",type=Path,required=True)
    a=arg.parse_args()
    run_shard(a.shard,a.cache,a.output)
