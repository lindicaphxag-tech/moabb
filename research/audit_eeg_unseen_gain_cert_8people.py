"""Independent no-drop, all-8-person original-artifact EEG certificate audit.

Recomputes the prospectively frozen person-level primary endpoint, per-scale
certified coverage and empirically flipped clean test labels from all four
original actions artifacts. The independent statistical unit is the human,
NOT runs, scales or trials. An independent software audit is not a second
scientific lab or a physical-field clinical replication.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

PLAN=Path("research/UNSEEN_EEG_TRAIN_GAIN_DECISION_CERT_PREREG_20261013.json")
PLAN_SHA="b5b8b92f53f8588c937446193c5c641c5aca09e8"
SHARDS={"A":[29,30],"B":[31,32],"C":[33,34],"D":[35,36]}
SOURCE_RUN=38355519800


def audit_artifacts(folder):
    actual=subprocess.check_output(["git","hash-object",str(PLAN)],
                                   text=True).strip()
    if actual!=PLAN_SHA:
        raise RuntimeError("Prospective preregistration changed")
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    allrows=[]
    for shard,people in SHARDS.items():
        f=folder/f"eeg_unseen_gain_{shard}.json"
        data=json.loads(f.read_text(encoding="utf-8"))
        if (data["schema"]!=plan["schema"]
            or data["shard"]!=shard
            or data["preoutcome_preregister_git_blob"]!=PLAN_SHA
            or data["registered_people"]!=people
            or data["actual_original_EDFs"]!=4
            or data["actual_heldout_clean_event_count"]!=20
            or len(data["records"])!=4):
            raise RuntimeError(f"Original {shard} partition/provenance altered")
        allrows.extend(data["records"])
    if (len(allrows)!=16 or
        {(r["person"],r["run"]) for r in allrows}!=
            {(p,run) for p in plan["subjects"] for run in plan["runs"]}):
        raise RuntimeError("Incomplete/duplicated original human or run cohort")
    by_subject={}
    total_original_events=0
    all_comparisons=0
    for person in plan["subjects"]:
        pair=[r for r in allrows if r["person"]==person]
        if {r["run"] for r in pair}!={4,8}:
            raise RuntimeError("Registered two real EDF runs not found")
        entries=[]
        for row in pair:
            if (len(row["original_edf_sha256"])!=64
                or row["original_edf_bytes"]<2048
                or row["original_clean_test_events"]!=5
                or sum(row["original_train_classes"])!=row["original_train_events"]
                or sum(row["original_test_classes_diagnostic_only"])!=5
                or len(row["registered_gain_results"])!=3):
                raise RuntimeError("Original EEG, event or source-count anomaly")
            total_original_events+=5
            gains=row["registered_gain_results"]
            if [g["alpha"] for g in gains]!=plan["training_sensor_gain_alpha"]:
                raise RuntimeError("Missing/reordered registered gain cells")
            for g in gains:
                n=g["actual_clean_test_event_count"]
                if (n!=5
                    or not g["raw_heldout_test_samples_unchanged"]
                    or not g["train_labels_used_only_in_training"]
                    or not g["standardizer_frozen_on_clean_train_only"]
                    or not (0<=g["baseline_lambda_certified"]
                              <=g["tighter_curvature_certified"]<=n)
                    or g["additional_tight_certificates"]!=
                       g["tighter_curvature_certified"]-
                       g["baseline_lambda_certified"]
                    or g["baseline_abstentions"]!=
                       n-g["baseline_lambda_certified"]
                    or g["tight_abstentions"]!=
                       n-g["tighter_curvature_certified"]
                    or g["certified_false_positives"]!=0
                    or g["observed_class_flips"]>g["tight_abstentions"]
                    or g["tight_parameter_radius"]>
                       g["baseline_parameter_radius"]+1e-12):
                    raise RuntimeError("Unsafe or incomplete certificate in original data")
                if not np.isfinite([
                    g["baseline_parameter_radius"],g["tight_parameter_radius"],
                    g["measured_refit_shift"],g["changed_fit_gradient_l2"]
                ]).all():
                    raise RuntimeError("Original artifact contained nonfinite model evidence")
                if (g["measured_refit_shift"] >
                    g["tight_parameter_radius"]+
                    g["changed_fit_gradient_l2"]/.5+1e-8):
                    raise RuntimeError("Observed actual refit outside reported certificate")
                all_comparisons+=1
                entries.append(g)
        denom=sum(x["actual_clean_test_event_count"] for x in entries)
        assert denom==30
        baseline=sum(x["baseline_lambda_certified"] for x in entries)
        refined=sum(x["tighter_curvature_certified"] for x in entries)
        by_subject[str(person)]={
            "registered_repeated_decisions":denom,
            "lambda_only_certified":baseline,
            "curvature_certified":refined,
            "incremental_coverage_fraction":(refined-baseline)/denom,
            "observed_actual_flip_count":sum(
                x["observed_class_flips"] for x in entries
            ),
            "actual_mean_radius_ratio":float(np.mean([
                x["tight_parameter_radius"]/max(
                    x["baseline_parameter_radius"],1e-15
                ) for x in entries
            ])),
        }
    if total_original_events!=80 or all_comparisons!=48:
        raise RuntimeError("Frozen 80 original tests x three interventions incomplete")
    perperson=np.array([x["incremental_coverage_fraction"]
                       for x in by_subject.values()])
    rng=np.random.default_rng(20261013)
    boots=np.array([
        np.mean(perperson[rng.integers(0,8,size=8)]) for _ in range(2000)
    ])
    by_gain={}
    for gain in plan["training_sensor_gain_alpha"]:
        values=[g for r in allrows for g in r["registered_gain_results"]
                if g["alpha"]==gain]
        if len(values)!=16:
            raise RuntimeError("Missing an original registered gain level")
        by_gain[str(gain)]={
            "original_subject_run_pairs":16,
            "total_compared_decisions":80,
            "baseline_certified":sum(g["baseline_lambda_certified"] for g in values),
            "curvature_certified":sum(g["tighter_curvature_certified"] for g in values),
            "observed_class_flips":sum(g["observed_class_flips"] for g in values)
        }
    person_increment=sum(x["incremental_coverage_fraction"]>0
                         for x in by_subject.values())
    primary=float(np.mean(perperson))
    return {
        "schema":"independent_blind_original_eeg_sensor_gain_certificate_audit_v1",
        "source_run_id":SOURCE_RUN,
        "independent_participant_count":8,
        "original_raw_edf_count":16,
        "real_original_clean_test_events":80,
        "registered_counterfactual_comparisons":48,
        "repeated_certificate_decisions_not_independent":240,
        "original_number_of_false_certifications":0,
        "original_all_sources_have_traceable_sha256":True,
        "primary_mean_subject_coverage_advantage":primary,
        "precommitted_go_no_go_condition_met":(
            primary>=.05 and person_increment>=2
        ),
        "subjects_with_strictly_positive_coverage_gain":person_increment,
        "descriptive_subject_cluster_bootstrap_95_interval":[
            float(x) for x in np.quantile(boots,[.025,.975])
        ],
        "bootstrap_seed":20261013,
        "by_subject":by_subject,
        "by_gain":by_gain,
        "real_EDF_counterfactual_is_synthetic_sensor_gain":True,
        "standard_mathematics_prior_art":True,
        "no_external_peer_review_or_upstream_acceptance":True,
    }


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--original-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    answer=audit_artifacts(args.original_dir)
    args.out.write_text(json.dumps(answer,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("INDEPENDENT_EEG_GAIN_CERT_COHORT_AUDIT",json.dumps({
        k:v for k,v in answer.items()
        if k not in ("by_subject","by_gain")
    },sort_keys=True),flush=True)
