"""Independent *original-artifact* audit of prospective real EEG certificates.

The original 8 EDF file hashes and all reference/refit parameter bounds are
rechecked from the separate original GitHub Actions outcome artifact. This
audits claimed numbers and formal inequalities but is not a different
laboratory, not fresh clinical data, and not a proof of novel mathematics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

PLAN=Path("research/PHYSIONET_PROSPECTIVE_DECISION_CERT_4_NEW_PREREG_20261013.json")
PLAN_BLOB="a28c932fa2f990a3f621b49cac224183ac2d64b0"
SOURCE_RUN_ID=38351450509


def audit(original):
    sha=subprocess.check_output(["git","hash-object",str(PLAN)],text=True).strip()
    if sha!=PLAN_BLOB:
        raise RuntimeError("Prospective cohort design not source-frozen")
    p=json.loads(PLAN.read_text(encoding="utf-8"))
    data=json.loads(original.read_text(encoding="utf-8"))
    if (
        data.get("schema")!=p["schema"]
        or data.get("preoutcome_prereg_git_blob")!=PLAN_BLOB
        or data.get("original_independent_people")!=4
        or data.get("original_edf_count")!=8
        or data.get("clean_original_heldout_test_events")!=40
        or data.get("certified_false_positives")!=0
        or len(data.get("all_rows",[]))!=8
    ):
        raise ValueError("Published primary original-study denominator or provenance mismatch")
    rows=data["all_rows"]
    expected={(s,r) for s in p["subjects"] for r in p["runs"]}
    keys={(r["participant"],r["run"]) for r in rows}
    if expected!=keys or len(keys)!=8:
        raise ValueError("Unexpected original participant or duplicate run")
    number_events=0
    number_cert=0
    number_flips=0
    count_saturated=0
    for row in rows:
        c=row["theoretical_certificate"]
        person=row["participant"]
        if (
            row["original_edf_sha256"]=="" or
            len(row["original_edf_sha256"])!=64
            or row["raw_edf_bytes"]<2048
            or not row["source_eeg_sign_flip_only_test_event_crops"]
            or not row["pretrained_scaler_uses_only_clean_train_event_features"]
            or not row["fixed_regularized_bias"]
            or row["original_train_event_count"]<5
            or row["original_test_event_count"]<1
            or c["n_test"]!=row["original_test_event_count"]
            or c["certified_no_flips"]+
                c["uncertified_or_abstained"]!=c["n_test"]
            or c["actual_flips"]<0 or
                c["actual_flips"]>c["uncertified_or_abstained"]
            or c["certificate_is_one_sided"] is not True
            or c["not_original_new_mathematical_theorem"] is not True
        ):
            raise ValueError(f"Source/CV/certificate contract violated for S{person:03d}")
        r=float(c["refit_gradient_residual"])
        upper=float(c["worst_shift_upper"])
        observed=float(c["observed_parameter_shift"])
        if (
            not np.isfinite([r,upper,observed]).all()
            or min(r,upper,observed)<0
            or observed>upper+r/p["lambda_reg"]+1e-8
            or abs(c["certificate_coverage"]-
                   c["certified_no_flips"]/c["n_test"])>1e-12
        ):
            raise RuntimeError("Numerical refit or strong convexity certificate false")
        number_events+=c["n_test"]
        number_cert+=c["certified_no_flips"]
        number_flips+=c["actual_flips"]
        count_saturated+=int(c["certified_no_flips"]==c["n_test"])
    if (
        data["certified_unchanged_events"]!=number_cert
        or data["certificate_abstentions"]!=number_events-number_cert
        or data["observed_changed_prediction_events"]!=number_flips
        or number_events!=40
        or sorted(data["person_cluster_results"])!=["25","26","27","28"]
    ):
        raise ValueError("Reported counts do not derive from original real EEG rows")
    for person in p["subjects"]:
        sub=[r for r in rows if r["participant"]==person]
        row=data["person_cluster_results"][str(person)]
        if (
            row["certified_unchanged"]!=sum(
                x["theoretical_certificate"]["certified_no_flips"] for x in sub
            )
            or row["abstentions"]!=sum(
                x["theoretical_certificate"]["uncertified_or_abstained"]
                for x in sub
            )
            or row["actual_flips"]!=sum(
                x["theoretical_certificate"]["actual_flips"] for x in sub
            )
        ):
            raise ValueError("Participant-cluster outcome inconsistent with source rows")
    return {
        "schema":"independent_source_artifact_audit_new_eeg_cert_v1",
        "original_independent_run_id":SOURCE_RUN_ID,
        "verified_original_person_count":4,
        "verified_original_edf_hash_count":8,
        "verified_original_heldout_events":number_events,
        "verified_certified_unchanged":number_cert,
        "verified_abstentions":number_events-number_cert,
        "verified_actual_predicted_flips":number_flips,
        "original_runs_with_full_certificate":count_saturated,
        "certificate_for_specific_synthetic_source_counterfactual":True,
        "mathematical_principle_known_strong_convexity":True,
        "not_external_replication_nor_real_neuroscience_discovery":True,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    summary=audit(args.original)
    args.out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("INDEPENDENT_REAL_EEG_CERT_AUDIT",
          json.dumps(summary,sort_keys=True),flush=True)
