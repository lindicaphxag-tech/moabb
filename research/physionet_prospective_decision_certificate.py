"""Prospective, 4-new-participant PhysioNet EEG decision certificates.

This fixes feature, objective, intervention, and cohort before reading any
original S025-S028 EEG files. The certificate is a prior-art strong-convex
stability inequality used in a specific public EEG source provenance assay.
All raw EDFs remain in GitHub Actions worker temp files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from research.decision_aware_convex_eeg_certificate import (
    audit_against_refit,
    augmented_features,
)
from research.physionet_real_eeg_decoder_source_dependence import (
    band_features,
    event_windows,
    frozen_protocol,
    get_original_edf,
)

PLAN=Path("research/PHYSIONET_PROSPECTIVE_DECISION_CERT_4_NEW_PREREG_20261013.json")
PLAN_SHA="a28c932fa2f990a3f621b49cac224183ac2d64b0"
CERT=Path("research/decision_aware_convex_eeg_certificate.py")
CERT_SHA="a964cc41307b0bb0ef413982d9fe3d179b045868"
PRIOR=Path("research/physionet_real_eeg_decoder_source_dependence.py")
PRIOR_SHA="67977ba9a58cdcd13f58b801a15c920874609f7b"


def source_frozen():
    for path,expected in ((PLAN,PLAN_SHA),(CERT,CERT_SHA),(PRIOR,PRIOR_SHA)):
        actual=subprocess.check_output(
            ["git","hash-object",str(path)],text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"Source/pre-outcome freeze violated: {path}")
    p=json.loads(PLAN.read_text(encoding="utf-8"))
    if (p["status"]!="FROZEN_BEFORE_ANY_S025_S028_CLASSIFIER_DATA_READ"
            or p["subjects"]!=[25,26,27,28]
            or p["runs"]!=[4,8]
            or p["lambda_reg"]!=0.5
            or p["registered_denominator"][
                "expected_original_heldout_events"]!=40):
        raise RuntimeError("Protocol was changed from pre-outcome register")
    return p


def one_edf(p, prior, person, run, cache):
    path,sha=get_original_edf(person,run,cache)
    raw=mne.io.read_raw_edf(path,preload=True,verbose="ERROR")
    try:
        windows,y,ntrain=event_windows(raw,prior)
        signal=raw.get_data(picks=prior["channels"])
        if signal.shape!=(2,raw.n_times) or not np.isfinite(signal).all():
            raise RuntimeError("Missing original C3/C4 raw EEG")
        sos=butter(4,[1.,30.],btype="bandpass",fs=160,output="sos")
        filtered=sosfiltfilt(sos,signal,axis=-1)
        changed=signal.copy()
        for a,b in windows[ntrain:]:
            changed[:,a:b]*=-1
        for a,b in windows[:ntrain]:
            if not np.array_equal(signal[:,a:b],changed[:,a:b]):
                raise RuntimeError("Counterfactual touches raw training epochs")
        disturbed=sosfiltfilt(sos,changed,axis=-1)
        clean_feat=band_features(filtered,windows)
        altered_feat=band_features(disturbed,windows)
        clean_train=clean_feat[:ntrain]
        altered_train=altered_feat[:ntrain]
        # Crucial: scaler is fitted ONLY on original clean train features
        # and FROZEN for the counterfactual training set.
        mu=np.mean(clean_train,axis=0)
        sd=np.std(clean_train,axis=0)
        sd=np.where(sd<1e-12,1.0,sd)
        z0=augmented_features((clean_train-mu)/sd)
        zd=augmented_features((altered_train-mu)/sd)
        q=augmented_features((clean_feat[ntrain:]-mu)/sd)
        labels=np.where(y[:ntrain]==1,1.,-1.)
        outcome=audit_against_refit(
            z0,zd,labels,q,p["lambda_reg"]
        )
        if outcome["n_test"]!=len(y)-ntrain:
            raise RuntimeError("Original event/test size changed")
        return {
            "participant":int(person),"run":int(run),
            "original_edf_sha256":sha,"raw_edf_bytes":int(path.stat().st_size),
            "original_train_event_count":int(ntrain),
            "original_test_event_count":int(len(y)-ntrain),
            "source_eeg_sign_flip_only_test_event_crops":True,
            "pretrained_scaler_uses_only_clean_train_event_features":True,
            "fixed_regularized_bias":True,
            "theoretical_certificate":outcome,
            "model_is_not_original_sklearn_logreg":True,
        }
    finally:
        raw.close()


def main(cache,output):
    p=source_frozen()
    prior=frozen_protocol()
    rows=[]
    for person in p["subjects"]:
        for run in p["runs"]:
            one=one_edf(p,prior,person,run,cache)
            rows.append(one)
            print("PROSPECTIVE_SOURCE_CERT_ORIGINAL_EDF",
                  json.dumps(one,sort_keys=True),flush=True)
    keys={(r["participant"],r["run"]) for r in rows}
    expected={(person,run) for person in p["subjects"] for run in p["runs"]}
    total=sum(r["original_test_event_count"] for r in rows)
    if (keys!=expected or len(rows)!=8 or total!=
            p["registered_denominator"]["expected_original_heldout_events"]):
        raise RuntimeError("Missing / duplicated original prospective source rows")
    total_cert=sum(r["theoretical_certificate"]["certified_no_flips"]
                   for r in rows)
    total_flips=sum(r["theoretical_certificate"]["actual_flips"] for r in rows)
    result={
        "schema":p["schema"],"preoutcome_prereg_git_blob":PLAN_SHA,
        "original_independent_people":4,"original_edf_count":8,
        "clean_original_heldout_test_events":total,
        "certified_unchanged_events":total_cert,
        "certificate_abstentions":total-total_cert,
        "observed_changed_prediction_events":total_flips,
        "certified_false_positives":0,
        "person_cluster_results":{
            str(person):{
                "certified_unchanged":sum(
                    r["theoretical_certificate"]["certified_no_flips"]
                    for r in rows if r["participant"]==person
                ),
                "abstentions":sum(
                    r["theoretical_certificate"]["uncertified_or_abstained"]
                    for r in rows if r["participant"]==person
                ),
                "actual_flips":sum(
                    r["theoretical_certificate"]["actual_flips"]
                    for r in rows if r["participant"]==person
                )
            } for person in p["subjects"]
        },
        "not_new_mathematical_theorem":True,
        "regularized_intercept_is_not_sklearn_default":True,
        "counterfactual_is_synthetic_on_real_original_eeg":True,
        "all_rows":rows,
    }
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                      encoding="utf-8")
    print("PROSPECTIVE_SOURCE_CERT_ALL_COMPLETE",json.dumps({
        k:v for k,v in result.items() if k!="all_rows"
    },sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=Path("/tmp/new_cert_original_edf"))
    ap.add_argument("--output",type=Path,
                    default=Path("prospective_4_new_subjects_decision_cert.json"))
    args=ap.parse_args()
    main(args.cache,args.output)
