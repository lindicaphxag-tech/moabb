"""Blind real EEGBCI C3 train-gain study: λ versus sound curvature certificates.

Eight entirely new people S029-S036, two original EDF files each.
The perturbation is registered SYNTHETIC gain on raw *training* events only:
it is not natural leakage, a clinical event, an EEG decoding accuracy claim,
or independent outside-lab acceptance. Immutable source/protocol verified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from research.decision_aware_convex_eeg_certificate import (
    augmented_features,
    fit_regularized_logistic,
)
from research.decision_aware_eeg_curvature_refinement import (
    compare_curvature_certificates,
)
from research.physionet_real_eeg_decoder_source_dependence import (
    band_features,
    event_windows,
    frozen_protocol,
    get_original_edf,
)

PLAN=Path("research/UNSEEN_EEG_TRAIN_GAIN_DECISION_CERT_PREREG_20261013.json")
REGISTER="b5b8b92f53f8588c937446193c5c641c5aca09e8"
LOCKED=(
    ("research/physionet_real_eeg_decoder_source_dependence.py",
     "67977ba9a58cdcd13f58b801a15c920874609f7b"),
    ("research/decision_aware_convex_eeg_certificate.py",
     "a964cc41307b0bb0ef413982d9fe3d179b045868"),
    ("research/decision_aware_eeg_curvature_refinement.py",
     "e973cb1a1d9dfcbd4a81cf009c58be2d81d4338d"),
)


def get_plan():
    for path,want in ((str(PLAN),REGISTER),*LOCKED):
        got=subprocess.check_output(["git","hash-object",path],
                                    text=True).strip()
        if got!=want:
            raise RuntimeError(f"Original prospectively frozen source was changed: {path}")
    p=json.loads(PLAN.read_text(encoding="utf-8"))
    if (p["status"]!="FROZEN_BEFORE_NEW_SUBJECT_DATA_AND_COVERAGE_RESULTS"
            or p["subjects"]!=list(range(29,37))
            or p["runs"]!=[4,8]
            or p["training_sensor_gain_alpha"]!=[.05,.2,.5]
            or p["expected_test_events"]!=80):
        raise RuntimeError("Cannot alter prospectively frozen study register")
    return p


def one_original_edf(p,prior,person,run,cache):
    path,sha=get_original_edf(person,run,cache)
    raw=mne.io.read_raw_edf(path,preload=True,verbose="ERROR")
    try:
        windows,y,ntrain=event_windows(raw,prior)
        signal=raw.get_data(picks=p["channels"])
        if (signal.shape!=(2,raw.n_times)
                or not np.isfinite(signal).all()):
            raise RuntimeError("Real EDF original C3/C4 absent or nonfinite")
        train_windows=windows[:ntrain]
        test_windows=windows[ntrain:]
        if len(test_windows)!=5:
            raise RuntimeError("Pre-registered true heldout event count not met")
        filt=butter(4,[1,30],btype="bandpass",fs=160,output="sos")
        clean=sosfiltfilt(filt,signal,axis=-1)
        X=band_features(clean,windows)
        mean=X[:ntrain].mean(axis=0)
        std=X[:ntrain].std(axis=0)
        std=np.where(std<1e-12,1.,std)
        zclean=augmented_features((X[:ntrain]-mean)/std)
        q=augmented_features((X[ntrain:]-mean)/std)
        train_label=np.where(y[:ntrain]==1,1.,-1.)
        lam=.5
        original=fit_regularized_logistic(zclean,train_label,lam)
        seen=[]
        for gain in p["training_sensor_gain_alpha"]:
            altered=signal.copy()
            # Source only in registered TRAIN windows; original C4 and
            # original complete heldout source are untouched.
            for a,b in train_windows:
                altered[0,a:b] *= 1.+gain
            if not np.array_equal(altered[1],signal[1]):
                raise RuntimeError("C4 source unexpectedly modified")
            for a,b in test_windows:
                if not np.array_equal(altered[:,a:b],signal[:,a:b]):
                    raise RuntimeError("Changed raw heldout test input")
            if not np.any(altered[0]!=signal[0]):
                raise RuntimeError("No registered EEG training perturbation")
            Xa=band_features(sosfiltfilt(filt,altered,axis=-1),train_windows)
            zd=augmented_features((Xa-mean)/std)
            changed=fit_regularized_logistic(zd,train_label,lam)
            cert=compare_curvature_certificates(
                original.coefficients,zd,train_label,q,lam,
                refinements=6,
            )
            original_pred=(q@original.coefficients)>0
            changed_pred=(q@changed.coefficients)>0
            flipped=original_pred!=changed_pred
            invalid=np.any(flipped & cert.refined_certified_unflipped)
            if invalid:
                raise RuntimeError("Unsafe certificate: false prediction invariance")
            if np.any(cert.base_certified_unflipped &
                      ~cert.refined_certified_unflipped):
                raise RuntimeError("Refinement paradoxically loses a valid certificate")
            actual_shift=float(np.linalg.norm(
                changed.coefficients-original.coefficients
            ))
            if actual_shift > (
                cert.refined_parameter_radius +
                changed.optimization_grad_l2/lam+1e-8
            ):
                raise RuntimeError("Observed true refit exited bounded parameter ball")
            seen.append({
                "alpha":gain,
                "actual_clean_test_event_count":len(test_windows),
                "baseline_lambda_certified":cert.base_certification_count,
                "tighter_curvature_certified":cert.refined_certification_count,
                "additional_tight_certificates":
                    cert.refined_certification_count-cert.base_certification_count,
                "observed_class_flips":int(flipped.sum()),
                "certified_false_positives":0,
                "baseline_abstentions":len(q)-cert.base_certification_count,
                "tight_abstentions":len(q)-cert.refined_certification_count,
                "baseline_parameter_radius":cert.base_parameter_radius,
                "tight_parameter_radius":cert.refined_parameter_radius,
                "measured_refit_shift":actual_shift,
                "changed_fit_gradient_l2":changed.optimization_grad_l2,
                "raw_heldout_test_samples_unchanged":True,
                "train_labels_used_only_in_training":True,
                "standardizer_frozen_on_clean_train_only":True,
            })
        return {
            "person":person, "run":run,
            "original_edf_sha256":sha, "original_edf_bytes":int(path.stat().st_size),
            "original_train_events":ntrain, "original_clean_test_events":len(test_windows),
            "original_train_classes":[int(np.sum(train_label==v)) for v in (-1.,1.)],
            "original_test_classes_diagnostic_only":[
                int(np.sum(y[ntrain:]==v)) for v in (0,1)
            ],
            "registered_gain_results":seen,
        }
    finally:
        raw.close()


def run_shard(shard,cache,out):
    plan=get_plan()
    prior=frozen_protocol()
    shards={"A":[29,30],"B":[31,32],"C":[33,34],"D":[35,36]}
    if shard not in shards:
        raise ValueError("Unregistered shard")
    rows=[]
    for subject in shards[shard]:
        for run in plan["runs"]:
            row=one_original_edf(plan,prior,subject,run,cache)
            print("REAL_EEG_GAIN_CERT_ORIGINAL",json.dumps(row,sort_keys=True),
                  flush=True)
            rows.append(row)
    if ({(row["person"],row["run"]) for row in rows}
         !={(s,r) for s in shards[shard] for r in plan["runs"]}
         or len(rows)!=4 or sum(r["original_clean_test_events"] for r in rows)!=20
         or any(len(r["registered_gain_results"])!=3 for r in rows)):
        raise RuntimeError("Original participant/run/gain factorial not complete")
    result={
        "schema":plan["schema"],"shard":shard,
        "preoutcome_preregister_git_blob":REGISTER,
        "registered_people":shards[shard],"actual_original_EDFs":4,
        "actual_heldout_clean_event_count":20,
        "records":rows,"original_data_is_public_EEGBCI":True,
        "counterfactual_is_synthetic_train_C3_gain":True,
        "no_novel_mathematical_theorem_or_external_adoption_claim":True,
    }
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                   encoding="utf-8")
    print("REAL_EEG_GAIN_CERT_SHARD_COMPLETE",json.dumps({
        "shard":shard,"people":shards[shard],
        "original_clean_events":20,
        "extra_certified":sum(
            g["additional_tight_certificates"]
            for row in rows for g in row["registered_gain_results"]
        ),"observed_flips":sum(
            g["observed_class_flips"]
            for row in rows for g in row["registered_gain_results"]
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--shard",choices=["A","B","C","D"],required=True)
    a.add_argument("--cache",type=Path,default=Path("/tmp/new_eeg_gain_original_edfs"))
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    run_shard(args.shard,args.cache,args.out)
