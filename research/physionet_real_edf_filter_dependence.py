"""Genuine PhysioNet EEGBCI EDF counterfactual: IIR-dependent CV windows.

Eight public EDF recordings (four people, two runs each), 64 exact 1-channel
window/guard perturbation conditions over C3/C4. No classification model
accuracy, natural-world label leakage, or group-level independent replication
claim. No EDF/samples uploaded to GitHub; report immutable file SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

PLAN_PATH=Path("research/PHYSIONET_EDF_FOLD_FILTER_PREREG_20261013.json")
PLAN_BLOB="8dca2a3c9f287967812dd9a5bec2a9d1ba90c120"


def verify_freeze():
    import subprocess
    actual=subprocess.check_output(
        ["git","hash-object",str(PLAN_PATH)],text=True
    ).strip()
    if actual!=PLAN_BLOB:
        raise RuntimeError("Original source/preregister content altered")
    spec=json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if (spec["status"]!="FROZEN_BEFORE_DATA_OUTCOMES"
            or spec["subjects"]!=[1,2,3,4]
            or spec["runs"]!=[4,8]
            or spec["guard_register_samples"]!=[0,80,320,640]
            or spec["heldout_source_interval_samples"]!=[9600,9920]
            or spec["raw_sampling_rate_hz"]!=160
            or spec["expected_counterfactual_cells"]!=64):
        raise RuntimeError("Original EDF cohort/perturbation denominator drift")
    return spec


def _edf_url(spec,subject,run):
    prefix=f"S{subject:03d}"
    fname=f"{prefix}R{run:02d}.edf"
    return spec["source_root"]+f"{prefix}/{fname}",fname


def fetch_exact_edf(spec,subject,run,cache):
    url,name=_edf_url(spec,subject,run)
    out=cache/f"S{subject:03d}"/name
    out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        tmp=out.with_suffix(".edf.part")
        if tmp.exists():
            tmp.unlink()
        try:
            with urlopen(url,timeout=90) as response, open(tmp,"wb") as target:
                while True:
                    block=response.read(1<<20)
                    if not block: break
                    target.write(block)
            if tmp.stat().st_size<2048:
                raise RuntimeError("PhysioNet download suspiciously small")
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    if out.stat().st_size<2048:
        raise RuntimeError("EDF is missing or truncated")
    return out,digest


def record_pair(spec,subject,run,cache):
    path,digest=fetch_exact_edf(spec,subject,run,cache)
    raw=mne.io.read_raw_edf(path,preload=True,verbose="ERROR")
    try:
        mne.datasets.eegbci.standardize(raw)
        sfreq=float(raw.info["sfreq"])
        if sfreq!=spec["raw_sampling_rate_hz"]:
            raise RuntimeError(f"Wrong real PhysioNet EDF sampling rate: {sfreq}")
        names=spec["channels"]
        if not set(names).issubset(set(raw.ch_names)):
            raise RuntimeError(f"Expected real EEG channels not found: {names}")
        a,b=spec["heldout_source_interval_samples"]
        guards=spec["guard_register_samples"]
        epoch_samples=spec["train_window_length_samples"]
        if raw.n_times<b+100:
            raise RuntimeError("Real EDF recording too short")
        signal=raw.get_data(picks=names).astype(np.float64,copy=False)
        if signal.shape!=(len(names),raw.n_times) or not np.isfinite(signal).all():
            raise RuntimeError("Unexpected EDF channel count / nonfinite samples")
        sos=butter(4,[1.0,30.0],btype="bandpass",fs=sfreq,output="sos")
        baseline=sosfiltfilt(sos,signal,axis=-1)
        changed=signal.copy()
        changed[:,a:b]*=-1
        if not np.array_equal(changed[:,:a],signal[:,:a]) or not np.array_equal(
            changed[:,b:],signal[:,b:]
        ):
            raise RuntimeError("Heldout perturbation changed training raw source")
        counter=sosfiltfilt(sos,changed,axis=-1)
        # A methodological negative control, NOT a proposed deployable EEG
        # pipeline: mask the same held-out source region BEFORE filtering.
        mask_original=signal.copy()
        mask_changed=changed.copy()
        mask_original[:,a:b]=0
        mask_changed[:,a:b]=0
        check_a=sosfiltfilt(sos,mask_original,axis=-1)
        check_b=sosfiltfilt(sos,mask_changed,axis=-1)
        rows=[]
        for ci,channel in enumerate(names):
            heldout_delta_norm=float(np.linalg.norm(
                changed[ci,a:b]-signal[ci,a:b]
            ))
            if heldout_delta_norm<=0:
                raise RuntimeError("Trivially zero synthetic perturbation")
            for guard in guards:
                lo,hi=a-guard-epoch_samples,a-guard
                if lo<0 or hi> a or hi<=lo:
                    raise RuntimeError("Training window outside prereg interval")
                before=baseline[ci,lo:hi]
                after=counter[ci,lo:hi]
                abs_effect=float(np.linalg.norm(after-before))
                mask_effect=float(np.linalg.norm(
                    check_a[ci,lo:hi]-check_b[ci,lo:hi]
                ))
                if mask_effect!=0:
                    raise RuntimeError("Negative control changed before filter")
                rows.append({
                    "participant":subject,"run":run,"channel":channel,
                    "guard_samples":guard,
                    "raw_train_test_overlap_samples":0,
                    "train_window_samples":[lo,hi],
                    "test_interval_samples":[a,b],
                    "train_original_filtered_l2":float(np.linalg.norm(before)),
                    "heldout_raw_intervention_l2":heldout_delta_norm,
                    "crossfold_training_change_l2":abs_effect,
                    "relative_crossfold_l2_gain":abs_effect/heldout_delta_norm,
                    "mask_before_filter_negative_control_l2":mask_effect,
                    "real_eeg_recording":True,
                    "counterfactual_is_artificial_polarity_flip":True,
                    "source_without_raw_patient_samples":True,
                })
        return {"participant":subject,"run":run,"edf_sha256":digest,
                "channel_order":names,
                "edf_bytes":path.stat().st_size,
                "real_recording_samples":raw.n_times,"rows":rows}
    finally:
        raw.close()


def run(cache,out):
    spec=verify_freeze()
    recordings=[]
    for subject in spec["subjects"]:
        for run in spec["runs"]:
            rec=record_pair(spec,subject,run,cache)
            print("REAL_PHYSIONET_EDF_FINISHED",json.dumps({
                "participant":subject,"run":run,"sha256":rec["edf_sha256"],
                "samples":rec["real_recording_samples"],
                "paired_guard_conditions":len(rec["rows"]),
            },sort_keys=True),flush=True)
            recordings.append(rec)
    allrows=[x for record in recordings for x in record["rows"]]
    expect={(s,r,c,g) for s in spec["subjects"] for r in spec["runs"]
            for c in spec["channels"]
            for g in spec["guard_register_samples"]}
    got={(x["participant"],x["run"],x["channel"],x["guard_samples"])
         for x in allrows}
    if got!=expect or len(allrows)!=64:
        raise RuntimeError("Missing/duplicated real EEG source/counterfactual cells")
    med={}
    for guard in spec["guard_register_samples"]:
        cells=[x for x in allrows if x["guard_samples"]==guard]
        med[str(guard)]={
            "real_subject_run_channel_repeated_observations":len(cells),
            "absolute_train_perturbation_l2_median":float(np.median([
                x["crossfold_training_change_l2"] for x in cells
            ])),
            "relative_l2_transfer_gain_median":float(np.median([
                x["relative_crossfold_l2_gain"] for x in cells
            ])),
            "nonzero_real_edf_counterfactual_differences":sum(
                x["crossfold_training_change_l2"]>1e-12 for x in cells
            ),
            "nonzero_mask_before_filter_control":sum(
                x["mask_before_filter_negative_control_l2"]>0 for x in cells
            ),
        }
    by_subject={}
    for subject in spec["subjects"]:
        by_subject[str(subject)]={
            str(g):float(np.mean([
                x["relative_crossfold_l2_gain"] for x in allrows
                if x["participant"]==subject and x["guard_samples"]==g
            ])) for g in spec["guard_register_samples"]
        }
    result={
        "schema":spec["schema"],
        "preregister_git_blob":PLAN_BLOB,
        "source_data_type":"real public EEGBCI EDF raw data",
        "original_public_subjects":spec["subjects"],
        "independent_subject_clusters":len(spec["subjects"]),
        "real_edf_recordings":len(recordings),
        "paired_real_subject_run_channel_intervention_conditions":64,
        "no_real_classifier_or_labels_used":True,
        "not_natural_world_artifact_not_external_lab_replication":True,
        "no_derived_patient_eeg_signal_values_committed":True,
        "registered_aggregate":med,
        "per_subject_repeated_measure_summary":by_subject,
        "public_original_file_sha256":[
            {k:v for k,v in x.items() if k in (
                "participant","run","edf_sha256","edf_bytes",
                "real_recording_samples"
            )} for x in recordings
        ],
        "rows":allrows,
    }
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                   encoding="utf-8")
    print("REAL_EDF_CROSSFOLD_COMPLETE",json.dumps({
        "subjects":len(spec["subjects"]),
        "recordings":len(recordings),
        "registered_cells":len(allrows),
        "by_guard":med,
    },sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=Path("/tmp/eegbci_real_edf"))
    ap.add_argument("--out",type=Path,
                    default=Path("physionet_real_eegbci_filter_perturbation.json"))
    opt=ap.parse_args()
    run(opt.cache,opt.out)


if __name__=="__main__":
    main()
