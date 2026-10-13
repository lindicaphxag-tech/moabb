"""Independent raw-EDF replay of frozen PhysioNet crossfold leakage report.

Auditor re-downloads original public EDF and predicts the train change by
filtering ONLY the synthetic perturbation, rather than subtracting two
whole-record filtered outputs. Checks complete original 4x2x2x4 study cells,
file hashes, and all numerical claims. No EEG signals leave the worker.
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

PLAN = Path("research/PHYSIONET_EDF_FOLD_FILTER_PREREG_20261013.json")
PLAN_BLOB = "8dca2a3c9f287967812dd9a5bec2a9d1ba90c120"


def source_reader(cache, subject, run):
    name=f"S{subject:03d}R{run:02d}.edf"
    folder=f"S{subject:03d}"
    target=cache/folder/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        tmp=target.with_suffix(".part")
        try:
            with urlopen(
                f"https://physionet.org/files/eegmmidb/1.0.0/{folder}/{name}",
                timeout=90
            ) as r,open(tmp,"wb") as f:
                while True:
                    chunk=r.read(1<<20)
                    if not chunk:break
                    f.write(chunk)
            if tmp.stat().st_size<2048:
                raise RuntimeError("Short official EDF download")
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    return target,sha


def verify_original_eeg_results(path,cache):
    import subprocess
    if subprocess.check_output(
        ["git","hash-object",str(PLAN)],text=True
    ).strip()!=PLAN_BLOB:
        raise RuntimeError("Original prospective EEG source prereg changed")
    plan=json.loads(PLAN.read_text(encoding="utf-8"))
    x=json.loads(path.read_text(encoding="utf-8"))
    if (x.get("preregister_git_blob")!=PLAN_BLOB
            or x.get("schema")!=plan["schema"]
            or x.get("independent_subject_clusters")!=4
            or x.get("real_edf_recordings")!=8
            or x.get("paired_real_subject_run_channel_intervention_conditions")!=64
            or len(x["rows"])!=64):
        raise ValueError("Missing original prospective real source denominator")
    keys={(row["participant"],row["run"],row["channel"],row["guard_samples"])
          for row in x["rows"]}
    expected={(s,r,c,g) for s in plan["subjects"] for r in plan["runs"]
              for c in plan["channels"] for g in plan["guard_register_samples"]}
    if keys!=expected:
        raise ValueError("Original real EDF pair/guard cases swapped/missing")
    provenance={(z["participant"],z["run"]):z
                for z in x["public_original_file_sha256"]}
    if len(provenance)!=8:
        raise ValueError("EDF file identities missing/duplicated")
    comparisons=0
    for subject in plan["subjects"]:
        for run in plan["runs"]:
            p,sha=source_reader(cache,subject,run)
            record=provenance[(subject,run)]
            if sha!=record["edf_sha256"] or record["edf_bytes"]!=p.stat().st_size:
                raise ValueError("Changed official source EDF bytes")
            raw=mne.io.read_raw_edf(p,preload=True,verbose="ERROR")
            try:
                mne.datasets.eegbci.standardize(raw)
                assert float(raw.info["sfreq"])==160.0
                assert int(raw.n_times)==record["real_recording_samples"]
                eeg=raw.get_data(picks=plan["channels"])
                if eeg.shape[0]!=2 or not np.isfinite(eeg).all():
                    raise RuntimeError("Incorrect actual EDF EEG sensors")
                sos=butter(4,[1.0,30.0],btype="bandpass",
                           fs=160,output="sos")
                for i,channel in enumerate(plan["channels"]):
                    a,b=plan["heldout_source_interval_samples"]
                    # Use a DIFFERENT numerical route from the producer:
                    # linearity gives F(x+delta)-F(x) = F(delta).
                    # This is not a data-privacy leak: delta only exists on worker.
                    delta=np.zeros(eeg.shape[-1])
                    delta[a:b]=-2*eeg[i,a:b]
                    filtered_delta=sosfiltfilt(sos,delta)
                    denom=float(np.linalg.norm(delta[a:b]))
                    if denom<=0:
                        raise RuntimeError("Source perturbation became zero")
                    for guard in plan["guard_register_samples"]:
                        match=[
                            y for y in x["rows"] if
                            (y["participant"],y["run"],y["channel"],y["guard_samples"])
                            ==(subject,run,channel,guard)
                        ]
                        assert len(match)==1
                        row=match[0]
                        lo=a-guard-plan["train_window_length_samples"]
                        hi=a-guard
                        predicted=float(np.linalg.norm(filtered_delta[lo:hi]))
                        got=row["crossfold_training_change_l2"]
                        allowed=1e-11*max(denom,got,predicted)+1e-13
                        if abs(got-predicted)>allowed:
                            raise ValueError(
                                "Whole-record source counterfactual disagrees "
                                "with independent linear-operator perturbation"
                            )
                        if (abs(row["relative_crossfold_l2_gain"]-got/denom)>1e-12
                                or row["mask_before_filter_negative_control_l2"]!=0
                                or row["raw_train_test_overlap_samples"]!=0
                                or row["train_window_samples"]!=[lo,hi]):
                            raise ValueError("Altered guard denominator or control")
                        comparisons+=1
            finally:
                raw.close()
    summaries={}
    for guard in plan["guard_register_samples"]:
        subset=[z for z in x["rows"] if z["guard_samples"]==guard]
        m=x["registered_aggregate"][str(guard)]
        obs=np.array([z["relative_crossfold_l2_gain"] for z in subset])
        if (len(subset)!=16
                or abs(np.median(obs)-m["relative_l2_transfer_gain_median"])>1e-13
                or m["nonzero_mask_before_filter_control"]!=0):
            raise ValueError("Wrong real EEG aggregated effect or control")
        summaries[str(guard)]={
            "source_channel_run_repeats":len(subset),
            "median_relative_gain":float(np.median(obs)),
            "nonzero_signal_effects":sum(
                z["crossfold_training_change_l2"]>1e-12 for z in subset
            ),
        }
    assert comparisons==64
    return {
        "result":"independent_raw_edf_linear_replay_success",
        "distinct_real_participants":4,
        "real_edf_recordings":8,
        "original_counterfactual_cells_verified":comparisons,
        "disjoint_original_train_and_test_sources":True,
        "patient_independent_replication":False,
        "counterfactual_is_artificial_polarity_flip":True,
        "classifier_accuracy_not_measured":True,
        "sos_linear_delta_only_agrees_with_full_record_counterfactual":True,
        "summary":summaries,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=Path("/tmp/edf_independent_audit"))
    parser.add_argument("--out",type=Path,
                        default=Path("independent_real_edf_filter_audit.json"))
    a=parser.parse_args()
    result=verify_original_eeg_results(a.original,a.cache)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("INDEPENDENT_REAL_EDF_SOURCE_AUDIT",
          json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()
