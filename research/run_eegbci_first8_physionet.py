"""Source-locked, genuinely public PhysioNet EEGBCI 8-subject LOSO pilot.

Run ID 2026-10-10. Downloads original EDF files via MNE, records EDF SHA256,
honestly includes all eight heldout subjects and does not mix motor imagery
(4,8,12) with hand-vs-foot events (6,10,14).

Strong cross-subject CSP+LDA is fit on exact same source epochs, never heldout.
The candidate and Ledoit-Wolf LDA use identical EEG log-PSD features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from research.transfer_consistency_shrinkage import (
    nested_subject_holdout, paired_subject_summary,
)

PROTO=Path("research/EEGBCI_FIRST8_PREOUTCOME_20261010.json")
PROTO_HASH="TO_BE_FROZEN"


def load_real_subject(subject:int, runs:list[int], directory:Path):
    import mne
    from mne.datasets import eegbci
    if runs!=[4,8,12]:
        raise ValueError("Motor-imagery left/right event semantics require [4,8,12]")
    paths=eegbci.load_data(subject,runs,path=str(directory),
                           update_path=False,verbose="ERROR")
    if len(paths)!=3:
        raise RuntimeError("Three EDF recordings required for every registered subject")
    features=[]; classes=[]; tensor=[]; hashes={}
    electrodes=None
    for file in paths:
        file=Path(file)
        hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        raw=mne.io.read_raw_edf(str(file),preload=True,verbose="ERROR")
        eegbci.standardize(raw)
        raw.pick_types(eeg=True,exclude=())
        # Apply exactly the same fixed 8-30Hz filter to every source and
        # heldout run. No source/target label is used in preprocessing.
        raw.filter(l_freq=8.,h_freq=30.,method="iir",
                   iir_params={"order":4,"ftype":"butter"},
                   verbose="ERROR")
        if electrodes is None:
            electrodes=tuple(raw.ch_names)
        elif tuple(raw.ch_names)!=electrodes:
            raise RuntimeError("Electrode identity mismatch across runs/subjects")
        sf=float(raw.info["sfreq"])
        if sf < 64:
            raise RuntimeError("Insufficient PhysioNet EEG sampling rate")
        events,_=mne.events_from_annotations(
            raw,event_id={"T1":1,"T2":2},verbose="ERROR")
        ep=mne.Epochs(raw,events,event_id={"T1":1,"T2":2},
                      tmin=.5,tmax=2.5,baseline=None,preload=True,
                      reject_by_annotation=True,verbose="ERROR")
        arr=ep.get_data(copy=True)
        y=np.asarray(ep.events[:,2]-1,dtype=int)
        if arr.ndim!=3 or min((y==0).sum(),(y==1).sum())<2:
            raise RuntimeError("No balanced imagery trials in registered EDF")
        freqs,pxx=welch(arr,fs=sf,nperseg=min(int(sf),arr.shape[-1]),
                       noverlap=min(int(sf)//2,arr.shape[-1]//2),axis=-1)
        bands=((8,12),(12,20),(20,30))
        feats=[]
        for lower,upper in bands:
            band=(freqs>=lower)&(freqs<upper)
            if not band.any():
                raise RuntimeError("Empty spectral band")
            feats.append(np.log10(np.maximum(pxx[:,:,band].mean(axis=-1),1e-20)))
        # features (trials, channel x band) with named band order fixed.
        features.append(np.concatenate(feats,axis=1))
        classes.append(y)
        tensor.append(arr)
        raw.close()
    return (np.concatenate(features),np.concatenate(classes),
            np.concatenate(tensor),electrodes,hashes)


def _csp_loso(X3,y,subjects):
    from mne.decoding import CSP
    cv={}
    for hold in sorted(np.unique(subjects)):
        source=subjects!=hold
        csp=CSP(n_components=6,reg="ledoit_wolf",
                log=True,norm_trace=False)
        # CSP feature extraction is fitted ONLY on training subjects.
        Xtr=csp.fit_transform(X3[source],y[source])
        Xte=csp.transform(X3[~source])
        lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
        lda.fit(Xtr,y[source])
        pred=lda.decision_function(Xte)
        cv[str(hold)]=float(roc_auc_score(y[~source],pred))
    return cv


def preflight():
    from subprocess import check_output
    blob=check_output(["git","hash-object",str(PROTO)],text=True).strip()
    if blob!=PROTO_HASH:
        raise ValueError("PREOUTCOME protocol SHA does not match published cohort registry")
    p=json.loads(PROTO.read_text(encoding="utf-8"))
    if (p["subjects"]!=list(range(1,9)) or p["runs"]!=[4,8,12] or
        p["ridge_grid"]!=[.03,.3,3] or
        p["transfer_penalty_grid"]!=[0,1,4]):
        raise ValueError("Subject/sample/calibration grid drift")
    return p


def run(directory:Path,output:Path):
    start=time.monotonic()
    p=preflight()
    XX=[];yy=[];g=[];tensor=[];recordings={}
    channel_map=None
    for subject in p["subjects"]:
        x,y,raw,names,sha=load_real_subject(subject,p["runs"],directory)
        if channel_map is None:
            channel_map=names
        elif names!=channel_map:
            raise ValueError("Cross-subject EEG channels mismatch; do not remap silently")
        XX.append(x);yy.append(y);tensor.append(raw)
        g.extend([subject]*len(y))
        recordings[str(subject)]={"runs":p["runs"],"edf_sha256":sha,
                                  "n_epochs":len(y),"n_class0":int((y==0).sum()),
                                  "n_class1":int((y==1).sum())}
        print("FIRST_REAL_EEG_SOURCE",
              json.dumps({"subject":subject,**recordings[str(subject)]},sort_keys=True),
              flush=True)
    X=np.concatenate(XX);y=np.concatenate(yy);groups=np.asarray(g)
    epochs=np.concatenate(tensor)
    if len(np.unique(groups))!=8 or X.shape[1]!=3*len(channel_map):
        raise RuntimeError("Invalid frozen first cohort or channel count")
    results=nested_subject_holdout(
        X,y,groups,outer_subjects=p["subjects"],
        base_grid=tuple(p["ridge_grid"]),
        penalty_grid=tuple(p["transfer_penalty_grid"]))
    # Real matched-source CSP is a stronger spatial-filtering baseline. It
    # uses exactly the same 8-30Hz filtered epochs but different representation.
    csp_scores=_csp_loso(epochs,y,groups)
    for result in results:
        result["test_auc_CSP_LDA"]=csp_scores[result["heldout_subject"]]
    summary=paired_subject_summary(results)
    summary["mean_auc"]["test_auc_CSP_LDA"]=float(np.mean(list(csp_scores.values())))
    summary["candidate_minus_CSP_mean_auc"]=float(
        np.mean([r["test_auc_transfer_penalized"]-r["test_auc_CSP_LDA"]
                 for r in results]))
    content={
        "schema":"ORIGINAL_PUBLIC_EEGBCI_8_SUBJECT_LOSO_PILOT_V1",
        "not_yet_peer_reviewed":True,
        "genuine_public_physionet_EEG_data_not_synthetic":True,
        "results_include_all_eight_heldout_subjects":True,
        "dataset_reference":"PhysioNet EEG Motor Movement/Imagery, MNE EEGBCI, runs 4/8/12",
        "protocol_sha256_gitblob":PROTO_HASH,
        "source_epoch_groups":recordings,
        "channels_ordered":list(channel_map),
        "epochs":int(len(y)),"n_features":int(X.shape[1]),
        "preprocessing":p["preprocess"],
        "per_independent_heldout_subject":results,
        "summary":summary,
        "environment":{"python":sys.version.split()[0],
                       "platform":platform.platform(),
                       "numpy":np.__version__,
                       "elapsed_seconds":float(time.monotonic()-start)},
        "not_claimed":["CI is not independent laboratory",
                       "8 subjects from one dataset do not justify journal SOTA",
                       "EEG epoch count is not number of independent subjects",
                       "CSP uses distinct representation; no equal-feature claim",
                       "No patient/clinical transfer or cross-dataset robustness tested",
                       "No claimed positive improvement before actual results"],
    }
    output.write_text(json.dumps(content,sort_keys=True,indent=2)+"\n",
                      encoding="utf-8")
    print("ORIGINAL_EEGBCI_FIRST8_REAL_SOURCE_RESULTS",
          json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("preflight","run"),required=True)
    parser.add_argument("--data",type=Path,default=Path("data/eegbci"))
    parser.add_argument("--output",type=Path,default=Path("original_eegbci_first8_results.json"))
    args=parser.parse_args()
    if args.mode=="preflight":
        print("PASS_EEGBCI_FIRST8_PREOUTCOME",preflight()["subjects"])
    else:
        run(args.data,args.output)
