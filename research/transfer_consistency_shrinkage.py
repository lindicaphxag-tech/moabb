"""Subject-blocked transfer-consistency spectral shrinkage research prototype.

This is a candidate method, not an established original theorem or verified
improvement over regularized LDA, CSP/FBCSP or Riemannian baselines.
All supervised tuning is nested inside the SOURCE subjects. Heldout labels are
used only for reporting the final outer-subject AUC, NEVER in model selection.

License/usage: research branch, separate from official MOABB benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _group_validate(X, y, subject) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y)
    subject=np.asarray(subject)
    if (X.ndim!=2 or X.shape[0]!=len(y) or len(y)!=len(subject)
            or X.shape[0]<12 or X.shape[1]<1 or
            not np.isfinite(X).all()):
        raise ValueError("Finite 2D features with aligned epoch labels/subjects required")
    if set(np.unique(y))!={0,1}:
        raise ValueError("Binary labels 0/1 required")
    if len(np.unique(subject))<3:
        raise ValueError("At least 3 truly independent source subjects")
    for sub in np.unique(subject):
        yy=y[subject==sub]
        if set(np.unique(yy))!={0,1} or min(sum(yy==0),sum(yy==1))<2:
            raise ValueError("Each source subject needs both classes and repeated trials")
    return X,y.astype(np.int64),subject


@dataclass(frozen=True)
class SpectralState:
    scaler: StandardScaler
    eigvec: np.ndarray
    eigval: np.ndarray
    mean_diff: np.ndarray
    offset: np.ndarray
    coherence: np.ndarray
    source_subjects: tuple[str,...]


def _subject_equal_spectral_state(X, y, groups) -> SpectralState:
    X,y,groups=_group_validate(X,y,groups)
    subs=np.unique(groups)
    # Each independent subject contributes equally, even if the number of
    # correlated trials differs. Never fit this normalization on heldout data.
    stacked=[]
    for s in subs:
        arr=X[groups==s]
        stacked.append(arr)
    sub_mean=np.stack([a.mean(axis=0) for a in stacked])
    grand_center=sub_mean.mean(axis=0)
    pooled_second=np.mean(np.stack([np.mean((a-grand_center)**2,axis=0)
                                    for a in stacked]),axis=0)
    scale=np.sqrt(np.maximum(pooled_second,1e-12))
    scaler=StandardScaler()
    scaler.mean_=grand_center
    scaler.scale_=scale
    scaler.var_=scale**2
    scaler.n_features_in_=X.shape[1]
    deltas=[]
    covs=[]
    for s in subs:
        Xi=scaler.transform(X[groups==s])
        yi=y[groups==s]
        mean0=Xi[yi==0].mean(axis=0)
        mean1=Xi[yi==1].mean(axis=0)
        deltas.append(mean1-mean0)
        centered=Xi-np.where(yi[:,None]==0,mean0,mean1)
        covs.append(centered.T@centered/len(Xi))  # invariant to duplicated trials of one source
    sigma=np.mean(np.stack(covs),axis=0)
    vals,vecs=eigh((sigma+sigma.T)/2,check_finite=True)
    vals=np.maximum(vals,0)
    projected=np.stack(deltas)@vecs
    mean_proj=projected.mean(axis=0)
    power=np.mean(projected**2,axis=0)
    k=len(subs)
    # Chance-corrected squared cross-subject discriminant alignment.
    # For incoherent signs this is clipped to zero. One source gets no
    # artificial extra weight from having more epochs.
    numerator=np.maximum(0., k*mean_proj**2-power)
    coherence=np.clip(numerator/np.maximum((k-1)*power,1e-12),0,1)
    weighted_diff=np.mean(np.stack(deltas),axis=0)
    # Subject-equal class midpoint, on source subjects only.
    midpoints=[]
    for s in subs:
        xx=scaler.transform(X[groups==s])
        yy=y[groups==s]
        midpoints.append((xx[yy==0].mean(axis=0)+xx[yy==1].mean(axis=0))/2)
    return SpectralState(scaler,vecs,vals,weighted_diff,
                         np.mean(midpoints,axis=0),coherence,
                         tuple(str(s) for s in subs))


def transfer_scores(state:SpectralState,X, base:float, extra:float):
    if not (np.isfinite(base) and base>0 and
            np.isfinite(extra) and extra>=0):
        raise ValueError("Positive ridge and nonnegative transfer penalty required")
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=len(state.eigval):
        raise ValueError("Mismatched inference features")
    # More regularization specifically for inconsistent discriminative
    # directions. Base lambda remains >0 even for rank-deficient EEG features.
    shrink=state.eigval+base*(1+extra*(1-state.coherence))
    direction=state.eigvec@((state.eigvec.T@state.mean_diff)/shrink)
    return (state.scaler.transform(X)-state.offset)@direction


def _source_nested_select(X,y,groups,base_grid,penalty_grid):
    X,y,groups=_group_validate(X,y,groups)
    distinct=np.unique(groups)
    choices=sorted(product(base_grid,penalty_grid),
                   key=lambda pair:(pair[1]>0,pair[1],pair[0]))
    if not choices or any(b<=0 or a<0 for b,a in choices):
        raise ValueError("Hyperparameter grid must be nonempty and valid")
    # Disjoint inner validation by whole SUBJECT, never by EEG epoch.
    fold_scores={choice:[] for choice in choices}
    for hidden in distinct:
        tr=groups!=hidden
        if len(np.unique(groups[tr]))<3:
            raise ValueError("Need at least four training subjects for nested LO-subject-out")
        state=_subject_equal_spectral_state(X[tr],y[tr],groups[tr])
        for choice in choices:
            pred=transfer_scores(state,X[~tr],*choice)
            fold_scores[choice].append(float(roc_auc_score(y[~tr],pred)))
    # Source-only performance and conservative tie rule (penalty zero first).
    selected=max(choices,key=lambda choice:
                 (np.mean(fold_scores[choice]),-int(choice[1]>0),
                  -choice[1],-choice[0]))
    return selected,{f"base={p[0]};penalty={p[1]}":np.mean(v)
                     for p,v in fold_scores.items()}


def nested_subject_holdout(X,y,subjects,*,outer_subjects=None,
                           base_grid=(0.03,0.3,3.0),
                           penalty_grid=(0.,1.,4.))->list[dict[str,Any]]:
    X,y,subjects=_group_validate(X,y,subjects)
    outer=np.unique(subjects) if outer_subjects is None else np.asarray(outer_subjects)
    if len(set(outer))!=len(outer) or not set(outer)<=set(np.unique(subjects)):
        raise ValueError("Outer registry contains repeated or unknown subjects")
    results=[]
    for subject in outer:
        is_test=subjects==subject
        source_set=tuple(str(s) for s in np.unique(subjects[~is_test]))
        if str(subject) in source_set:
            raise AssertionError("Holdout participant leaked to model selection")
        chosen,inner_scores=_source_nested_select(
            X[~is_test],y[~is_test],subjects[~is_test],base_grid,penalty_grid)
        spectral=_subject_equal_spectral_state(X[~is_test],y[~is_test],
                                                subjects[~is_test])
        fitted=transfer_scores(spectral,X[is_test],*chosen)
        # Baseline fit uses only same exact SOURCE data/feature normalization.
        baseline=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
        source_x=spectral.scaler.transform(X[~is_test])
        target_x=spectral.scaler.transform(X[is_test])
        baseline.fit(source_x,y[~is_test])
        baseline_scores=baseline.decision_function(target_x)
        # Zero-penalty ablation under the exact SAME selected base ridge.
        no_transfer=transfer_scores(spectral,X[is_test],chosen[0],0.)
        result={
            "heldout_subject":str(subject),
            "training_subjects":list(source_set),
            "n_train_subjects":len(source_set),
            "n_test_epochs":int(is_test.sum()),
            "source_only_selected_base":float(chosen[0]),
            "source_only_selected_transfer_penalty":float(chosen[1]),
            "source_only_inner_AUC":float(max(inner_scores.values())),
            "test_auc_transfer_penalized":float(roc_auc_score(y[is_test],fitted)),
            "test_auc_same_base_no_transfer":float(roc_auc_score(y[is_test],no_transfer)),
            "test_auc_LDA_LedoitWolf_auto":float(roc_auc_score(y[is_test],baseline_scores)),
            "test_labels_never_accessed_in_training":True,
            "pipeline_scope":"PUBLIC_EEG_PILOT_NOT_COMPLETE_JOURNAL_BENCHMARK",
        }
        results.append(result)
    return results


def paired_subject_summary(rows:list[dict[str,Any]]) -> dict[str,Any]:
    if not rows or len({x["heldout_subject"] for x in rows})!=len(rows):
        raise ValueError("Require unique independent outer heldout subjects")
    keys=["test_auc_transfer_penalized","test_auc_same_base_no_transfer",
          "test_auc_LDA_LedoitWolf_auto"]
    means={k:float(np.mean([r[k] for r in rows])) for k in keys}
    delta=np.array([r[keys[0]]-r[keys[2]] for r in rows])
    # Paired whole-subject signs are meaningful units, not trials/epochs.
    from itertools import product as all_signs
    if len(rows)<=16:
        null=[abs(np.mean(delta*np.asarray(signs))) for signs in
              all_signs((-1,1),repeat=len(rows))]
        p=float(np.mean(np.asarray(null)>=abs(float(delta.mean()))-1e-12))
    else:
        p=None
    return {"n_independent_outer_subjects":len(rows),
            "mean_auc":means,
            "candidate_minus_LedoitWolf_mean_auc":float(delta.mean()),
            "paired_exact_subject_signflip_p_exploratory":p,
            "exploratory_p_not_confirmatory_multidataset_result":True,
            "claim_prohibitions":[
                "Do not use per-epoch IID confidence intervals",
                "Do not claim improvement if paired delta is zero/negative",
                "No heldout subject label may select hyperparameters",
                "Pilot is not external adoption nor completed journal validation",
                "Only one public EEG dataset is not broad clinical generalization",
            ]}
