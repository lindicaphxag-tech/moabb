"""Tighter *numerical* fold-perturbation guard from singular values.

Unlike the conservative Frobenius surrogate, the induced L2 norm of the
finite, fixed, linear SOS filtfilt cross-fold map is its largest singular
value. Explicitly exposing the stronger bound can retain additional training
samples for the SAME calibrated relative perturbation tolerance. This is
standard matrix analysis; source-aware application is experimental, not an
original mathematical theorem, a statistical independence proof or a
physically certified absolute-accuracy guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.signal import sosfiltfilt

from research.filter_aware_guard_budget import cross_fold_filter_operator


@dataclass(frozen=True)
class TightGuardCandidate:
    guard_samples: int
    spectral_gain_float64: float
    conservative_numerical_margin: float
    frobenius_gain: float
    spectral_budget_pass: bool
    frobenius_budget_pass: bool
    # Explicit: machine-precision SVD with margin is not interval arithmetic.
    mathematically_verified_float_rounding: bool = False


def compare_tight_filter_guard(
    sos: np.ndarray,
    *,
    record_samples: int,
    test_interval: tuple[int, int],
    training_epoch_samples: int,
    candidate_guards: tuple[int, ...],
    relative_l2_budget: float,
    numerical_relative_margin: float = 1e-9,
    numerical_absolute_margin: float = 1e-12,
) -> tuple[TightGuardCandidate | None, TightGuardCandidate | None,
           tuple[TightGuardCandidate, ...]]:
    """Compute first feasible spectral and Frobenius guards, respectively.

    For fixed finite linear map A=T F H^T the exact induced norm is sigma_max(A).
    This implementation uses a full float64 SVD, adding a small stated
    numerical margin for decision stability; it does NOT supply a formal
    floating-point interval proof. A perturbation with arbitrary magnitude
    must be bounded by its own L2 norm; this budget limits gain only.

    Return None for either method when the frozen guard register fails.
    """
    if (type(training_epoch_samples) is not int or training_epoch_samples<1
            or not candidate_guards or any(
                type(g) is not int or g < 0 for g in candidate_guards
            )
            or tuple(sorted(set(candidate_guards)))!=tuple(candidate_guards)
            or not all(isfinite(float(x)) and float(x)>=0 for x in (
                relative_l2_budget,numerical_relative_margin,
                numerical_absolute_margin
            ))):
        raise ValueError("Invalid filter tolerance or prospective guard register")
    a,_=test_interval
    rows=[]
    for guard in candidate_guards:
        start=a-guard-training_epoch_samples
        if start<0:
            raise ValueError("Guard selected impossible pre-record training window")
        A=cross_fold_filter_operator(
            sos,record_samples,(start,a-guard),test_interval
        )
        singular=np.linalg.svd(A,compute_uv=False)
        sigma=float(singular[0])
        frob=float(np.linalg.norm(A,ord="fro"))
        # This is an explicit numerical guard, not an exact verified interval.
        margin=float(numerical_absolute_margin
                     + numerical_relative_margin*sigma)
        upper=float(np.nextafter(sigma+margin,np.inf))
        rows.append(TightGuardCandidate(
            guard,sigma,upper,frob,
            upper<=relative_l2_budget,
            frob<=relative_l2_budget,
        ))
    tight=next((x for x in rows if x.spectral_budget_pass),None)
    loose=next((x for x in rows if x.frobenius_budget_pass),None)
    return tight,loose,tuple(rows)


def worst_direction_counterfactual(
    sos: np.ndarray,
    *,
    record_samples: int,
    train_window: tuple[int,int],
    test_window: tuple[int,int],
) -> dict:
    """Independent actual filtfilt replay of the top singular-vector fault.

    The selected delta is adversarial in the finite *signal* model, not a
    naturally occurring EEG event and not independent real-world validation.
    """
    A=cross_fold_filter_operator(sos,record_samples,train_window,test_window)
    _,s,vh=np.linalg.svd(A,full_matrices=False)
    d=vh[0].copy()
    n=np.linalg.norm(d)
    if not n>0:
        raise ValueError("Degenerate singular direction")
    d/=n
    injected=np.zeros(record_samples)
    injected[test_window[0]:test_window[1]]=d
    actual=sosfiltfilt(sos,injected)[train_window[0]:train_window[1]]
    return {
        "declared_perturbation_l2":float(np.linalg.norm(d)),
        "observed_train_change_l2":float(np.linalg.norm(actual)),
        "operator_predicted_train_change_l2":float(np.linalg.norm(A@d)),
        "computed_induced_gain":float(s[0]),
        "no_clinical_eeg_or_classifier_accuracy_claim":True,
    }
