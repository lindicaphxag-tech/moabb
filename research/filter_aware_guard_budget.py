"""Filter-specific, source-operator-certified cross-fold leakage budget.

For a fixed length-N linear SOS filtfilt F, train sample selection T and
heldout sample selection H, write A=T F H^T.  For any delta supported on H:
  ||T[F(x+H^T delta)-F(x)]||_2 <= ||A||_F ||delta||_2.
The Frobenius norm is a conservative *computational* upper bound on spectral
norm. It is not a proof that IIR is exactly independent for finite guards,
a patient-generalization claim, or a certificate for nonlinear preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.signal import sosfiltfilt


@dataclass(frozen=True)
class LinearFilterGuard:
    guard_samples: int
    frobenius_gain_bound: float
    spectral_gain_diagnostic: float
    admissible_for_declared_tolerance: bool
    linear_filter_only: bool = True
    physical_independence_certified: bool = False


def cross_fold_filter_operator(
    sos: np.ndarray,
    record_samples: int,
    training_interval: tuple[int, int],
    test_interval: tuple[int, int],
    *,
    max_test_samples: int = 512,
) -> np.ndarray:
    """Build the EXACT fixed-configuration train/test linear map by impulses.

    Only feasible as an oracle on bounded toy/short signals; O(N*|test|).
    The physical filtering backend, boundary padding and recording length
    must be held identical for every counterfactual input.
    """
    if type(record_samples) is not int or record_samples < 64:
        raise ValueError("record length must be a positive feasible integer")
    if (not isinstance(sos, np.ndarray) or sos.ndim != 2
            or sos.shape[1] != 6 or not np.isfinite(sos).all()):
        raise ValueError("finite SciPy SOS (sections x 6) required")
    try:
        lo, hi = training_interval
        a, b = test_interval
    except (ValueError, TypeError):
        raise ValueError("two integer half-open intervals required") from None
    if (any(type(x) is not int for x in (lo, hi, a, b))
            or not 0 <= lo < hi <= record_samples
            or not 0 <= a < b <= record_samples
            or max(lo, a) < min(hi, b)
            or b-a > max_test_samples):
        raise ValueError("nonoverlap bounded half-open train/test intervals required")
    # Fixed initial, final and internal pad behavior of scipy.sosfiltfilt
    # is encoded by applying that very implementation to canonical impulses.
    matrix = np.empty((hi-lo, b-a), dtype=np.float64)
    unit = np.zeros(record_samples, dtype=np.float64)
    for j in range(a, b):
        unit[j] = 1.0
        filtered = sosfiltfilt(sos, unit)
        unit[j] = 0.0
        matrix[:, j-a] = filtered[lo:hi]
    if not np.isfinite(matrix).all():
        raise RuntimeError("linear filter operator contains nonfinite coefficients")
    return matrix


def select_filter_guard(
    sos: np.ndarray,
    *,
    record_samples: int,
    test_interval: tuple[int, int],
    training_epoch_samples: int,
    candidate_guards: tuple[int, ...],
    max_amplification: float,
) -> tuple[LinearFilterGuard | None, tuple[LinearFilterGuard, ...]]:
    """Select first fixed guard where ||leak_train||<=eps*||delta_test||.

    A predeclared bound on the *amplitude* of delta is needed for absolute
    distortion guarantees: eps is the relative L2 gain, not an EEG accuracy
    bias or percentage-points bound. Return None, not a false pass, when
    all candidate guards violate the declared budget.
    """
    if (type(training_epoch_samples) is not int
            or training_epoch_samples <= 0
            or not isfinite(max_amplification)
            or max_amplification < 0
            or not candidate_guards
            or any(type(g) is not int or g < 0 for g in candidate_guards)
            or tuple(sorted(set(candidate_guards))) != tuple(candidate_guards)):
        raise ValueError("invalid prospective guard register/tolerance")
    a, _ = test_interval
    rows = []
    for g in candidate_guards:
        start, stop = a-g-training_epoch_samples, a-g
        if start < 0:
            raise ValueError("guard moves training interval before record start")
        op = cross_fold_filter_operator(
            sos, record_samples, (start, stop), test_interval
        )
        # Frobenius is an *upper* bound on every singular value.
        gain = float(np.linalg.norm(op, ord="fro"))
        # SVD is only a diagnostic independent numerical oracle, not needed
        # to obtain the upper bound or run a guard decision in production.
        spectral = float(np.linalg.svd(op, compute_uv=False)[0])
        rows.append(LinearFilterGuard(
            guard_samples=g, frobenius_gain_bound=gain,
            spectral_gain_diagnostic=spectral,
            admissible_for_declared_tolerance=gain <= max_amplification
        ))
    admissible = next(
        (row for row in rows if row.admissible_for_declared_tolerance), None
    )
    return admissible, tuple(rows)
