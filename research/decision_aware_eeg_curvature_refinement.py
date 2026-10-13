"""A data-dependent tighter (still conservative) strongly convex radius.

Prior-art linear algebra: globally safe logistic-Hessian curvature is derived
over a radius ALREADY justified by the lambda-strongly-convex gradient bound.
Iterating the certified enclosing balls may reduce conservative abstentions.
It is NOT a newly discovered theorem, a formal floating-point interval proof,
or proof of real-world EEG accuracy improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit

from research.decision_aware_convex_eeg_certificate import (
    _validated_training, objective_and_gradient
)


@dataclass(frozen=True)
class CurvatureRefinedCertificate:
    base_parameter_radius: float
    refined_parameter_radius: float
    recorded_curvature_floor: float
    base_certified_unflipped: np.ndarray
    refined_certified_unflipped: np.ndarray
    base_certification_count: int
    refined_certification_count: int
    numerical_float_interval_verified: bool = False


def certified_curvature_radius(theta, z_changed, y, lambda_reg,
                               *, refinements=6):
    """Certified real-number enclosure, implemented in float64.

    The true minimizer lies in B(theta,R0) by strong convexity.
    For any weights w in that ball, |y z_i dot w| <= |y z_i dot theta|
    + ||z_i||*R0. The logistic curvature p(1-p) is minimized at
    the maximal absolute logit. Hence
      Hessian F(w) >= lambda I + Z^T diag(k_i) Z/n,
      k_i = sigmoid(-M_i)*sigmoid(M_i).
    Integrating the Hessian on the segment to the true minimizer
    gives a globally safe stronger radius R1 = ||grad F(theta)|| / mu.
    This is iterated using the nested certified balls.

    Floating arithmetic is stabilized with a small outward slack.
    This does NOT constitute rigorous machine-checked rounding bounds;
    for safety-critical absolute guarantees, use verified intervals.
    """
    z,y=_validated_training(z_changed,y,lambda_reg)
    theta=np.asarray(theta,dtype=np.float64)
    if theta.shape!=(z.shape[1],) or not np.isfinite(theta).all():
        raise ValueError("Invalid reference parameter for certificate")
    if type(refinements) is not int or refinements<0 or refinements>50:
        raise ValueError("Bounded integer refinement count required")
    _,grad=objective_and_gradient(theta,z,y,lambda_reg)
    gn=float(np.linalg.norm(grad))
    base=gn/lambda_reg
    if not np.isfinite(base):
        raise ValueError("Nonfinite initial safe parameter radius")
    R=base
    mu=lambda_reg
    for _ in range(refinements):
        a=np.abs(y*(z@theta))+np.linalg.norm(z,axis=1)*R
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite Hessian envelope")
        k=expit(-a)*expit(a)
        weighted=(z.T*k)@z/len(y)
        # The symmetric feature curvature has nonnegative true eigenvalues.
        # Clamp small negative numerical eigenvalues to zero (safe side).
        min_eig=max(0.,float(np.linalg.eigvalsh(weighted)[0]))
        mu=lambda_reg+min_eig
        candidate=gn/mu
        if not np.isfinite(candidate):
            raise ValueError("Invalid data-informed curvature bound")
        # A small explicitly stated numerical slack, not interval arithmetic.
        candidate=np.nextafter(candidate*(1.+1e-10)+1e-12,np.inf)
        R=min(R,candidate)
    return base,R,mu


def compare_curvature_certificates(theta,z_changed,y,heldout_fixed,lambda_reg,
                                   *,refinements=6):
    q=np.asarray(heldout_fixed,dtype=np.float64)
    if (
        q.ndim!=2 or q.shape[1]!=len(theta)
        or not np.isfinite(q).all()
    ):
        raise ValueError("Nonfinite/mismatched clean heldout input")
    base,R,mu=certified_curvature_radius(
        theta,z_changed,y,lambda_reg,refinements=refinements
    )
    norms=np.linalg.norm(q,axis=1)
    margins=np.abs(q@theta)
    eps=1e-14
    classic=margins>norms*base+eps
    tight=margins>norms*R+eps
    if np.any(classic & ~tight) or R>base:
        raise RuntimeError("Refinement invalidly relaxed baseline certificate")
    return CurvatureRefinedCertificate(
        base_parameter_radius=base,
        refined_parameter_radius=R,
        recorded_curvature_floor=mu,
        base_certified_unflipped=classic,
        refined_certified_unflipped=tight,
        base_certification_count=int(classic.sum()),
        refined_certification_count=int(tight.sum()),
    )
