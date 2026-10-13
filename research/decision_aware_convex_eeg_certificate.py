"""A rigorous but *conservative* decision-stability certificate for EEG folds.

This is standard strongly convex ERM sensitivity mathematics (not a novel
general theorem). It links measured preprocessing-originated changes in
training features to a bound on a regularized logistic decoder's held-out
decision. The exact model is explicitly distinct from sklearn's earlier
unregularized-intercept experimental pipeline.

For F_delta(theta) = mean log(1+exp(-y_i z_i^T theta))
                     + (lambda/2)*||theta||^2, lambda>0,
the whole objective is lambda-strongly convex. For *any* candidate theta0
and perturbed optimum theta_delta*, strong monotonicity implies

||theta_delta* - theta0||_2 <= ||grad F_delta(theta0)||_2 / lambda.

Therefore a fixed heldout z is CERTIFIED unflipped when
|theta0^T z| > ||z|| * ||grad F_delta(theta0)|| / lambda.

All coordinate preprocessing/normalization must be fixed before the raw
source counterfactual. Bias is appended as 1 and deliberately penalized.
No labels/heldout features are ever used to fit theta; heldout z is only
used to evaluate the final margin. This certificate covers the *specific
counterfactual matrix*, not all possible sources or hidden contamination.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class TrainedLogistic:
    coefficients: np.ndarray
    lambda_reg: float
    optimization_grad_l2: float


@dataclass(frozen=True)
class CertificateResult:
    certified_unflipped: np.ndarray
    original_class_1: np.ndarray
    margin_absolute: np.ndarray
    worst_logit_change_bound: np.ndarray
    max_parameter_shift_bound: float
    perturbed_objective_gradient_l2: float
    certification_fraction: float


def _validated_training(z, y, lambda_reg):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if z.ndim != 2 or not z.shape[0] or not z.shape[1]:
        raise ValueError("Training data must have nonempty rows and features")
    if y.shape != (z.shape[0],) or not set(np.unique(y)).issubset({-1., 1.}):
        raise ValueError("Binary labels must be -1 / +1, training-only")
    if not set(np.unique(y)) == {-1., 1.}:
        raise ValueError("Both original training classes must be represented")
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite source features")
    if not np.isfinite(lambda_reg) or lambda_reg <= 0:
        raise ValueError("Strict positive global regularization required")
    return z, y


def augmented_features(x):
    """Append an explicitly *regularized* bias term as last feature."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected finite 2D features, width >= 1")
    return np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))


def objective_and_gradient(theta, z, y, lambda_reg):
    z, y = _validated_training(z, y, lambda_reg)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (z.shape[1],) or not np.isfinite(theta).all():
        raise ValueError("Invalid linear-model weights")
    signed_logits = y * (z @ theta)
    loss = float(
        np.logaddexp(0., -signed_logits).mean()
        + 0.5 * lambda_reg * (theta @ theta)
    )
    grad = z.T @ (-y * expit(-signed_logits)) / len(y) + lambda_reg * theta
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise RuntimeError("Numerical logistic objective failure")
    return loss, grad


def fit_regularized_logistic(z, y, lambda_reg=0.1):
    """Use a deterministic convergent convex optimizer, fail if inaccurate."""
    z, y = _validated_training(z, y, lambda_reg)
    def f(t):
        return objective_and_gradient(t, z, y, lambda_reg)
    result = minimize(
        f, np.zeros(z.shape[1], dtype=np.float64), jac=True,
        method="BFGS",
        options={"gtol": 1e-11, "maxiter": 2000},
    )
    grad = f(result.x)[1]
    residual = float(np.linalg.norm(grad))
    # BFGS may report precision loss at a mathematically converged solution.
    # Verify numerical stationarity rather than trusting a status string.
    if not np.isfinite(residual) or residual > 1e-7:
        raise RuntimeError(f"Nonstationary logistic fit: grad={residual}")
    return TrainedLogistic(result.x.copy(),float(lambda_reg),residual)


def certify_known_training_source_change(
    theta0, z_original_training, z_changed_training,
    original_training_labels, heldout_clean_features, lambda_reg
):
    """Certify against exact known counterfactual, not hypothetical all attacks.

    theta0 can be any finite fixed reference parameter vector; the bound
    applies to the *exact mathematical optimum* of the altered regularized
    empirical risk. The heldout measurements are never used to fit a model.
    """
    z0, y = _validated_training(
        z_original_training, original_training_labels, lambda_reg
    )
    zd, yd = _validated_training(
        z_changed_training, original_training_labels, lambda_reg
    )
    if zd.shape != z0.shape or not np.array_equal(yd,y):
        raise ValueError("Counterfactual changed original labels or fold size")
    q = np.asarray(heldout_clean_features, dtype=np.float64)
    theta = np.asarray(theta0, dtype=np.float64)
    if (q.ndim != 2 or q.shape[1] != z0.shape[1]
            or theta.shape != (z0.shape[1],)
            or not np.isfinite(q).all() or not np.isfinite(theta).all()):
        raise ValueError("Fixed clean heldout input/weights invalid")
    _, g = objective_and_gradient(theta, zd, y, lambda_reg)
    r = float(np.linalg.norm(g))
    displacement = r/lambda_reg
    margins = np.abs(q @ theta)
    change_bounds = np.linalg.norm(q,axis=1)*displacement
    if not np.isfinite(change_bounds).all():
        raise RuntimeError("Nonfinite certificate: cannot authorize")
    # Strict > is required for a sign-flip certificate, equality abstains.
    accepts = margins > change_bounds + 1e-14
    return CertificateResult(
        certified_unflipped=accepts,
        original_class_1=(q @ theta)>0,
        margin_absolute=margins,
        worst_logit_change_bound=change_bounds,
        max_parameter_shift_bound=displacement,
        perturbed_objective_gradient_l2=r,
        certification_fraction=float(np.mean(accepts)) if len(accepts) else 0.,
    )


def audit_against_refit(z0, zd, y, q, lambda_reg=0.1):
    """Report soundness AND nonvacuity against genuine training refit."""
    original=fit_regularized_logistic(z0,y,lambda_reg)
    perturbed=fit_regularized_logistic(zd,y,lambda_reg)
    cert=certify_known_training_source_change(
        original.coefficients,z0,zd,y,q,lambda_reg
    )
    actual_shift=float(np.linalg.norm(
        perturbed.coefficients-original.coefficients
    ))
    # Numerical optimizer residuals give an extra + ||grad(refit)||/lambda.
    # Exact-optimum certificate is checked conservatively against imperfect
    # numeric refit with that second-order slack.
    extra=perturbed.optimization_grad_l2/lambda_reg
    if actual_shift > cert.max_parameter_shift_bound+extra+1e-8:
        raise RuntimeError("Strong convexity certificate violated")
    observed_original=(q@original.coefficients)>0
    observed_changed=(q@perturbed.coefficients)>0
    violating=np.any(cert.certified_unflipped &
                     (observed_original!=observed_changed))
    if violating:
        raise RuntimeError("A certified prediction empirically flipped")
    return {
        "n_test":int(q.shape[0]),
        "certified_no_flips":int(np.sum(cert.certified_unflipped)),
        "certificate_coverage":cert.certification_fraction,
        "actual_flips":int(np.sum(observed_original!=observed_changed)),
        "uncertified_or_abstained":int(np.sum(~cert.certified_unflipped)),
        "worst_shift_upper":cert.max_parameter_shift_bound,
        "observed_parameter_shift":actual_shift,
        "refit_gradient_residual":perturbed.optimization_grad_l2,
        "certificate_is_one_sided":True,
        "prior_art_strong_convexity_used":True,
        "not_original_new_mathematical_theorem":True,
    }
