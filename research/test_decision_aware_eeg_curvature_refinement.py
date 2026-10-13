"""Exact-refit and near-margin challenge for Hessian-refined EEG certificates."""
import numpy as np
import pytest

from research.decision_aware_convex_eeg_certificate import (
    augmented_features, fit_regularized_logistic
)
from research.decision_aware_eeg_curvature_refinement import (
    certified_curvature_radius,compare_curvature_certificates
)


def _problem(seed=0):
    rng=np.random.default_rng(seed)
    z=augmented_features(rng.normal(size=(55,4)))
    label=np.where(z[:,0]+.3*z[:,1]+.1*z[:,2]>0,1.,-1.)
    delta=z.copy()
    delta[:,:-1]+=rng.normal(scale=.08,size=delta[:,:-1].shape)
    return z,delta,label


@pytest.mark.parametrize("seed",list(range(12)))
def test_refined_ball_contains_actual_optimum_on_strongly_convex_problem(seed):
    z,d,y=_problem(seed)
    lam=.35
    clean=fit_regularized_logistic(z,y,lam)
    changed=fit_regularized_logistic(d,y,lam)
    base,tight,curvature=certified_curvature_radius(
        clean.coefficients,d,y,lam
    )
    observed=np.linalg.norm(changed.coefficients-clean.coefficients)
    assert 0<=tight<=base
    assert curvature>=lam
    assert observed<=tight+changed.optimization_grad_l2/lam+1e-8
    assert tight<base # For fixed full-rank real-valued features, curvature > 0


def test_stronger_curvature_certificate_can_be_strictly_more_informative():
    z,d,y=_problem(17)
    lam=.4
    model=fit_regularized_logistic(z,y,lam)
    altered=fit_regularized_logistic(d,y,lam)
    base,tight,curv=certified_curvature_radius(model.coefficients,d,y,lam)
    w=model.coefficients
    mag=np.linalg.norm(w)
    assert 0<tight<base<mag
    # Construct a fixed candidate at a logit margin BETWEEN two proved
    # radii; this is an adversarial synthetic numerical witness, not real EEG.
    target=(tight+base)/2
    v=w/mag
    orth=np.zeros_like(v)
    orth[0]=1
    orth-=v*np.dot(orth,v)
    orth/=np.linalg.norm(orth)
    q=target/mag*v+np.sqrt(1-(target/mag)**2)*orth
    result=compare_curvature_certificates(
        w,d,y,q.reshape(1,-1),lam
    )
    assert result.base_certification_count==0
    assert result.refined_certification_count==1
    assert (q@model.coefficients>0)==(q@altered.coefficients>0)
    assert not result.numerical_float_interval_verified


@pytest.mark.parametrize("seed",list(range(8)))
def test_every_certified_change_survives_exact_numeric_refit(seed):
    z,d,y=_problem(100+seed)
    lam=.25
    model=fit_regularized_logistic(z,y,lam)
    altered=fit_regularized_logistic(d,y,lam)
    rng=np.random.default_rng(seed)
    q=augmented_features(rng.normal(size=(200,4)))
    s=compare_curvature_certificates(model.coefficients,d,y,q,lam)
    assert s.refined_certification_count>=s.base_certification_count
    observed_flip=(q@model.coefficients>0)!=(q@altered.coefficients>0)
    assert not np.any(observed_flip&s.refined_certified_unflipped)


@pytest.mark.parametrize("lam",[0.,-0.5,np.nan])
def test_strong_convexity_is_mandatory(lam):
    z,d,y=_problem()
    theta=fit_regularized_logistic(z,y,.3).coefficients
    with pytest.raises(ValueError):
        certified_curvature_radius(theta,d,y,lam)


def test_unsafe_nonfinite_observation_or_weight_refuses_a_certificate():
    z,d,y=_problem()
    theta=fit_regularized_logistic(z,y,.3).coefficients
    with pytest.raises(ValueError):
        compare_curvature_certificates(theta,d,y,
                                      np.array([[np.nan]*len(theta)]),.3)
    with pytest.raises(ValueError):
        certified_curvature_radius(np.full_like(theta,np.inf),d,y,.3)


def test_bound_at_zero_perturbation_remains_nonnegative_and_exact_reference():
    z,_,y=_problem()
    theta=fit_regularized_logistic(z,y,.3).coefficients
    base,tight,mu=certified_curvature_radius(theta,z,y,.3)
    assert base<1e-6
    assert tight<=base
    assert mu>=.3
