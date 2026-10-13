"""Falsifiable finite-dimensional certificate tests and exact refit control."""
import numpy as np
import pytest

from research.decision_aware_convex_eeg_certificate import (
    augmented_features,
    audit_against_refit,
    certify_known_training_source_change,
    fit_regularized_logistic,
    objective_and_gradient,
)


def _data(seed=19):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(45,3))
    y=np.where(x[:,0] + .4*x[:,1] > 0,1.,-1.)
    q=augmented_features(rng.normal(size=(27,3)))
    return augmented_features(x),y,q


def test_pure_newton_reference_and_finite_differences():
    z,y,_=_data()
    lam=.3
    t=np.array([.2,-.4,.8,.1])
    loss,grad=objective_and_gradient(t,z,y,lam)
    assert np.isfinite(loss)
    for j in range(len(t)):
        delta=np.zeros_like(t)
        delta[j]=1e-6
        fd=(objective_and_gradient(t+delta,z,y,lam)[0]-
            objective_and_gradient(t-delta,z,y,lam)[0])/(2e-6)
        np.testing.assert_allclose(fd,grad[j],rtol=1e-6,atol=1e-8)


def test_actual_perturbed_optimum_is_always_inside_gradient_ball():
    z,y,q=_data()
    lam=.35
    ref=fit_regularized_logistic(z,y,lam)
    for seed in range(6):
        rng=np.random.default_rng(seed)
        disturbed=z.copy()
        disturbed[:,:-1] += rng.normal(
            scale=[0.,.001,.01,.02,.1,.35][seed],
            size=disturbed[:,:-1].shape
        )
        altered=fit_regularized_logistic(disturbed,y,lam)
        cert=certify_known_training_source_change(
            ref.coefficients,z,disturbed,y,q,lam
        )
        actual=np.linalg.norm(ref.coefficients-altered.coefficients)
        assert actual<=cert.max_parameter_shift_bound+(
            altered.optimization_grad_l2/lam)+1e-8
        flip=(q@ref.coefficients>0)!=(q@altered.coefficients>0)
        assert not np.any(cert.certified_unflipped & flip)


@pytest.mark.parametrize("scale",[0.,1e-7,1e-4,.001,.02,.2,.8])
def test_train_perturbation_report_is_truthful_and_never_overclaims(scale):
    z,y,q=_data(21)
    changed=z.copy()
    changed[:,:-1]+=scale*np.random.default_rng(987).standard_normal(
        changed[:,:-1].shape
    )
    x=audit_against_refit(z,changed,y,q,lambda_reg=.4)
    assert x["n_test"]==len(q)
    assert 0<=x["certified_no_flips"]<=len(q)
    assert x["uncertified_or_abstained"] == len(q)-x["certified_no_flips"]
    assert x["actual_flips"]<=x["uncertified_or_abstained"]
    assert x["observed_parameter_shift"]<=(
        x["worst_shift_upper"]+x["refit_gradient_residual"]/.4+1e-8
    )
    assert x["certificate_is_one_sided"]
    assert x["not_original_new_mathematical_theorem"]


def test_no_changed_sources_can_reach_full_coverage_if_margin_separated():
    z,y,q=_data(42)
    original=fit_regularized_logistic(z,y,.5)
    cert=certify_known_training_source_change(
        original.coefficients,z,z.copy(),y,q,.5
    )
    assert cert.certification_fraction>=.8
    assert cert.max_parameter_shift_bound<1e-6


def test_borderline_margin_must_abstain_even_if_both_predictions_equal():
    z,y,_=_data()
    t=fit_regularized_logistic(z,y,.4).coefficients
    # Fixed heldout all-zero row has margin == 0 and norm 0.
    cert=certify_known_training_source_change(
        t,z,z.copy(),y,np.zeros((1,z.shape[1])),.4
    )
    assert cert.certified_unflipped.tolist()==[False]


@pytest.mark.parametrize("bad_lambda",[0.,-1.,float("nan")])
def test_nonpositive_or_nan_regularization_fails_closed(bad_lambda):
    z,y,q=_data()
    with pytest.raises(ValueError):
        fit_regularized_logistic(z,y,bad_lambda)


def test_label_and_feature_breach_fails_closed():
    z,y,q=_data()
    theta=fit_regularized_logistic(z,y,.2).coefficients
    with pytest.raises(ValueError,match="Counterfactual changed"):
        certify_known_training_source_change(
            theta,z,z[:-1],y,q,.2
        )
    with pytest.raises(ValueError):
        certify_known_training_source_change(
            theta,z,z,y,np.ones((10,2)),.2
        )
    invalid=z.copy()
    invalid[0,0]=np.inf
    with pytest.raises(ValueError):
        certify_known_training_source_change(
            theta,z,invalid,y,q,.2
        )


def test_does_not_certify_retraining_with_different_bias_objective():
    z,y,q=_data()
    altered=z.copy()
    altered[:,:-1]*=1.02
    # Completely different objective/regularization cannot pass as matched.
    with pytest.raises(ValueError):
        fit_regularized_logistic(z,y,0)
    assert audit_against_refit(z,altered,y,q,.4)["n_test"]==27
