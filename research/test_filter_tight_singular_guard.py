"""Tight spectral-vs-Frobenius fold guard reproducibility and adversarial tests."""
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from research.filter_tight_singular_guard import (
    compare_tight_filter_guard, worst_direction_counterfactual,
)


def _filter():
    return butter(4,[1,30],btype="bandpass",fs=250,output="sos")


def test_stronger_induced_gain_preserves_160_more_raw_training_samples():
    tight,loose,rows=compare_tight_filter_guard(
        _filter(),record_samples=4096,test_interval=(1800,2120),
        training_epoch_samples=320,
        candidate_guards=(0,80,320,640,800),
        relative_l2_budget=0.0006,
    )
    assert tight is not None and loose is not None
    assert tight.guard_samples==640
    assert loose.guard_samples==800
    assert loose.guard_samples-tight.guard_samples==160
    assert len(rows)==5
    assert all(row.spectral_gain_float64<=row.frobenius_gain+1e-12
               for row in rows)
    assert all(row.conservative_numerical_margin>=row.spectral_gain_float64
               for row in rows)
    assert not any(row.mathematically_verified_float_rounding for row in rows)


def test_top_singular_direction_reproduces_worst_observation_in_live_scipy():
    sos=_filter()
    probe=worst_direction_counterfactual(
        sos,record_samples=700,train_window=(200,225),
        test_window=(300,325)
    )
    assert probe["declared_perturbation_l2"]==pytest.approx(1.0,abs=1e-10)
    assert probe["observed_train_change_l2"]==pytest.approx(
        probe["computed_induced_gain"],rel=1e-10,abs=1e-12
    )
    assert probe["operator_predicted_train_change_l2"]==pytest.approx(
        probe["computed_induced_gain"],rel=1e-10,abs=1e-12
    )
    assert probe["no_clinical_eeg_or_classifier_accuracy_claim"]


@pytest.mark.parametrize("budget", [0.0,1e-10])
def test_no_false_positive_when_guard_register_inadequate(budget):
    x,y,rows=compare_tight_filter_guard(
        _filter(),record_samples=1400,test_interval=(550,570),
        training_epoch_samples=30,candidate_guards=(0,20,40),
        relative_l2_budget=budget,
    )
    assert x is None and y is None
    assert all(not r.spectral_budget_pass and not r.frobenius_budget_pass
               for r in rows)


@pytest.mark.parametrize("guards,budget",[
    ((0,0),.001),((-1,1),.001),((2,0),.001),
    ((1.5,2),.001),((0,),float("nan")),((0,),-1)
])
def test_invalid_configuration_fails_closed(guards,budget):
    with pytest.raises(ValueError):
        compare_tight_filter_guard(
            _filter(),record_samples=1000,test_interval=(500,540),
            training_epoch_samples=30,candidate_guards=guards,
            relative_l2_budget=budget,
        )
