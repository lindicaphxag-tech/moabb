"""Adversarial subject independence and numerical checks, not EEG evidence."""
import unittest

import numpy as np

from research.transfer_consistency_shrinkage import (
    _source_nested_select, _subject_equal_spectral_state,
    nested_subject_holdout, paired_subject_summary, transfer_scores,
)


def synthetic_cohort(seed=1928, subjects=5, epochs=26, features=8):
    rng=np.random.default_rng(seed)
    X=[]; y=[]; group=[]
    for subject in range(subjects):
        labels=np.r_[np.zeros(epochs//2),np.ones(epochs//2)].astype(int)
        rng.shuffle(labels)
        a=rng.normal(size=(epochs,features))
        a[:,0]+=(labels-.5)*.9
        a[:,1]+=(labels-.5)*(.4 if subject%2==0 else -.4)
        a[:,2]+=.2*subject
        X.append(a); y.append(labels);group.extend([subject]*epochs)
    return np.vstack(X),np.concatenate(y),np.asarray(group)


class TransferConsistencyContract(unittest.TestCase):
    def test_one_inconsistent_direction_shrinks_more(self):
        X,y,g=synthetic_cohort()
        model=_subject_equal_spectral_state(X,y,g)
        self.assertEqual(len(model.coherence),X.shape[1])
        self.assertTrue(np.isfinite(model.coherence).all())
        self.assertGreaterEqual(model.coherence.min(),0)
        self.assertLessEqual(model.coherence.max(),1)
        for extra in (0.,1.,8.):
            p=transfer_scores(model,X[:7],.3,extra)
            self.assertTrue(np.isfinite(p).all())

    def test_outer_test_label_permutation_cannot_change_selected_hyperparameters(self):
        X,y,g=synthetic_cohort()
        v1=nested_subject_holdout(X,y,g,outer_subjects=[4])
        y2=y.copy()
        y2[g==4]=1-y2[g==4]
        v2=nested_subject_holdout(X,y2,g,outer_subjects=[4])
        a,b=v1[0],v2[0]
        for key in ("source_only_selected_base","source_only_selected_transfer_penalty",
                    "source_only_inner_AUC","training_subjects"):
            self.assertEqual(a[key],b[key],key)
        self.assertNotEqual(a["test_auc_LDA_LedoitWolf_auto"],b["test_auc_LDA_LedoitWolf_auto"])

    def test_outer_train_subject_registry_has_no_target_id(self):
        X,y,g=synthetic_cohort()
        results=nested_subject_holdout(X,y,g,outer_subjects=[1,4])
        self.assertEqual([r["heldout_subject"] for r in results],["1","4"])
        self.assertTrue(all(r["heldout_subject"] not in r["training_subjects"] for r in results))
        self.assertTrue(all(r["n_train_subjects"]==4 for r in results))

    def test_source_subj_not_epochs_are_tuning_splits(self):
        X,y,g=synthetic_cohort()
        params,trace=_source_nested_select(X,y,g,base_grid=(.3,3.),
                                           penalty_grid=(0.,1.))
        self.assertEqual(len(trace),4)
        self.assertTrue(all(0<=v<=1 for v in trace.values()))
        self.assertIn(params[0],(.3,3.))
        self.assertIn(params[1],(0.,1.))

    def test_reject_single_class_subject(self):
        X,y,g=synthetic_cohort()
        y[g==2]=1
        with self.assertRaisesRegex(ValueError,"both classes"):
            _subject_equal_spectral_state(X,y,g)

    def test_reject_insufficient_source_groups(self):
        X,y,g=synthetic_cohort(subjects=3)
        with self.assertRaisesRegex(ValueError,"at least four"):
            nested_subject_holdout(X,y,g,outer_subjects=[0])

    def test_same_subject_all_epochs_one_vote_for_mean_discriminant(self):
        X,y,g=synthetic_cohort()
        state=_subject_equal_spectral_state(X,y,g)
        # One group's repeated, identical observations should not increase
        # that group's transferability coefficient via sample-count weighting.
        idx=g==0
        X2=np.concatenate([X,np.tile(X[idx],(3,1))])
        y2=np.concatenate([y,np.tile(y[idx],3)])
        g2=np.concatenate([g,np.tile(g[idx],3)])
        other=_subject_equal_spectral_state(X2,y2,g2)
        np.testing.assert_allclose(state.mean_diff,other.mean_diff,atol=1e-8)
        np.testing.assert_allclose(state.coherence,other.coherence,atol=1e-6)

    def test_summary_uses_unique_heldout_subjects_not_epochs(self):
        X,y,g=synthetic_cohort()
        res=nested_subject_holdout(X,y,g,outer_subjects=[0,1])
        report=paired_subject_summary(res)
        self.assertEqual(report["n_independent_outer_subjects"],2)
        self.assertIn("test_auc_LDA_LedoitWolf_auto",report["mean_auc"])
        with self.assertRaises(ValueError):
            paired_subject_summary(res+[res[0]])

if __name__=="__main__":
    unittest.main()
