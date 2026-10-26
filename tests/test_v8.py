"""Behavioral validation of frozen evaluation and prospective experiment planning."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from banking_churn.schema import FEATURES
from banking_churn.synthetic import make_synthetic_data
from banking_churn.v8_validation import (ValidationEvidence,audit_frozen_model,
    build_reference_profile,demo_bundle,read_validation_csv)
from banking_churn.v8_pilot import PilotPlan,approximate_power,simulate_pilot,wilson_interval


class FrozenAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=make_synthetic_data(1000,42)
        cls.sha=hashlib.sha256(cls.reference.to_csv(index=False).encode()).hexdigest()
        cls.bundle=demo_bundle(cls.reference,cls.sha)
        cls.profile=build_reference_profile(cls.reference,cls.bundle,cls.sha)
        cls.incoming=make_synthetic_data(1000,123).loc[:,FEATURES+["CustomerId","Exited"]]
        cls.incoming["CustomerId"]="NEW-"+cls.incoming.CustomerId
        cls.evidence=ValidationEvidence("Unit test fixture","Fictional population","Fictional period",90,True,True,True)

    def audit(self,frame=None,**kwargs):
        arguments=dict(evidence=self.evidence,reference_profile=self.profile,incoming_sha256="new-test-file",resamples=20)
        arguments.update(kwargs)
        return audit_frozen_model(self.incoming if frame is None else frame,self.bundle,**arguments)

    def test_frozen_estimator_has_no_fit_and_policy_or_input_never_changes(self):
        self.assertFalse(hasattr(self.bundle["estimator"],"fit"))
        original=self.incoming.copy(deep=True);metadata=copy.deepcopy(self.bundle["metadata"])
        result=self.audit()
        self.assertEqual(result["summary"]["rows_scored"],1000)
        self.assertFalse(result["summary"]["model_refit"])
        self.assertEqual(metadata,self.bundle["metadata"])
        pd.testing.assert_frame_equal(original,self.incoming)

    def test_subset_of_development_is_detected_despite_changed_file_hash(self):
        frame=self.reference.loc[:,FEATURES+["CustomerId","Exited"]].head(50)
        summary=self.audit(frame)["summary"]
        self.assertEqual(summary["overlap_customer_ids"],50)
        self.assertEqual(summary["overlap_unique_feature_rows"],50)
        self.assertFalse(summary["external_validation_verified"])
        self.assertTrue(any("independensi belum" in g for g in summary["readiness_gaps"]))

    def test_renaming_ids_does_not_hide_identical_feature_rows(self):
        frame=self.reference.loc[:,FEATURES+["CustomerId","Exited"]].head(50).copy()
        frame["CustomerId"]="RENAMED-"+frame.CustomerId
        summary=self.audit(frame)["summary"]
        self.assertEqual(summary["overlap_customer_ids"],0)
        self.assertEqual(summary["overlap_unique_feature_rows"],50)

    def test_same_file_hash_is_reported_without_reference_profile(self):
        summary=self.audit(reference_profile=None,incoming_sha256=self.sha)["summary"]
        self.assertTrue(summary["same_source_file"])
        self.assertFalse(summary["external_validation_verified"])

    def test_no_ids_cannot_complete_independence_review(self):
        summary=self.audit(self.incoming.drop(columns="CustomerId"))["summary"]
        self.assertIsNone(summary["overlap_customer_ids"])
        self.assertTrue(any("Overlap ID" in g for g in summary["readiness_gaps"]))

    def test_csv_leading_zeros_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"incoming.csv";frame=self.incoming.head(2).copy()
            frame["CustomerId"]=["0001","0002"];frame.to_csv(path,index=False)
            self.assertEqual(read_validation_csv(path).CustomerId.tolist(),["0001","0002"])

    def test_blank_or_canonical_duplicate_ids_block_all_scoring(self):
        for identifiers in [["a"," a "],["a"," "],["a",None]]:
            frame=self.incoming.head(2).copy();frame["CustomerId"]=identifiers
            result=self.audit(frame)
            self.assertEqual(result["summary"]["status"],"blocked_invalid_schema")
            self.assertTrue(result["scores"].empty)

    def test_invalid_schema_or_labels_never_export_partial_scores(self):
        for frame in [self.incoming.drop(columns="Age"),self.incoming.assign(typo=1),
            self.incoming.assign(Exited=2),self.incoming.assign(Age=np.inf)]:
            result=self.audit(frame)
            self.assertEqual(result["summary"]["rows_scored"],0)
            self.assertTrue(result["metrics"].empty)

    def test_missing_or_immature_labels_delay_predictive_metrics(self):
        immature=ValidationEvidence("fixture","population","period",90,True,False,True)
        for frame,evidence in [(self.incoming.drop(columns="Exited"),self.evidence),(self.incoming,immature)]:
            result=self.audit(frame,evidence=evidence)
            self.assertFalse(result["summary"]["labels_evaluated"])
            self.assertTrue(result["metrics"].empty)
            self.assertEqual(len(result["scores"]),len(frame))

    def test_single_class_auc_ap_and_zero_denominator_recall_are_undefined(self):
        result=self.audit(self.incoming.head(50).assign(Exited=0))
        metrics=result["metrics"].set_index("metric")
        self.assertTrue(pd.isna(metrics.loc["roc_auc","value"]))
        self.assertTrue(pd.isna(metrics.loc["average_precision","value"]))
        self.assertTrue(pd.isna(metrics.loc["recall","value"]))
        self.assertEqual(metrics.loc["recall","denominator"],0)
        self.assertEqual(sum(result["summary"]["confusion_counts"].values()),50)

    def test_synthetic_bundle_cannot_be_presented_as_verified_external_validation(self):
        result=self.audit(synthetic=False)
        self.assertEqual(result["summary"]["status"],"synthetic_demonstration")
        self.assertFalse(result["summary"]["external_validation_verified"])

    def test_fully_declared_no_overlap_batch_still_only_qualifies_for_human_review(self):
        bundle=copy.deepcopy(self.bundle);bundle["metadata"]["synthetic"]=False
        for bounds in bundle["metadata"]["training_numeric_ranges"].values():
            bounds.update(minimum=0.,maximum=1e9)
        result=audit_frozen_model(self.incoming,bundle,self.evidence,self.profile,"new-test-file",resamples=20)
        self.assertEqual(result["summary"]["status"],"ready_for_independent_review")
        self.assertFalse(result["summary"]["external_validation_verified"])
        self.assertTrue(result["summary"]["evidence_is_user_declaration"])

    def test_profile_from_another_model_or_source_is_rejected(self):
        with self.assertRaises(ValueError):build_reference_profile(self.reference,self.bundle,"wrong")
        altered={**self.profile,"model_run_id":"another-model"}
        with self.assertRaises(ValueError):self.audit(reference_profile=altered)

    def test_bootstrap_is_deterministic_and_intervals_are_bounded(self):
        left=self.audit();right=self.audit()
        pd.testing.assert_frame_equal(left["metrics"],right["metrics"])
        metrics=left["metrics"].set_index("metric")
        for name in ["precision","recall","roc_auc","average_precision","brier_score"]:
            self.assertTrue(0<=metrics.loc[name,"low_95"]<=metrics.loc[name,"high_95"]<=1)


class PilotTests(unittest.TestCase):
    def test_wilson_boundaries_and_published_nist_example(self):
        # NIST's 26/200 one-sided 95% lower bound equals a two-sided 90% lower bound.
        low,_=wilson_interval(26,200,.90)
        self.assertAlmostEqual(low,.09577,places=5)
        self.assertEqual(wilson_interval(0,0),(None,None))
        self.assertEqual(wilson_interval(0,10)[0],0)
        self.assertEqual(wilson_interval(10,10)[1],1)

    def test_sample_plan_rounding_meets_power_and_smaller_effect_needs_more_people(self):
        plan=PilotPlan().calculate();n=plan["analyzable_per_arm"]
        self.assertGreaterEqual(approximate_power(.2,.17,n),.8)
        self.assertLess(approximate_power(.2,.17,n-1),.8)
        self.assertGreaterEqual(plan["enrolment_per_arm"]*.9,n)
        self.assertGreater(PilotPlan(absolute_reduction=.01).calculate()["analyzable_per_arm"],n)
        self.assertFalse(plan["effect_estimated_from_churn_model"])

    def test_monte_carlo_confirms_target_power_and_type_one_error(self):
        plan=PilotPlan()
        alternative,_=simulate_pilot(plan,repetitions=20000,seed=92)
        null,_=simulate_pilot(plan,repetitions=20000,seed=93,null_effect=True)
        self.assertAlmostEqual(alternative["rejection_rate"],.8,delta=.025)
        self.assertAlmostEqual(null["rejection_rate"],.05,delta=.01)
        self.assertLess(null["benefit_direction_detection_rate"],null["rejection_rate"])

    def test_larger_sample_improves_detection_and_seed_repeats_exactly(self):
        plan=PilotPlan()
        a,draws=simulate_pilot(plan,per_arm=200,repetitions=5000,seed=42)
        b,same=simulate_pilot(plan,per_arm=200,repetitions=5000,seed=42)
        large,_=simulate_pilot(plan,per_arm=4000,repetitions=5000,seed=42)
        pd.testing.assert_frame_equal(draws,same)
        self.assertEqual(a,b)
        self.assertGreater(large["rejection_rate"],a["rejection_rate"])

    def test_invalid_or_impossible_plans_fail_clearly(self):
        for plan in [PilotPlan(control_rate=0),PilotPlan(absolute_reduction=0),PilotPlan(absolute_reduction=.2),
            PilotPlan(power=True),PilotPlan(missing_outcome_fraction=np.nan),PilotPlan(absolute_reduction=1e-12)]:
            with self.assertRaises(ValueError):plan.calculate()
        for kwargs in [{"per_arm":True},{"repetitions":10},{"seed":-1}]:
            with self.assertRaises(ValueError):simulate_pilot(PilotPlan(),**kwargs)


if __name__=="__main__":unittest.main()
