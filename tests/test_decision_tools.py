"""Meaningful numerical, data-quality and report-integrity tests for embedded V6 code."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
threadpool_limits(2)
NB = json.loads((ROOT/"notebooks/banking_churn_v6.ipynb").read_text(encoding="utf-8"))
CODE = next("".join(c["source"]) for c in NB["cells"] if c["id"]=="v6_helpers")
env = {"__name__":"embedded_v6_tests"}
exec(compile(CODE,"notebook:v6_helpers","exec"),env)


class DecisionTests(unittest.TestCase):
    def advisor(self, **overrides):
        args = dict(labels=[1,1,0,0],probabilities=[.9,.8,.2,.1],locked_policies={},
            max_contacts=4,budget=45,contact_cost=10,value_per_saved=100,success_rate=.2,fixed_cost=5)
        args.update(overrides)
        return env["v6_policy_advisor"](**args)

    def test_known_economic_optimum_uses_two_contacts(self):
        result = self.advisor()
        self.assertEqual(result["recommended"]["contacts"],2)
        self.assertAlmostEqual(result["recommended"]["expected_net_um"],9)

    def test_budget_and_fixed_cost_make_launch_impossible(self):
        result = self.advisor(budget=14)
        self.assertEqual(result["status"],"no_positive_feasible_campaign")
        self.assertEqual(result["recommended"]["cost_um"],0)

    def test_losing_and_break_even_campaigns_are_not_recommended(self):
        for u in [0.,.1]:
            result = self.advisor(success_rate=u,fixed_cost=0)
            self.assertEqual(result["recommended"]["contacts"],0)

    def test_zero_capacity_and_budget_do_not_charge_fixed_cost(self):
        r = self.advisor(max_contacts=0,budget=0)
        self.assertEqual(r["recommended"]["expected_net_um"],0)

    def test_minimum_validation_precision_can_block_all_campaigns(self):
        result = self.advisor(labels=[0,0,1,1],min_precision=.8)
        self.assertEqual(result["status"],"no_positive_feasible_campaign")

    def test_zero_contact_cost_still_obeys_fixed_cost_budget(self):
        result = self.advisor(contact_cost=0,budget=4)
        self.assertEqual(result["recommended"]["contacts"],0)

    def test_locked_threshold_can_be_ineligible_without_being_silently_truncated(self):
        result = self.advisor(locked_policies={"all":{"kind":"threshold","threshold":0}},max_contacts=2)
        row = result["candidates"].set_index("policy").loc["all"]
        self.assertEqual(row.contacts,4)
        self.assertFalse(row.resource_feasible)

    def test_ties_are_resolved_by_unique_text_key(self):
        p,order = env["v6_order"]([.5,.5,.5],["z","a","b"])
        self.assertEqual(order.tolist(),[1,2,0])
        with self.assertRaises(ValueError): env["v6_order"]([.5,.5],[1,"1"])

    def test_inputs_are_not_mutated(self):
        policies={"x":{"kind":"threshold","threshold":.4}}
        original=copy.deepcopy(policies)
        self.advisor(locked_policies=policies)
        self.assertEqual(original,policies)

    def test_invalid_numeric_inputs_are_rejected(self):
        for override in [{"budget":np.nan},{"budget":-1},{"max_contacts":1.5},{"max_contacts":5},
                         {"contact_cost":np.inf},{"success_rate":1.1},{"min_precision":-1},{"max_contacts":True}]:
            with self.subTest(override=override),self.assertRaises(ValueError): self.advisor(**override)


class UncertaintyTests(unittest.TestCase):
    def sample(self, **overrides):
        args=dict(probabilities=[.9,.8,.2,.1],contacts=2,success_range=[.2,.2],cost_range=[10,10],
            benefit_range=[100,100],fixed_cost=5,n_draws=1000,seed=2)
        args.update(overrides)
        return env["v6_uncertain_campaign"](**args)

    def test_fixed_parameters_match_known_expected_net_and_variance(self):
        result=self.sample()
        self.assertAlmostEqual(result["summary"]["mean_assumed_expected_net_um"],9)
        self.assertAlmostEqual(result["summary"]["parameter_expected_net_sd_um"],0)
        expected_variance=100**2*(.18*.82+.16*.84)
        self.assertAlmostEqual(result["summary"]["conditional_outcome_rms_sd_um"]**2,expected_variance)

    def test_zero_contacts_never_charge_costs(self):
        result=self.sample(contacts=0,fixed_cost=999)
        self.assertTrue((result["draws"].outcome_net_um==0).all())
        self.assertEqual(result["summary"]["profit_fraction_under_assumptions"],0)

    def test_seed_replays_identical_parameter_and_outcome_draws(self):
        a=self.sample(success_range=[.1,.3],cost_range=[8,12],benefit_range=[80,120])
        b=self.sample(success_range=[.1,.3],cost_range=[8,12],benefit_range=[80,120])
        pd.testing.assert_frame_equal(a["draws"],b["draws"])

    def test_parameter_variation_is_separate_from_outcome_variation(self):
        result=self.sample(success_range=[.1,.3],cost_range=[8,12])
        s=result["summary"]
        self.assertGreater(s["parameter_expected_net_sd_um"],0)
        self.assertGreater(s["combined_sd_from_variance_decomposition_um"],s["parameter_expected_net_sd_um"])
        self.assertFalse(result["assumptions"]["model_calibration_and_population_shift_uncertainty_included"])

    def test_all_scenarios_have_valid_assumed_success_and_finite_net(self):
        result=self.sample(success_range=[0,1],cost_range=[0,100],benefit_range=[0,200])
        self.assertTrue(result["draws"].success_assumption.between(0,1).all())
        self.assertTrue(np.isfinite(result["draws"].to_numpy()).all())

    def test_reversed_or_unbounded_ranges_and_work_limits_rejected(self):
        for override in [{"success_range":[.3,.1]},{"cost_range":[1,np.inf]},
                         {"benefit_range":[-1,2]},{"n_draws":99},{"n_draws":10001},
                         {"seed":-1},{"contacts":2.5}]:
            with self.subTest(override=override),self.assertRaises(ValueError): self.sample(**override)


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from synthetic_fixtures import make_synthetic_bundle
        cls.bundle,cls.reference,cls.helper=make_synthetic_bundle(NB)
        cls.ref_p=cls.helper.predict_customers(cls.reference,cls.bundle).estimated_churn_probability.to_numpy()

    def quality(self,frame,**kwargs):
        return env["v6_batch_quality"](self.reference,frame,self.bundle,self.helper.predict_customers,self.ref_p,**kwargs)

    def test_identical_batch_has_zero_distribution_distance(self):
        result=self.quality(self.reference.copy())
        self.assertEqual(result["summary"]["status"],"no_alert_at_selected_threshold")
        self.assertEqual(result["features"].total_variation_distance.sum(),0)
        self.assertAlmostEqual(result["summary"]["score_distribution_tv"],0)

    def test_unknown_category_and_missing_are_reported_while_scoring(self):
        frame=self.reference.copy()
        frame.loc[frame.index[:20],"Geography"]="NEW"
        frame.loc[frame.index[:30],"Balance"]=np.nan
        result=self.quality(frame)
        self.assertEqual(result["summary"]["scored_rows"],200)
        self.assertTrue({"Balance","Geography"}<=set(result["summary"]["flagged_features"]))

    def test_whitespace_is_normalized_before_quality_comparison(self):
        frame=self.reference.copy()
        frame["Geography"]=frame.Geography.map(lambda x:" "+x+" ")
        result=self.quality(frame)
        self.assertEqual(result["features"].set_index("feature").loc["Geography","unseen_category_rate"],0)

    def test_binary_proportion_shift_is_not_hidden_by_quantile_bins(self):
        frame=self.reference.copy()
        frame["HasCrCard"]=1-self.reference.HasCrCard
        result=self.quality(frame)
        distance=result["features"].set_index("feature").loc["HasCrCard","total_variation_distance"]
        self.assertAlmostEqual(distance,abs(frame.HasCrCard.mean()-self.reference.HasCrCard.mean()))
        self.assertGreater(distance,0)

    def test_invalid_schema_does_not_leak_partial_scores(self):
        frame=self.reference.copy()
        frame.loc[0,"HasCrCard"]=9
        result=self.quality(frame)
        self.assertEqual(result["summary"]["status"],"blocked_invalid_schema")
        self.assertTrue(result["scores"].empty)

    def test_missing_required_or_extra_or_duplicate_headers_block_scoring(self):
        for frame in [self.reference.drop(columns="Age"),self.reference.assign(typo=1),
                      pd.concat([self.reference,self.reference[["Age"]]],axis=1)]:
            with self.subTest(columns=list(frame.columns)):
                self.assertEqual(self.quality(frame)["summary"]["scored_rows"],0)

    def test_customer_ids_are_preserved_and_source_frame_not_mutated(self):
        frame=self.reference.head(4).copy()
        frame["CustomerId"]=["0001","0002","0003","0004"]
        old=frame.copy(deep=True)
        result=self.quality(frame)
        self.assertEqual(result["scores"].index.tolist(),frame.CustomerId.tolist())
        pd.testing.assert_frame_equal(frame,old)

    def test_optional_labels_produce_audit_without_model_refit(self):
        frame=self.reference.copy()
        frame["Exited"]=np.resize([0,1],len(frame))
        result=self.quality(frame)
        self.assertIn("label_audit",result["summary"])
        self.assertFalse(result["summary"]["label_audit"]["labels_used_to_refit_or_select_policy"])

    def test_total_variation_extremes_and_missing_token_collision(self):
        self.assertEqual(env["v6_total_variation"]([1,0],[0,1]),1)
        counts=env["v6_distribution_counts"](pd.Series([None,"__MISSING__"]))
        self.assertEqual(len(counts),2)


class ReportTests(unittest.TestCase):
    def context(self):
        return dict(run_id="current-run",data_sha256="csv-hash",model="Example",
            test_scores=dict(roc_auc=.8,average_precision=.6,test_rows=10,test_churn=3,test_prevalence=.3),
            default_metrics=dict(contacts=4,contact_rate=.4,tp=2,fp=2,fn=1,precision=.5,recall=2/3),
            default_policy="balanced_f1",threshold=.33)

    def test_report_numbers_follow_changed_context(self):
        context=self.context()
        text=env["v6_test_narrative"](context)
        context["model"]="Changed"
        context["threshold"]=.77
        other=env["v6_test_narrative"](context)
        self.assertNotEqual(text,other)
        self.assertIn("Changed",other)
        self.assertIn("0.770000",other)

    def test_same_run_snapshot_refreshes_report_and_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/"bank_churn_artifacts_v6"
            directory=base/"runs/current-run"
            directory.mkdir(parents=True)
            (base/"latest_run.json").write_text(json.dumps({"run_id":"current-run","status":"complete"}),encoding="utf-8")
            a=dict(status="no_positive_feasible_campaign",recommended={"contacts":0,"cost_um":0,"expected_net_um":0,"policy":"no_campaign"})
            env["v6_save_section"]("advisor",{},a,directory,self.context())
            report=(directory/"current_report.md").read_text(encoding="utf-8")
            self.assertIn("Tidak ada kampanye",report)
            with zipfile.ZipFile(base/"current-run.zip") as z:
                self.assertIsNone(z.testzip())
                self.assertIn("current_report.md",z.namelist())

    def test_cross_run_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            (directory/"v6_advisor_latest.json").write_text(json.dumps({"run_id":"other","source_sha256":"csv-hash"}),encoding="utf-8")
            with self.assertRaises(ValueError): env["v6_refresh_report"](directory,self.context())

    def test_nonfinite_json_values_are_explicitly_null(self):
        clean=env["v6_json"]({"x":np.nan,"y":np.inf})
        self.assertEqual(clean,{"x":None,"y":None})
        json.dumps(clean,allow_nan=False)

    def test_protocol_mismatch_suppresses_comparison_delta(self):
        old=dict(run_id="v5",data_sha256="old",metrics={"AP":.5})
        new=dict(run_id="v6",data_sha256="new",metrics={"AP":.6})
        result=env["v6_compare_versions"](old,new)
        self.assertIn("data_sha256",result["summary"]["protocol_mismatches"])
        self.assertTrue(pd.isna(result["table"].delta_v6_minus_v5.iloc[0]))


if __name__=="__main__":
    unittest.main(verbosity=2)
