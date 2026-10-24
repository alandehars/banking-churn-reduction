"""End-to-end synthetic tests exercise the public training and report path."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader

from banking_churn.config import TrainingConfig
from banking_churn.decision_tools import v6_policy_advisor
from banking_churn.inference import normalize_features,predict_customers,score_customer_csv
from banking_churn.live import save_live_section
from banking_churn.pipeline import run_pipeline,run_training,finish_run
from banking_churn.reporting import aggregate_summary
from banking_churn.schema import FEATURES
from banking_churn.synthetic import make_synthetic_data,write_synthetic_csv
from banking_churn.training_core import choose_threshold


class InputTests(unittest.TestCase):
    def test_invalid_training_parameters_fail_before_fitting(self):
        for key,value in [("seed",True),("n_jobs",0),("outer_folds",1),("search_trials",0),
                          ("capacity_fraction",np.nan),("min_validation_precision",1.1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                replace(TrainingConfig(),**{key:value}).validate()

    def test_synthetic_generation_is_deterministic_and_explicit(self):
        a=make_synthetic_data(1000,42);b=make_synthetic_data(1000,42)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(a.CustomerId.astype(str).str.startswith("SYNTH-").all())
        self.assertGreaterEqual(a.Exited.value_counts().min(),30)
        self.assertEqual(set(normalize_features(a)),set(FEATURES))

    def test_synthetic_file_never_overwrites_existing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=write_synthetic_csv(Path(tmp)/"fixture.csv")
            original=target.read_bytes()
            with self.assertRaises(FileExistsError):write_synthetic_csv(target)
            self.assertEqual(original,target.read_bytes())

    def test_data_context_cannot_mislabel_synthetic_as_reference(self):
        with self.assertRaises(ValueError):
            run_training("unused.csv",config=TrainingConfig.quick(),strict_reference=False)
        with self.assertRaises(ValueError):
            run_training("unused.csv",config=TrainingConfig(),data_context="synthetic_smoke",strict_reference=False)

    def test_duplicate_customer_identifiers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame=make_synthetic_data();frame.loc[1,"CustomerId"]=frame.loc[0,"CustomerId"]
            path=Path(tmp)/"bad.csv";frame.to_csv(path,index=False)
            with warnings.catch_warnings(),self.assertRaisesRegex(ValueError,"CustomerId"):
                warnings.simplefilter("ignore")
                run_training(path,Path(tmp)/"out",TrainingConfig.quick(),"synthetic_smoke",False)


class FullPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.base=Path(cls.tmp.name)
        cls.source=write_synthetic_csv(cls.base/"synthetic.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            cls.training=run_pipeline(cls.source,cls.base/"artifacts",TrainingConfig.quick(),"synthetic_smoke",False)
        cls.directory=cls.training.artifact_dir

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_all_stages_complete_before_latest_manifest(self):
        manifest=json.loads((self.base/"artifacts/latest_run.json").read_text())
        self.assertEqual(manifest["status"],"complete")
        self.assertEqual(manifest["checks_passed"],13)
        self.assertTrue(all(c["status"]=="PASS" for c in self.training.checks))

    def test_splits_are_disjoint_cover_source_and_use_60_20_20(self):
        frames=self.training.frames
        sets=[set(f.index) for f in frames.values()]
        self.assertEqual([len(s) for s in sets],[600,200,200])
        self.assertEqual(len(set.union(*sets)),1000)
        self.assertTrue(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3)))
        self.assertFalse({"CustomerId","Surname","RowNumber","Exited"}&set(frames["train"].columns))

    def test_calibration_and_nested_search_are_exercised(self):
        self.assertTrue(self.training.metadata["calibrated"])
        self.assertEqual(self.training.metadata["outer_folds"],2)
        self.assertEqual(len(self.training.tables["nested_outer_folds"]),6)
        self.assertIn("raw",set(self.training.tables["calibration_diagnostic"].model))

    def test_fixed_half_policy_and_uncalibrated_path_are_supported(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            fixed=run_training(self.source,self.base/"fixed-policy",replace(TrainingConfig.quick(),
                default_policy="threshold_0_5",calibrate=False),"synthetic_smoke",False)
        self.assertEqual(fixed.metadata["threshold"],.5)
        self.assertFalse(fixed.metadata["calibrated"])
        self.assertTrue(all(c["status"]=="PASS" for c in fixed.checks))

    def test_threshold_is_recomputed_only_from_validation(self):
        expected,_=choose_threshold(self.training.labels["validation"],self.training.probabilities["validation"],1.)
        self.assertAlmostEqual(self.training.metadata["threshold"],expected)

    def test_artifact_reload_and_portable_helper_preserve_scores(self):
        spec=importlib.util.spec_from_file_location("portable_v7",self.directory/"predict_customers.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        bundle=joblib.load(self.directory/"bank_churn_bundle.joblib")
        batch=self.training.frames["validation"].head(4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected=predict_customers(batch,self.training.bundle)
            actual=module.predict_customers(batch,bundle)
        np.testing.assert_allclose(expected.estimated_churn_probability,actual.estimated_churn_probability,rtol=1e-12)

    def test_batch_csv_scoring_keeps_row_count_and_probability_bounds(self):
        source=self.base/"customer_examples.csv";output=self.base/"scored.csv"
        self.training.frames["validation"].head(4).to_csv(source,index=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result,path=score_customer_csv(source,self.training.bundle,output)
        self.assertEqual(len(pd.read_csv(path)),4)
        self.assertTrue(result.estimated_churn_probability.between(0,1).all())

    def test_reports_label_synthetic_data_and_expose_assumptions(self):
        reader=PdfReader(self.directory/"executive_summary.pdf")
        self.assertEqual(len(reader.pages),2)
        text="\n".join(p.extract_text() for p in reader.pages)
        for label in ["DATA SINTETIS","Biaya per kontak C","Keberhasilan u","SHA-256 sumber"]:
            self.assertIn(label,text)
        html=(self.directory/"executive_summary.html").read_text(encoding="utf-8")
        self.assertIn("DATA SINTETIS",html);self.assertIn("data:image/png;base64,",html)
        self.assertNotIn("<script",html)

    def test_public_summary_contains_no_customer_rows_or_identifiers(self):
        summary=aggregate_summary(self.directory)
        raw=json.dumps(summary)
        self.assertFalse(summary["customer_records_included"])
        self.assertFalse(summary["causal_campaign_effect_estimated"])
        for forbidden in ["CustomerId","Surname","SYNTH-","selected_positions","estimated_churn_probability"]:
            self.assertNotIn(forbidden,raw)
        self.assertTrue(summary["synthetic"])

    def test_archive_contains_complete_exports_and_passes_crc(self):
        manifest=json.loads((self.base/"artifacts/latest_run.json").read_text())
        with zipfile.ZipFile(manifest["archive"]) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue({"bank_churn_bundle.joblib","executive_summary.pdf","executive_summary.html",
                "public_summary.json","predict_customers.py","validation_checks.json"}<=set(archive.namelist()))

    def test_finish_is_idempotent_and_does_not_reseed_live_inputs(self):
        before=(self.directory/"v6_advisor_latest.json").read_bytes()
        manifest=finish_run(self.training,initialize_examples=False)
        self.assertEqual(manifest["checks_passed"],13)
        self.assertEqual(before,(self.directory/"v6_advisor_latest.json").read_bytes())

    def test_live_form_updates_html_pdf_and_archive_without_training(self):
        run=self.training
        advisor=v6_policy_advisor(run.labels["validation"],run.probabilities["validation"],
            run.metadata["policies"],0,0,10,100,.2,500,index=run.frames["validation"].index)
        original=(self.directory/"v6_advisor_latest.json").read_bytes()
        try:
            save_live_section("advisor",advisor["inputs"],{"status":advisor["status"],
                "recommended":advisor["recommended"]},self.directory,run.context)
            summary=aggregate_summary(self.directory)
            self.assertEqual(summary["latest_sections"]["advisor"]["inputs"]["max_contacts"],0)
            self.assertEqual(summary["latest_sections"]["advisor"]["summary"]["recommended"]["cost_um"],0)
            manifest=json.loads((self.base/"artifacts/latest_run.json").read_text())
            with zipfile.ZipFile(manifest["archive"]) as archive:
                exported=json.loads(archive.read("public_summary.json"))
                self.assertEqual(exported["latest_sections"]["advisor"]["inputs"]["max_contacts"],0)
        finally:
            (self.directory/"v6_advisor_latest.json").write_bytes(original)
            finish_run(run,initialize_examples=False)

    def test_report_rejects_snapshot_from_another_run(self):
        target=self.directory/"v6_advisor_latest.json";original=target.read_bytes()
        try:
            item=json.loads(original);item["run_id"]="different-run"
            target.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError,"Cross-run"):aggregate_summary(self.directory)
        finally:target.write_bytes(original)
