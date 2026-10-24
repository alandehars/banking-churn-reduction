"""Shared V7 training/evaluation pipeline used by notebooks, CLI and integration tests."""
from dataclasses import dataclass
from datetime import datetime,timezone
import ast
import hashlib
import importlib.metadata as package_metadata
import json
import platform
from pathlib import Path
import shutil
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,brier_score_loss,confusion_matrix,log_loss,roc_auc_score)
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from . import __version__
from .config import TrainingConfig
from .data import load_source_data
from .decision_tools import (v6_compare_versions,v6_fingerprint,v6_json,v6_policy_advisor,
    v6_refresh_report,v6_save_section,v6_uncertain_campaign,v6_batch_quality)
from .inference import normalize_features,predict_customers,apply_policy
from .schema import (TARGET,FEATURES,NUMERIC_COLS,BINARY_COLS,CATEGORY_COLS,NUMERIC_RULES,
    DATA_SOURCE_URL,DATA_DRIVE_ID,SOURCE_NOTEBOOK_ID,REFERENCE_SHA256)
from .training_core import nested_compare,make_search,choose_threshold,policy_metrics,bootstrap_intervals


@dataclass
class TrainingRun:
    model: object
    bundle: dict
    metadata: dict
    artifact_dir: Path
    frames: dict
    labels: dict
    probabilities: dict
    tables: dict
    context: dict
    checks: list

    @property
    def run_id(self):
        return self.metadata["run_id"]


def model_specifications(config):
    specs={
        "Dummy":{"estimator":DummyClassifier(strategy="prior"),"parameters":{}},
        "LogisticRegression":{"estimator":LogisticRegression(max_iter=2000,random_state=config.seed),
            "parameters":{"model__C":[.05,.2,1.,5.,20.],"model__class_weight":[None,"balanced"]}},
        "RandomForest":{"estimator":RandomForestClassifier(n_jobs=config.n_jobs,random_state=config.seed),
            "parameters":{"model__n_estimators":[150,300],"model__max_depth":[6,12,None],
                "model__min_samples_leaf":[2,5,10],"model__max_features":["sqrt",.8],"model__class_weight":[None,"balanced_subsample"]}},
        "HistGradientBoosting":{"estimator":HistGradientBoostingClassifier(early_stopping=False,random_state=config.seed),
            "parameters":{"model__max_iter":[150,300],"model__learning_rate":[.04,.08,.12],
                "model__max_leaf_nodes":[7,15,31],"model__l2_regularization":[0.,1.,5.],"model__min_samples_leaf":[15,30,50]}},
        "XGBoost":{"estimator":XGBClassifier(objective="binary:logistic",eval_metric="logloss",tree_method="hist",n_jobs=config.n_jobs,random_state=config.seed),
            "parameters":{"model__n_estimators":[200,400,600],"model__learning_rate":[.03,.06,.1],
                "model__max_depth":[2,3,4,5],"model__min_child_weight":[1,5,10],"model__subsample":[.8,1.],
                "model__colsample_bytree":[.8,1.],"model__reg_lambda":[1.,3.,8.]}}
    }
    if config.smoke:
        specs={name:specs[name] for name in ["Dummy","LogisticRegression","HistGradientBoosting"]}
        specs["HistGradientBoosting"]["parameters"]={"model__max_iter":[40],"model__max_leaf_nodes":[7],"model__min_samples_leaf":[15]}
        specs["LogisticRegression"]["parameters"]={"model__C":[1.]}
    return specs


def runtime_versions():
    packages=["numpy","pandas","scipy","scikit-learn","matplotlib","seaborn","joblib","xgboost", "gdown",
              "packaging","ipywidgets","threadpoolctl","nbformat","nbclient","ipykernel","reportlab"]
    return {**{name:package_metadata.version(name) for name in packages},"python":platform.python_version(),"banking-churn-reduction":__version__}


def _segment_audit(frame,labels,prediction):
    audit=frame[["Geography","Gender","Age"]].copy()
    audit["AgeGroup"]=pd.cut(audit.Age,[-np.inf,30,45,60,np.inf],labels=["<=30","31-45","46-60",">60"])
    audit["actual"],audit["predicted"]=labels,prediction
    rows=[]
    for attribute in ["Geography","Gender","AgeGroup"]:
        for group,sub in audit.groupby(attribute,observed=True,dropna=False):
            positives=int(sub.actual.sum());negative=len(sub)-positives
            tn,fp,fn,tp=confusion_matrix(sub.actual,sub.predicted,labels=[0,1]).ravel()
            rows.append({"attribute":attribute,"group":str(group),"n":len(sub),"churn_n":positives,
                "churn_rate":float(sub.actual.mean()),"recall":tp/positives if positives else np.nan,
                "false_positive_rate":fp/negative if negative else np.nan,
                "limited_sample":len(sub)<100 or min(positives,negative)<20})
    return pd.DataFrame(rows)


def portable_inference_source():
    """Export exact shared schema/inference code without requiring this project package."""
    from . import schema,inference
    schema_source=Path(schema.__file__).read_text(encoding="utf-8")
    inference_source=Path(inference.__file__).read_text(encoding="utf-8")
    tree=ast.parse(inference_source)
    keep=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.level)]
    return '"""Portable V7 inference. Load only a trusted bundle with matching package versions."""\n'+schema_source+"\n"+ast.unparse(ast.Module(body=keep,type_ignores=[]))+"\n"


def _write_json(path,value):
    Path(path).write_text(json.dumps(v6_json(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def run_training(data_file,output_base="bank_churn_artifacts_v7",config=None,data_context="reference_source",strict_reference=True):
    """Train and audit once; no test-based model/policy selection and no directory reuse."""
    config=(config or TrainingConfig()).validate()
    if data_context not in {"reference_source","synthetic_smoke","external_reviewed"}:
        raise ValueError("Data context must be explicit")
    if data_context=="synthetic_smoke" and (not config.smoke or strict_reference):
        raise ValueError("Synthetic smoke data requires quick configuration and explicit non-reference mode")
    if config.smoke and data_context!="synthetic_smoke":
        raise ValueError("Smoke configuration is reserved for explicitly synthetic data")
    if not strict_reference and data_context=="reference_source":
        raise ValueError("Unverified data cannot be labelled as the reference source")
    with threadpool_limits(limits=config.n_jobs):
        return _run_training(data_file,Path(output_base),config,data_context,strict_reference)


def _run_training(data_file,base,config,data_context,strict_reference):
    started=time.perf_counter()
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    directory=base/"runs"/run_id
    directory.mkdir(parents=True,exist_ok=False)
    raw,source=load_source_data(Path(data_file),strict_reference=strict_reference)
    frame=raw.drop_duplicates().copy()
    if "CustomerId" in frame and (frame.CustomerId.isna().any() or frame.CustomerId.duplicated().any()):
        raise ValueError("CustomerId must be complete and unique; repeated customers need group/time splitting")
    labels=pd.to_numeric(frame[TARGET],errors="raise")
    if labels.isna().any() or not labels.isin([0,1]).all() or labels.nunique()!=2 or labels.value_counts().min()<30:
        raise ValueError("Target must be complete, binary and have at least 30 records per class")
    labels=labels.astype(int)
    features=normalize_features(frame)
    development,test,yd,yt=train_test_split(features,labels,test_size=.2,stratify=labels,random_state=config.seed)
    train,valid,ytrain,yvalid=train_test_split(development,yd,test_size=.25,stratify=yd,random_state=config.seed)
    if train.isna().all().any():
        raise ValueError("A training feature is entirely missing")
    frames={"train":train,"validation":valid,"test":test}
    targets={"train":ytrain,"validation":yvalid,"test":yt}
    split_fingerprints={key:v6_fingerprint(value.index) for key,value in frames.items()}
    leaderboard,folds,oof,fold_ids=nested_compare(train,ytrain,model_specifications(config),config.outer_folds,
        config.inner_folds,config.search_trials,config.seed)
    name=str(leaderboard.iloc[0].model)
    search=make_search(model_specifications(config)[name],config.seed+10,config.inner_folds,config.search_trials)
    if config.calibrate and name!="Dummy":
        estimator=CalibratedClassifierCV(estimator=search,method="sigmoid",ensemble=False,
            cv=StratifiedKFold(n_splits=config.outer_folds,shuffle=True,random_state=config.seed+20),n_jobs=1).fit(train,ytrain)
        fitted_search=estimator.calibrated_classifiers_[0].estimator
    else:
        fitted_search=search.fit(train,ytrain);estimator=fitted_search
    positive=int(np.flatnonzero(estimator.classes_==1)[0])
    pv=estimator.predict_proba(valid)[:,positive]
    raw_p=fitted_search.predict_proba(valid)[:,positive]
    policies={"threshold_0_5":{"kind":"threshold","threshold":.5}}
    threshold_tables={}
    for policy,beta in [("precision_f05",.5),("balanced_f1",1.),("recall_f2",2.)]:
        threshold,table=choose_threshold(yvalid,pv,beta,config.min_validation_precision,config.max_validation_contact_rate)
        policies[policy]={"kind":"threshold","threshold":threshold,"validation_beta":beta}
        threshold_tables[policy]=table
    policies["capacity_top_k"]={"kind":"top_k","fraction":config.capacity_fraction}
    threshold=float(policies[config.default_policy]["threshold"])
    # First evaluate test only after family, calibration and policies have been fixed.
    pt=estimator.predict_proba(test)[:,positive]
    prediction=apply_policy(pt,policies[config.default_policy],test.index)
    valid_metrics=pd.DataFrame([policy_metrics(yvalid,pv,p,key,valid.index) for key,p in policies.items()])
    test_metrics=pd.DataFrame([policy_metrics(yt,pt,p,key,test.index) for key,p in policies.items()])
    default=test_metrics.loc[test_metrics.policy==config.default_policy].iloc[0]
    scores={"roc_auc":roc_auc_score(yt,pt),"average_precision":average_precision_score(yt,pt),
        "brier_score":brier_score_loss(yt,pt),"log_loss":log_loss(yt,pt),"test_rows":len(yt),
        "test_churn":int(yt.sum()),"test_prevalence":float(yt.mean())}
    perm=permutation_importance(estimator,valid,yvalid,scoring="average_precision",n_repeats=2 if config.smoke else 5,
        random_state=config.seed,n_jobs=1)
    importance=pd.DataFrame({"feature":FEATURES,"ap_decrease_mean":perm.importances_mean,
        "ap_decrease_sd":perm.importances_std}).sort_values("ap_decrease_mean",ascending=False)
    capacity=[]
    for fraction in sorted({.05,.1,.2,.3,config.capacity_fraction}):
        decision=apply_policy(pt,{"kind":"top_k","fraction":fraction},test.index).astype(bool)
        k=int(decision.sum());tp=int(yt.to_numpy()[decision].sum())
        capacity.append({"capacity_fraction":fraction,"contacts":k,"churn_captured":tp,
            "precision_at_k":tp/k,"recall_at_k":tp/int(yt.sum()),"lift_at_k":(tp/k)/float(yt.mean())})
    versions=runtime_versions()
    metadata={"revision":7,"package_version":__version__,"run_id":run_id,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "data_context":data_context,"synthetic":data_context=="synthetic_smoke","data_sha256":source["sha256"],
        "source_reference_matches":bool(source["reference_matches"]),"source_rows":len(frame),"rows_used":len(frame),
        "source_url":DATA_SOURCE_URL if data_context=="reference_source" else None,
        "source_notebook_id":SOURCE_NOTEBOOK_ID,"data_drive_id":DATA_DRIVE_ID if data_context=="reference_source" else None,
        "config":config.as_dict(),"seed":config.seed,"model_name":name,"model":name,
        "selection_metric":"train_nested_cv_average_precision","outer_folds":config.outer_folds,
        "inner_folds":config.inner_folds,"search_trials":config.search_trials,"best_parameters":fitted_search.best_params_,
        "calibrated":config.calibrate and name!="Dummy","calibration_method":"sigmoid" if config.calibrate and name!="Dummy" else None,
        "threshold":threshold,"default_policy":config.default_policy,"policies":policies,"features":FEATURES,
        "numeric_features":NUMERIC_COLS,"binary_features":BINARY_COLS,"categorical_features":CATEGORY_COLS,
        "numeric_rules":NUMERIC_RULES,"training_categories":{c:sorted(train[c].dropna().unique().tolist()) for c in CATEGORY_COLS},
        "training_numeric_ranges":{c:{"minimum":float(train[c].min()),"maximum":float(train[c].max())} for c in NUMERIC_COLS},
        "positive_class":1,"versions":versions,"test_scores":scores,"split_fingerprints":split_fingerprints,
        "test_holdout_status":"synthetic_test_only" if config.smoke else "iterative_internal_holdout",
        "training_seconds":time.perf_counter()-started,
        "limits":["Synthetic tests do not validate bank performance" if config.smoke else "No new temporal/external validation",
            "No causal retention effect estimated","Business parameters remain assumptions"]}
    bundle={"estimator":estimator,"metadata":v6_json(metadata)}
    joblib.dump(bundle,directory/"bank_churn_bundle.joblib",compress=3)
    reloaded=joblib.load(directory/"bank_churn_bundle.joblib")
    example=pd.DataFrame([{ "CreditScore":650,"Age":45,"Tenure":5,"Balance":50000.,"NumOfProducts":2,
        "EstimatedSalary":60000.,"HasCrCard":1,"IsActiveMember":1,"Geography":"France","Gender":"Male"}],
        index=pd.Index(["EXAMPLE-001"],name="example_id"))
    portable=portable_inference_source()
    (directory/"predict_customers.py").write_text(portable,encoding="utf-8")
    standalone={"__name__":"portable_inference_check"}
    exec(compile(portable,"predict_customers.py","exec"),standalone)
    tables={"nested_model_comparison":leaderboard,"nested_outer_folds":folds,
        "validation_policies":valid_metrics,"test_policies":test_metrics,
        "test_bootstrap_intervals":bootstrap_intervals(yt,pt,threshold,config.bootstrap_resamples,config.seed),
        "test_capacity":pd.DataFrame(capacity),"validation_importance":importance,
        "test_segments":_segment_audit(test,yt,prediction),
        "calibration_diagnostic":pd.DataFrame([{"model":"raw","AP":average_precision_score(yvalid,raw_p),"Brier":brier_score_loss(yvalid,raw_p)},
            {"model":"calibrated" if metadata["calibrated"] else "final","AP":average_precision_score(yvalid,pv),"Brier":brier_score_loss(yvalid,pv)}]),
        "split_summary":pd.DataFrame([{"split":k,"rows":len(v),"churn_n":int(v.sum()),"churn_rate":float(v.mean())} for k,v in targets.items()])}
    context=v6_json({"run_id":run_id,"data_sha256":source["sha256"],"model":name,"test_scores":scores,
        "default_metrics":default.to_dict(),"threshold":threshold,"default_policy":config.default_policy,
        "versions":versions,"split_fingerprints":split_fingerprints,"rows_used":len(frame),"source_url":metadata["source_url"],
        "data_context":data_context,"synthetic":metadata["synthetic"]})
    checks=[]
    def check(label,condition):
        if not bool(condition):raise AssertionError(label)
        checks.append({"check":label,"status":"PASS"})
    sets=[set(v.index) for v in frames.values()]
    check("Three disjoint splits",all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3)))
    check("Splits cover every source row",set.union(*sets)==set(features.index))
    check("Identifiers and target excluded",not ({"CustomerId","Surname","RowNumber",TARGET}&set(FEATURES)))
    check("Every train row has out-of-fold scores",oof.index.equals(train.index) and np.isfinite(oof.to_numpy()).all())
    check("Nested selection uses train scores",name==str(leaderboard.iloc[0].model))
    check("Default threshold is fixed or selected from validation",
        threshold==.5 if config.default_policy=="threshold_0_5" else np.isclose(pv,threshold).any())
    check("Test probabilities are finite",np.isfinite(pt).all() and ((pt>=0)&(pt<=1)).all())
    check("Confusion counts equal test rows",int(default.tp+default.fp+default.fn+default.tn)==len(yt))
    check("Reload preserves predictions",np.allclose(predict_customers(example,bundle).estimated_churn_probability,
        predict_customers(example,reloaded).estimated_churn_probability,rtol=1e-12,atol=1e-12))
    check("Standalone helper preserves predictions",np.allclose(predict_customers(example,bundle).estimated_churn_probability,
        standalone["predict_customers"](example,reloaded).estimated_churn_probability,rtol=1e-12,atol=1e-12))
    check("Importance uses raw feature names",set(importance.feature)==set(FEATURES))
    check("Synthetic/reference context is explicit",metadata["synthetic"]==(data_context=="synthetic_smoke"))
    for key,table in tables.items():table.to_csv(directory/(key+".csv"),index=False,lineterminator="\n")
    for key,value in [("model_metadata",metadata),("test_scores",scores),("validation_checks",checks)]:
        _write_json(directory/(key+".json"),value)
    (directory/"requirements-tested.txt").write_text("\n".join(f"{k}=={v}" for k,v in versions.items() if k not in {"python","banking-churn-reduction"})+"\n",encoding="utf-8")
    _write_json(directory/"run_context.json",context)
    return TrainingRun(estimator,reloaded,v6_json(metadata),directory,frames,targets,
        {"validation":pv,"test":pt},tables,context,checks)


def initialize_decision_examples(run):
    """Create explicit validation-only defaults; does not alter the trained model/policies."""
    pv=run.probabilities["validation"];yv=run.labels["validation"];xv=run.frames["validation"]
    count=min(400,len(yv));budget=count*10+500
    advisor=v6_policy_advisor(yv,pv,run.metadata["policies"],count,budget,10,100,.2,500,index=xv.index)
    v6_save_section("advisor",advisor["inputs"],{"status":advisor["status"],"recommended":advisor["recommended"],
        "data_context":"validation","business_inputs_are_assumptions":True},run.artifact_dir,run.context,
        {"candidates":advisor["candidates"],"top_k_curve":advisor["top_k_curve"]})
    uncertainty=v6_uncertain_campaign(pv,count,[.1,.3],[8,12],[80,120],500,n_draws=500 if run.metadata["config"]["smoke"] else 2000,index=xv.index)
    v6_save_section("uncertainty",uncertainty["assumptions"],{"results":uncertainty["summary"],"assumptions":uncertainty["assumptions"]},
        run.artifact_dir,run.context,{"draws":uncertainty["draws"]})
    batch=v6_batch_quality(run.frames["train"],xv.head(min(200,len(xv))),run.bundle,predict_customers,pv,data_context="illustrative_validation")
    v6_save_section("batch",{"source":"illustrative_validation"},batch["summary"],run.artifact_dir,run.context,
        {"features":batch["features"],"scores":batch["scores"]})
    return advisor,uncertainty,batch


def finish_run(run,initialize_examples=True):
    """Publish a complete manifest only after reports, archive and export checks succeed."""
    from .reporting import generate_executive_reports,plot_model_audit
    plot_model_audit(run)
    if initialize_examples:
        initialize_decision_examples(run)
    reports=generate_executive_reports(run.artifact_dir)
    for path in reports.values():
        if not Path(path).is_file() or Path(path).stat().st_size==0:raise AssertionError("Missing report export")
    if not any(item["check"]=="HTML, PDF and aggregate summary exported" for item in run.checks):
        run.checks.append({"check":"HTML, PDF and aggregate summary exported","status":"PASS"})
    _write_json(run.artifact_dir/"validation_checks.json",run.checks)
    archive=Path(shutil.make_archive(str(run.artifact_dir.parent.parent/run.run_id),"zip",root_dir=run.artifact_dir))
    import zipfile
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise AssertionError("Invalid ZIP export")
        if not {"executive_summary.html","executive_summary.pdf","public_summary.json","predict_customers.py"}<=set(z.namelist()):
            raise AssertionError("Missing ZIP content")
    manifest={"run_id":run.run_id,"revision":7,"status":"complete","data_context":run.metadata["data_context"],
        "synthetic":run.metadata["synthetic"],"run_dir":str(run.artifact_dir.resolve()),"archive":str(archive.resolve()),
        "source_sha256":run.metadata["data_sha256"],"model":run.metadata["model_name"],"checks_passed":len(run.checks),
        "test_scores":run.metadata["test_scores"],"package_version":__version__,"config":run.metadata["config"]}
    base=run.artifact_dir.parent.parent
    temporary=base/(".latest-"+run.run_id+".json")
    _write_json(temporary,manifest);temporary.replace(base/"latest_run.json")
    return manifest


def run_pipeline(data_file,output_base="bank_churn_artifacts_v7",config=None,data_context="reference_source",strict_reference=True):
    run=run_training(data_file,output_base,config,data_context,strict_reference)
    finish_run(run)
    return run
