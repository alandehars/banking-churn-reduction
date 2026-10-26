"""Audit a frozen V7 model without fitting, tuning or asserting independence from a mode name."""
from dataclasses import asdict, dataclass
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .inference import normalize_features, predict_customers
from .schema import FEATURES, NUMERIC_COLS, BINARY_COLS, CATEGORY_COLS
from .v8_pilot import wilson_interval, _integer


@dataclass(frozen=True)
class ValidationEvidence:
    source_description: str = ""
    population: str = ""
    observation_period: str = ""
    horizon_days: int = 0
    target_definition_confirmed: bool = False
    labels_mature: bool = False
    independence_reviewed: bool = False

    def validate(self):
        for name in ("source_description","population","observation_period"):
            if not isinstance(getattr(self,name),str):
                raise ValueError(f"{name} harus teks.")
        _integer(self.horizon_days,"horizon_days",0,36500)
        for name in ("target_definition_confirmed","labels_mature","independence_reviewed"):
            if not isinstance(getattr(self,name),bool):
                raise ValueError(f"{name} harus boolean.")
        return self


def _ids(frame):
    if "CustomerId" not in frame:
        return None
    values = frame.CustomerId.astype("string").str.strip()
    if values.isna().any() or values.eq("").any() or not values.is_unique:
        raise ValueError("CustomerId harus lengkap dan unik setelah whitespace dibersihkan.")
    return values


def read_validation_csv(path):
    # Preserve leading zeros; use the same strict raw-header/row validation as V7.
    from .inference import read_source_csv
    return read_source_csv(path,required_columns=FEATURES,dtype={"CustomerId":"string"})


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature_fingerprints(frame):
    clean = normalize_features(frame)
    # Explicit order, numeric normalization and a typed null avoid ambiguous concatenations.
    result = set()
    for row in clean.itertuples(index=False,name=None):
        values = [None if pd.isna(v) else float(v) if name in NUMERIC_COLS+BINARY_COLS else str(v)
            for name,v in zip(FEATURES,row)]
        result.add(_digest(json.dumps(values,separators=(",",":"),ensure_ascii=False,allow_nan=False)))
    return result


def build_reference_profile(frame, model_bundle, source_sha256):
    """Private overlap index for the full model-development CSV, never a public artifact."""
    meta = model_bundle["metadata"]
    if not source_sha256 or source_sha256 != meta.get("data_sha256"):
        raise ValueError("Hash CSV referensi berbeda dari sumber development model.")
    identifiers = _ids(frame)
    return {"model_run_id":meta["run_id"],"source_sha256":source_sha256,"rows":len(frame),
        "customer_id_hashes":None if identifiers is None else {_digest(v) for v in identifiers},
        "feature_row_hashes":_feature_fingerprints(frame)}


def _metric_table(y,p,decision,resamples,seed):
    tp = int(((y==1)&(decision==1)).sum())
    fp = int(((y==0)&(decision==1)).sum())
    fn = int(((y==1)&(decision==0)).sum())
    tn = int(((y==0)&(decision==0)).sum())
    both = np.unique(y).size==2
    rows = []
    for name,events,total in [("precision",tp,tp+fp),("recall",tp,tp+fn),
        ("contact_rate",tp+fp,len(y)),("prevalence",tp+fn,len(y))]:
        low,high = wilson_interval(events,total)
        rows.append({"metric":name,"value":events/total if total else None,"denominator":total,
            "low_95":low,"high_95":high,"interval_method":"Wilson" if total else "undefined_zero_denominator"})
    functions = {"roc_auc":roc_auc_score,"average_precision":average_precision_score,
        "brier_score":brier_score_loss,"log_loss":lambda a,b:log_loss(a,b,labels=[0,1])}
    rng = np.random.default_rng(seed)
    samples = {name:[] for name in functions}
    for _ in range(resamples):
        positions = rng.integers(0,len(y),len(y))
        sy,sp = y[positions],p[positions]
        for name,fn_score in functions.items():
            if name in {"roc_auc","average_precision"} and np.unique(sy).size<2:
                continue
            samples[name].append(float(fn_score(sy,sp)))
    for name,fn_score in functions.items():
        defined = both or name in {"brier_score","log_loss"}
        values = samples[name] if defined else []
        interval = np.quantile(values,[.025,.975]).tolist() if values else [None,None]
        rows.append({"metric":name,"value":float(fn_score(y,p)) if defined else None,
            "denominator":len(y),"low_95":interval[0],"high_95":interval[1],
            "interval_method":"paired_row_percentile_bootstrap" if values else "undefined_single_class",
            "valid_resamples":len(values)})
    return pd.DataFrame(rows),{"tp":tp,"fp":fp,"fn":fn,"tn":tn}


def audit_frozen_model(incoming, model_bundle, evidence=None, reference_profile=None,
                       incoming_sha256=None, synthetic=False, resamples=200, seed=42):
    """Return descriptive audit and readiness gaps; never certify provenance or causality."""
    evidence = (evidence or ValidationEvidence()).validate()
    resamples = _integer(resamples,"resamples",20,2000)
    seed = _integer(seed,"seed",0,2**32-1)
    if not isinstance(synthetic,bool):
        raise ValueError("synthetic harus boolean.")
    meta = model_bundle["metadata"]
    if reference_profile is not None and (reference_profile["model_run_id"]!=meta["run_id"] or
        reference_profile["source_sha256"]!=meta.get("data_sha256")):
        raise ValueError("Profil referensi berasal dari model/sumber lain.")
    policy_before = json.dumps(meta["policies"],sort_keys=True)
    base = {"notebook_version":8,"model_run_id":meta["run_id"],"model_package_version":meta.get("package_version"),
        "policy":meta["default_policy"],"model_refit":False,"policy_tuned":False,
        "external_validation_verified":False,"evidence_is_user_declaration":True,
        "evidence":asdict(evidence),"causal_retention_effect_estimated":False,
        "synthetic":synthetic,"raw_customer_rows_included":False}
    empty = lambda summary:{"summary":summary,"metrics":pd.DataFrame(),"quality":pd.DataFrame(),
        "calibration":pd.DataFrame(),"scores":pd.DataFrame()}
    try:
        if not isinstance(incoming,pd.DataFrame) or incoming.empty or incoming.columns.duplicated().any():
            raise ValueError("Batch harus DataFrame tidak kosong dengan header unik.")
        extra = sorted(set(incoming.columns)-set(FEATURES)-{"CustomerId","Exited"})
        if extra:
            raise ValueError(f"Kolom batch tidak dikenal: {extra}")
        identifiers = _ids(incoming)
        clean = normalize_features(incoming)
        if identifiers is not None:
            clean.index = pd.Index(identifiers,name="CustomerId")
        if "Exited" in incoming:
            target = pd.to_numeric(incoming.Exited,errors="raise")
            if target.isna().any() or not target.isin([0,1]).all():
                raise ValueError("Exited harus lengkap dan 0/1; jangan isi label yang belum matang dengan nol.")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            scores = predict_customers(clean,model_bundle)
    except (TypeError,ValueError) as exc:
        return empty({**base,"status":"blocked_invalid_schema","errors":[str(exc)],"rows_scored":0,
            "recommendations":["Perbaiki schema/ID/label lalu jalankan ulang; tidak ada skor parsial yang diekspor."]})
    if policy_before!=json.dumps(meta["policies"],sort_keys=True):
        raise AssertionError("Audit mengubah kebijakan model.")
    synthetic = synthetic or bool(meta.get("synthetic",False)) or (identifiers is not None and bool(identifiers.str.startswith("SYNTH-").all()))
    profile = reference_profile
    id_overlap = row_overlap = None
    if profile is not None:
        row_keys = _feature_fingerprints(incoming)
        row_overlap = len(row_keys & profile["feature_row_hashes"])
        if identifiers is not None and profile["customer_id_hashes"] is not None:
            id_overlap = len({_digest(v) for v in identifiers} & profile["customer_id_hashes"])
    same_source = bool(incoming_sha256 and incoming_sha256==meta.get("data_sha256"))
    gaps = []
    if synthetic:gaps.append("Data sintetis: hanya demonstrasi software.")
    if same_source or (id_overlap or 0)>0 or (row_overlap or 0)>0:
        gaps.append("Ada data/ID/baris fitur yang sama dengan development; independensi belum terpenuhi.")
    if profile is None or id_overlap is None:
        gaps.append("Overlap ID terhadap seluruh data development belum dapat diperiksa.")
    if not incoming_sha256:gaps.append("Hash CSV batch belum dicatat.")
    if not evidence.independence_reviewed:gaps.append("Independensi populasi/periode belum ditinjau.")
    if not all(getattr(evidence,k).strip() for k in ("source_description","population","observation_period")):
        gaps.append("Sumber, populasi dan periode observasi perlu dijelaskan.")
    if not evidence.target_definition_confirmed or not evidence.horizon_days:
        gaps.append("Definisi target dan horizon label perlu dikonfirmasi.")
    if "Exited" not in incoming:gaps.append("Label belum tersedia; metrik diskriminasi/kalibrasi tidak dihitung.")
    elif not evidence.labels_mature:gaps.append("Label belum dinyatakan matang; metrik prediksi ditunda.")
    quality = []
    for name in FEATURES:
        unseen = float((clean[name].notna() & ~clean[name].isin(meta["training_categories"][name])).mean()) if name in CATEGORY_COLS else 0.
        bounds = meta["training_numeric_ranges"].get(name)
        outside = float(((clean[name]<bounds["minimum"])|(clean[name]>bounds["maximum"])).mean()) if bounds else 0.
        missing = float(clean[name].isna().mean())
        quality.append({"feature":name,"missing_rate":missing,"unseen_category_rate":unseen,
            "out_of_train_range_rate":outside,"review_needed":bool(missing or unseen or outside)})
    quality = pd.DataFrame(quality)
    if quality.review_needed.any():gaps.append("Tinjau missing, kategori asing atau nilai di luar rentang training.")
    metrics,counts,calibration = pd.DataFrame(),None,pd.DataFrame()
    if "Exited" in incoming and evidence.labels_mature:
        y = pd.to_numeric(incoming.Exited).to_numpy(dtype=int)
        p = scores.estimated_churn_probability.to_numpy()
        decision = scores.flag_for_review.to_numpy(dtype=int)
        metrics,counts = _metric_table(y,p,decision,resamples,seed)
        if min(int(y.sum()),len(y)-int(y.sum()))<30:
            gaps.append("Salah satu kelas kurang dari 30; estimasi terbatas, bukan bukti siap operasi.")
        grouped = pd.DataFrame({"risk":p,"actual":y,"bin":pd.cut(p,np.linspace(0,1,11),include_lowest=True)})
        calibration = grouped.groupby("bin",observed=True).agg(rows=("actual","size"),
            mean_estimated_risk=("risk","mean"),observed_rate=("actual","mean")).reset_index()
        calibration["bin"] = calibration.bin.astype(str)
    status = "synthetic_demonstration" if synthetic else "review_required" if gaps else "ready_for_independent_review"
    summary = {**base,"synthetic":bool(synthetic),"status":status,"errors":[],"rows_scored":len(scores),
        "incoming_sha256":incoming_sha256,"development_source_sha256":meta.get("data_sha256"),
        "same_source_file":same_source,"overlap_customer_ids":id_overlap,"overlap_unique_feature_rows":row_overlap,
        "overlap_detection_limit":"Changed/missing identifiers or changed feature rows can evade matching; no overlap is not proof of independence",
        "labels_evaluated":not metrics.empty,"confusion_counts":counts,
        "prediction_warnings":[str(w.message) for w in captured],"readiness_gaps":gaps,
        "recommendations":gaps+["Bekukan model/kebijakan; gunakan audit ini untuk review independen sebelum pilot terkontrol."],
        "interval_limit":"Row bootstrap assumes independent observations; it excludes model-development and population uncertainty"}
    return {"summary":summary,"metrics":metrics,"quality":quality,"calibration":calibration,"scores":scores}


class DemoRiskEstimator:
    """Fixed, intentionally untrained scoring rule for the V8 synthetic demonstration."""
    classes_ = np.array([0,1])

    def predict_proba(self,frame):
        p = np.clip(.1+.004*(frame.Age.fillna(40)-18)+.000001*frame.Balance.fillna(0)
            -.08*frame.IsActiveMember.fillna(0),.01,.99).to_numpy()
        return np.column_stack([1-p,p])


def demo_bundle(reference, source_sha256):
    from .schema import NUMERIC_RULES
    return {"estimator":DemoRiskEstimator(),"metadata":{"run_id":"v8-synthetic-untrained-demo",
        "package_version":"synthetic_rule","synthetic":True,"features":FEATURES,"numeric_features":NUMERIC_COLS,
        "binary_features":BINARY_COLS,"categorical_features":CATEGORY_COLS,"numeric_rules":NUMERIC_RULES,
        "training_categories":{c:sorted(reference[c].dropna().unique().tolist()) for c in CATEGORY_COLS},
        "training_numeric_ranges":{c:{"minimum":float(reference[c].min()),"maximum":float(reference[c].max())} for c in NUMERIC_COLS},
        "data_sha256":source_sha256,"positive_class":1,"default_policy":"balanced_f1",
        "policies":{"balanced_f1":{"kind":"threshold","threshold":.3}}}}
