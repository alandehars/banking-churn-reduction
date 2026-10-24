"""Shared schema validation and inference used by V7 notebooks, CLI and tests."""
import csv
import math
import tempfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import TARGET,NUMERIC_COLS,BINARY_COLS,CATEGORY_COLS,FEATURES,NUMERIC_RULES


def read_source_csv(path, required_columns=None, dtype=None):
    # pandas mengganti header duplikat; periksa header asli terlebih dahulu.
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next((row for row in reader if row), None)
        if not header:
            raise ValueError("CSV kosong atau tidak memiliki header.")
        header = [name.strip() for name in header]
        if any(not name for name in header):
            raise ValueError("Header CSV mengandung nama kolom kosong.")
        if len(header) != len(set(header)):
            raise ValueError("Header CSV duplikat setelah pembersihan.")
        expected = FEATURES + [TARGET] if required_columns is None else list(required_columns)
        absent = sorted(set(expected) - set(header))
        if absent:
            raise ValueError(f"Kolom wajib tidak tersedia: {absent}")
        for row in reader:
            if row and len(row) != len(header):
                raise ValueError(f"Jumlah kolom tidak konsisten pada baris CSV {reader.line_num}.")
    frame = pd.read_csv(path, encoding="utf-8-sig", header=0, names=header, dtype=dtype)
    if frame.empty:
        raise ValueError("CSV tidak memiliki baris data.")
    return frame


def normalize_features(frame, strict_columns=False, numeric_rules=None):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input harus DataFrame berisi minimal satu baris.")
    if frame.columns.duplicated().any():
        raise ValueError("Kolom input duplikat.")
    absent = sorted(set(FEATURES) - set(frame.columns))
    if absent:
        raise ValueError(f"Fitur wajib tidak tersedia: {absent}")
    extra = sorted(set(frame.columns) - set(FEATURES))
    if strict_columns and extra:
        raise ValueError(f"Kolom prediksi tidak dikenal: {extra}. Pilih hanya fitur model sebelum prediksi.")
    clean = frame.loc[:, FEATURES].copy()
    rules = NUMERIC_RULES if numeric_rules is None else numeric_rules
    for name in NUMERIC_COLS + BINARY_COLS:
        clean[name] = pd.to_numeric(clean[name], errors="raise").astype(float)
        if np.isinf(clean[name]).any():
            raise ValueError(f"{name} mengandung infinity.")
        nonmissing = clean[name].dropna()
        rule = rules.get(name, {})
        if "minimum" in rule and (nonmissing < rule["minimum"]).any():
            raise ValueError(f"{name} di bawah batas minimum {rule['minimum']} pada skema sumber.")
        if "maximum" in rule and (nonmissing > rule["maximum"]).any():
            raise ValueError(f"{name} di atas batas maksimum {rule['maximum']} pada skema sumber.")
        if rule.get("integer") and (nonmissing != np.floor(nonmissing)).any():
            raise ValueError(f"{name} harus bilangan bulat sesuai skema sumber.")
    for name in BINARY_COLS:
        if not clean[name].dropna().isin([0, 1]).all():
            raise ValueError(f"{name} harus 0, 1, atau missing.")
    for name in CATEGORY_COLS:
        clean[name] = clean[name].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
        clean[name] = clean[name].replace("", np.nan).astype(object)
    return clean


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilitas harus vektor finite, tidak kosong, dan berada di antara 0 dan 1.")
    return p


def rank_order(probabilities, index=None):
    p = validate_probabilities(probabilities)
    keys = np.arange(len(p)) if index is None else np.asarray(index)
    if keys.ndim != 1 or len(keys) != len(p) or pd.isna(keys).any():
        raise ValueError("Index ranking tidak sesuai skor atau mengandung nilai kosong.")
    text_keys = np.asarray([str(key) for key in keys])
    if not pd.Index(text_keys).is_unique:
        raise ValueError("Index ranking harus unik, termasuk setelah diubah menjadi teks.")
    return np.lexsort((text_keys, -p))


def apply_policy(probabilities, policy, index=None):
    p = validate_probabilities(probabilities)
    kind = policy.get("kind")
    if kind == "threshold":
        threshold = policy.get("threshold")
        if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold kebijakan tidak valid.")
        return (p >= threshold).astype(int)
    if kind == "top_k":
        fraction = policy.get("fraction")
        if fraction is None or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Fraksi kapasitas harus antara 0 dan 1.")
        k = min(len(p), math.ceil(fraction * len(p)))
        decision = np.zeros(len(p), dtype=int)
        decision[rank_order(p, index)[:k]] = 1
        return decision
    raise ValueError(f"Jenis kebijakan tidak dikenal: {kind}")


def predict_customers(frame, model_bundle, policy_name=None):
    meta = model_bundle["metadata"]
    if meta["features"] != FEATURES:
        raise ValueError("Skema fitur notebook berbeda dari bundle.")
    clean = normalize_features(frame, strict_columns=True, numeric_rules=meta["numeric_rules"])
    for column in meta["categorical_features"]:
        unseen = sorted(set(clean[column].dropna()) - set(meta["training_categories"][column]))
        if unseen:
            warnings.warn(f"Kategori baru pada {column}: {unseen}; skor di luar dukungan kategori train.")
    for column, bounds in meta["training_numeric_ranges"].items():
        outside = (clean[column] < bounds["minimum"]) | (clean[column] > bounds["maximum"])
        if outside.any():
            warnings.warn(f"{column} di luar rentang train pada {int(outside.sum())} baris.")
    estimator = model_bundle["estimator"]
    class_index = int(np.flatnonzero(estimator.classes_ == meta["positive_class"])[0])
    score = validate_probabilities(estimator.predict_proba(clean)[:, class_index])
    if "policies" in meta:
        name = meta["default_policy"] if policy_name is None else policy_name
        if name not in meta["policies"]:
            raise ValueError(f"Kebijakan tidak tersedia dalam bundle: {name}")
        policy = meta["policies"][name]
    else:  # Kompatibilitas untuk bundle review 2.
        name = "stored_threshold"
        policy = {"kind": "threshold", "threshold": meta["threshold"]}
    order = rank_order(score, frame.index)
    ranks = np.empty(len(score), dtype=int)
    ranks[order] = np.arange(1, len(score) + 1)
    return pd.DataFrame({
        "estimated_churn_probability": score,
        "flag_for_review": apply_policy(score, policy, frame.index),
        "rank_within_batch": ranks, "policy": name,
        "decision_threshold": policy.get("threshold", np.nan),
    }, index=frame.index)


def score_customer_csv(input_path, model_bundle, output_path=None, policy_name=None):
    input_path = Path(input_path)
    frame = read_source_csv(input_path, required_columns=FEATURES, dtype={"CustomerId": "string"})
    extra = sorted(set(frame.columns) - set(FEATURES) - {"CustomerId"})
    if extra:
        raise ValueError(f"Batch prediksi hanya menerima fitur dan CustomerId opsional. Kolom tambahan: {extra}")
    if "CustomerId" in frame:
        if frame["CustomerId"].isna().any() or not frame["CustomerId"].is_unique:
            raise ValueError("CustomerId pada batch harus lengkap dan unik.")
        frame = frame.set_index("CustomerId")
    else:
        frame.index = pd.Index(np.arange(1, len(frame) + 1), name="row_number")
    result = predict_customers(frame.loc[:, FEATURES], model_bundle, policy_name)
    target = Path(output_path) if output_path is not None else input_path.with_name(input_path.stem + "_scored.csv")
    if target.resolve() == input_path.resolve():
        raise ValueError("Output prediksi tidak boleh menimpa CSV input.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".scores-", suffix=".csv", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        result.to_csv(temporary, index=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return result, target
