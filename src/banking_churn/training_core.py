"""Train-only nested selection and validation-only policy functions."""
import json
import math
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,brier_score_loss,
    confusion_matrix,f1_score,fbeta_score,log_loss,precision_recall_curve,precision_score,recall_score,roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from .preprocessing import make_pipeline
from .inference import apply_policy


def validate_label_scores(labels, probabilities):
    label_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=float)
    if label_array.ndim != 1 or probability_array.ndim != 1 or len(label_array) != len(probability_array):
        raise ValueError("Label dan probabilitas harus vektor 1D dengan panjang yang sama.")
    if len(label_array) == 0 or not np.isin(label_array, [0, 1]).all() or np.unique(label_array).size != 2:
        raise ValueError("Label wajib lengkap, bernilai 0/1, dan memiliki kedua kelas.")
    if not np.isfinite(probability_array).all() or ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Probabilitas wajib finite dan berada di antara 0 dan 1.")
    return label_array.astype(int), probability_array


def choose_threshold(labels, probabilities, beta=2.0, min_precision=None, max_contact_rate=None):
    labels, probabilities = validate_label_scores(labels, probabilities)
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta harus finite dan positif.")
    for name, value in [("min_precision", min_precision), ("max_contact_rate", max_contact_rate)]:
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} harus None atau angka finite antara 0 dan 1.")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    p, r = precision[:-1], recall[:-1]
    # Bentuk ekuivalen ini menghindari overflow ketika beta besar.
    weight = (1.0 / beta) ** 2 if beta >= 1 else beta ** 2
    denominator = p + weight * r if beta >= 1 else weight * p + r
    fbeta = (1 + weight) * p * r / np.maximum(denominator, np.finfo(float).eps)
    contacts = len(probabilities) - np.searchsorted(np.sort(probabilities), thresholds, side="left")
    table = pd.DataFrame({
        "threshold": thresholds, "precision": p, "recall": r, "fbeta": fbeta,
        "contacts": contacts, "contact_rate": contacts / len(probabilities),
    })
    table["eligible"] = True
    if min_precision is not None:
        table["eligible"] &= table["precision"] >= min_precision
    if max_contact_rate is not None:
        table["eligible"] &= table["contact_rate"] <= max_contact_rate
    feasible = table.loc[table["eligible"]]
    if feasible.empty:
        raise ValueError("Tidak ada threshold validation yang memenuhi batas precision/kapasitas. Tinjau konfigurasi.")
    best = feasible.sort_values(["fbeta", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), table


def metric_row(labels, probabilities, threshold, label):
    labels, probabilities = validate_label_scores(labels, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold harus finite antara 0 dan 1.")
    prediction = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "policy": label, "threshold": float(threshold),
        "accuracy": accuracy_score(labels, prediction),
        "balanced_accuracy": balanced_accuracy_score(labels, prediction),
        "precision_churn": precision_score(labels, prediction, zero_division=0),
        "recall_churn": recall_score(labels, prediction, zero_division=0),
        "f1_churn": f1_score(labels, prediction, zero_division=0),
        "f2_churn": fbeta_score(labels, prediction, beta=2, zero_division=0),
        "average_precision": average_precision_score(labels, probabilities),
        "roc_auc": roc_auc_score(labels, probabilities),
        "brier_score": brier_score_loss(labels, probabilities),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "contact_rate": prediction.mean(), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_intervals(labels, probabilities, threshold, n_resamples=500, seed=42):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    scores = {"roc_auc": [], "average_precision": [], "precision_churn": [], "recall_churn": []}
    for _ in range(n_resamples):
        sample = rng.integers(0, len(labels), size=len(labels))
        sample_y, sample_p = labels[sample], probabilities[sample]
        if np.unique(sample_y).size < 2:
            continue
        pred = (sample_p >= threshold).astype(int)
        scores["roc_auc"].append(roc_auc_score(sample_y, sample_p))
        scores["average_precision"].append(average_precision_score(sample_y, sample_p))
        scores["precision_churn"].append(precision_score(sample_y, pred, zero_division=0))
        scores["recall_churn"].append(recall_score(sample_y, pred, zero_division=0))
    return pd.DataFrame([
        {"metric": name, "low_95": np.quantile(values, 0.025), "high_95": np.quantile(values, 0.975),
         "valid_resamples": len(values)} for name, values in scores.items()
    ])


def make_search(specification, seed, folds, trials):
    parameters = specification.get("parameters", {})
    combinations = math.prod(len(values) for values in parameters.values()) if parameters else 1
    return RandomizedSearchCV(
        make_pipeline(clone(specification["estimator"])), parameters,
        n_iter=min(trials, combinations),
        scoring={"ap": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"},
        refit="ap", cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        random_state=seed, n_jobs=1, error_score="raise", return_train_score=False,
    )


def nested_compare(features, labels, specifications, outer_folds=3, inner_folds=3, trials=6, seed=42):
    if not isinstance(features, pd.DataFrame) or not features.index.is_unique:
        raise ValueError("Nested CV memerlukan DataFrame dengan index unik.")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(features) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Label nested CV harus vektor 0/1 yang sesuai baris fitur.")
    if min(outer_folds, inner_folds) < 2 or trials < 1 or not specifications:
        raise ValueError("Konfigurasi fold, search trials, atau kandidat tidak valid.")
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    fold_rows, oof = [], pd.DataFrame(index=features.index)
    fold_ids = np.full(len(features), -1, dtype=int)
    partitions = list(outer.split(features, labels))
    for fold_number, (_, validation_positions) in enumerate(partitions, start=1):
        fold_ids[validation_positions] = fold_number
    for name, specification in specifications.items():
        scores = np.full(len(features), np.nan)
        counts = np.zeros(len(features), dtype=int)
        for fold_number, (training_positions, validation_positions) in enumerate(partitions, start=1):
            started = time.perf_counter()
            search = make_search(specification, seed + 10, inner_folds, trials)
            search.fit(features.iloc[training_positions], labels[training_positions])
            positive_index = int(np.flatnonzero(search.classes_ == 1)[0])
            p = search.predict_proba(features.iloc[validation_positions])[:, positive_index]
            scores[validation_positions] = p
            counts[validation_positions] += 1
            fold_rows.append({
                "model": name, "outer_fold": fold_number, "n_fit": len(training_positions),
                "n_evaluate": len(validation_positions),
                "average_precision": average_precision_score(labels[validation_positions], p),
                "roc_auc": roc_auc_score(labels[validation_positions], p),
                "brier_score": brier_score_loss(labels[validation_positions], p),
                "seconds": time.perf_counter() - started,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            })
            print(f"{name} · outer {fold_number}/{outer_folds} · AP={fold_rows[-1]['average_precision']:.3f}", flush=True)
        if not np.isfinite(scores).all() or not (counts == 1).all():
            raise AssertionError("Setiap baris train harus mendapat tepat satu prediksi outer-fold.")
        oof[name] = scores
    folds = pd.DataFrame(fold_rows)
    leaderboard = folds.groupby("model").agg(
        outer_ap_mean=("average_precision", "mean"), outer_ap_sd=("average_precision", "std"),
        outer_roc_auc_mean=("roc_auc", "mean"), outer_brier_mean=("brier_score", "mean"),
        fit_seconds=("seconds", "sum"),
    ).reset_index().sort_values(
        ["outer_ap_mean", "outer_roc_auc_mean", "model"], ascending=[False, False, True],
    ).reset_index(drop=True)
    return leaderboard, folds, oof, pd.Series(fold_ids, index=features.index, name="outer_fold")


def policy_metrics(labels, probabilities, policy, name, index=None):
    labels, p = validate_label_scores(labels, probabilities)
    decision = apply_policy(p, policy, index)
    tn, fp, fn, tp = confusion_matrix(labels, decision, labels=[0, 1]).ravel()
    return {
        "policy": name, "kind": policy["kind"], "threshold": policy.get("threshold", np.nan),
        "contacts": int(decision.sum()), "contact_rate": decision.mean(),
        "precision": precision_score(labels, decision, zero_division=0),
        "recall": recall_score(labels, decision, zero_division=0),
        "f1": f1_score(labels, decision, zero_division=0),
        "f2": fbeta_score(labels, decision, beta=2, zero_division=0),
        "accuracy": accuracy_score(labels, decision),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
