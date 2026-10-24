"""V6 decision, uncertainty, batch-quality and live-report helpers, embedded in Colab."""
import copy
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def v6_number(value, name, low=0, high=None, integer=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} harus berupa angka, bukan boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} harus berupa angka.") from exc
    if not np.isfinite(number) or number < low or (high is not None and number > high):
        raise ValueError(f"{name} berada di luar rentang yang diizinkan.")
    if integer and not number.is_integer():
        raise ValueError(f"{name} harus bilangan bulat.")
    return int(number) if integer else number


def v6_fingerprint(index):
    values = [str(x) for x in index]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def v6_json(value):
    if isinstance(value, dict):
        return {str(k): v6_json(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return v6_json(value.to_dict(orient="records"))
    if isinstance(value, (list, tuple, np.ndarray)):
        return [v6_json(x) for x in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Nilai tidak dapat diekspor: {type(value).__name__}")


def v6_order(probabilities, index=None):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Skor harus vektor finite, tidak kosong, antara 0 dan 1.")
    keys = np.arange(len(p)) if index is None else np.asarray(index)
    if keys.ndim != 1 or len(keys) != len(p) or pd.isna(keys).any():
        raise ValueError("Index tidak sesuai skor.")
    keys = np.array([str(x) for x in keys])
    if not pd.Index(keys).is_unique:
        raise ValueError("Index harus unik setelah konversi teks.")
    return p, np.lexsort((keys, -p))


def v6_policy_advisor(labels, probabilities, locked_policies, max_contacts, budget,
                      contact_cost, value_per_saved, success_rate, fixed_cost=0,
                      min_precision=0, index=None):
    """Choose a conditional scenario using validation only; never updates the model policy."""
    p, order = v6_order(probabilities, index)
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Label validation harus lengkap, 0/1, dan sesuai skor.")
    maximum = v6_number(max_contacts, "Batas kontak", high=len(p), integer=True)
    budget = v6_number(budget, "Anggaran")
    cost = v6_number(contact_cost, "Biaya kontak")
    benefit = v6_number(value_per_saved, "Manfaat")
    success = v6_number(success_rate, "Keberhasilan", high=1)
    fixed = v6_number(fixed_cost, "Biaya tetap")
    minimum = v6_number(min_precision, "Minimum precision", high=1)
    inputs = {"max_contacts": maximum, "budget_um": budget, "contact_cost_um": cost,
              "value_per_saved_um": benefit, "success_rate_assumption": success,
              "fixed_cost_um": fixed, "min_validation_precision": minimum}

    cumulative_risk = np.r_[0., np.cumsum(p[order])]
    cumulative_tp = np.r_[0, np.cumsum(y[order])]
    k = np.arange(len(p) + 1)
    total_cost = k * cost + np.where(k > 0, fixed, 0)
    net = cumulative_risk * success * benefit - total_cost
    precision = np.divide(cumulative_tp, k, out=np.zeros(len(k)), where=k > 0)
    eligible = (k > 0) & (k <= maximum) & (total_cost <= budget + 1e-10) & (precision >= minimum)
    curve = pd.DataFrame({"contacts": k, "risk_sum": cumulative_risk, "tp": cumulative_tp,
        "validation_precision": precision, "cost_um": total_cost, "expected_net_um": net,
        "eligible": eligible})
    feasible_k = k[eligible]
    best_k = int(feasible_k[np.argmax(net[feasible_k])]) if len(feasible_k) else 0
    resource_k = int(k[(k <= maximum) & (total_cost <= budget + 1e-10)].max())
    decisions = {"no_campaign": np.zeros(len(p), dtype=int)}
    for name, policy in locked_policies.items():
        if policy.get("kind") == "threshold":
            threshold = v6_number(policy.get("threshold"), "Threshold", high=1)
            decisions[name] = (p >= threshold).astype(int)
        elif policy.get("kind") == "top_k":
            share = v6_number(policy.get("fraction"), "Fraksi", high=1)
            decision = np.zeros(len(p), dtype=int)
            decision[order[:math.ceil(share * len(p))]] = 1
            decisions[name] = decision
        else:
            raise ValueError("Jenis kebijakan terkunci tidak dikenal.")
    for name, count in [("top_k_resource_limit", resource_k), ("top_k_conditional_value", best_k)]:
        decision = np.zeros(len(p), dtype=int)
        decision[order[:count]] = 1
        decisions[name] = decision
    rows = []
    for name, decision in decisions.items():
        selected = decision.astype(bool)
        contacts = int(selected.sum())
        tp = int(y[selected].sum())
        precision_value = tp / contacts if contacts else None
        bill = contacts * cost + (fixed if contacts else 0)
        risk_sum = float(p[selected].sum())
        resources_ok = contacts <= maximum and bill <= budget + 1e-10
        quality_ok = bool(contacts and precision_value >= minimum)
        rows.append({"policy": name, "contacts": contacts, "tp": tp, "fp": contacts - tp,
            "fn": int(y.sum()) - tp, "validation_precision": precision_value,
            "validation_recall": tp / int(y.sum()) if y.sum() else None,
            "risk_sum": risk_sum, "expected_saved": risk_sum * success, "cost_um": bill,
            "expected_net_um": risk_sum * success * benefit - bill,
            "resource_feasible": resources_ok, "quality_feasible": quality_ok,
            "eligible_campaign": resources_ok and quality_ok})
    candidates = pd.DataFrame(rows)
    positive = candidates.loc[candidates.eligible_campaign & (candidates.expected_net_um > 1e-10)]
    if positive.empty:
        chosen = candidates.loc[candidates.policy == "no_campaign"].iloc[0]
        status = "no_positive_feasible_campaign"
    else:
        chosen = positive.sort_values(["expected_net_um", "contacts", "policy"],
                                     ascending=[False, True, True]).iloc[0]
        status = "conditional_candidate_found"
    return {"inputs": inputs, "status": status, "recommended": v6_json(chosen.to_dict()),
            "candidates": candidates, "top_k_curve": curve,
            "data_context": "validation", "business_inputs_are_assumptions": True,
            "not_a_causal_effect": True, "changes_final_policy": False}


def v6_range(value, name, high=None):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} memerlukan batas bawah dan atas.")
    lo, hi = (v6_number(x, name, high=high) for x in value)
    if lo > hi:
        raise ValueError(f"Batas bawah {name} melebihi batas atas.")
    return lo, hi


def v6_uncertain_campaign(probabilities, contacts, success_range, cost_range, benefit_range,
                          fixed_cost=0, n_draws=2000, seed=42, index=None):
    """Uniform parameter assumptions plus independent outcomes conditional on each draw."""
    p, order = v6_order(probabilities, index)
    count = v6_number(contacts, "Kontak", high=len(p), integer=True)
    draws = v6_number(n_draws, "Draw", low=100, high=10000, integer=True)
    random_seed = v6_number(seed, "Seed", high=2**32 - 1, integer=True)
    if count * draws > 50_000_000:
        raise ValueError("Kontak × draw melebihi 50 juta; kurangi salah satunya.")
    u_range = v6_range(success_range, "Keberhasilan", high=1)
    c_range = v6_range(cost_range, "Biaya")
    b_range = v6_range(benefit_range, "Manfaat")
    fixed = v6_number(fixed_cost, "Biaya tetap")
    rng = np.random.default_rng(random_seed)
    u = rng.uniform(*u_range, size=draws)
    cost = rng.uniform(*c_range, size=draws)
    benefit = rng.uniform(*b_range, size=draws)
    selected_p = p[order[:count]]
    risk_sum = float(selected_p.sum())
    total_cost = count * cost + (fixed if count else 0)
    expected_net = risk_sum * u * benefit - total_cost
    saved = np.zeros(draws, dtype=int)
    conditional_variance = np.zeros(draws)
    if count:
        for start in range(0, draws, 128):
            end = min(draws, start + 128)
            q = u[start:end, None] * selected_p[None, :]
            saved[start:end] = (rng.random(q.shape) < q).sum(axis=1)
            conditional_variance[start:end] = benefit[start:end]**2 * (q * (1 - q)).sum(axis=1)
    realized_net = saved * benefit - total_cost
    probability = float((realized_net > 0).mean())
    assumption_sd = float(expected_net.std(ddof=0))
    conditional_sd = float(np.sqrt(conditional_variance.mean()))
    summary = {"contacts": count, "risk_sum": risk_sum, "draws": draws, "seed": random_seed,
        "mean_assumed_expected_net_um": float(expected_net.mean()),
        "assumed_expected_net_q025_um": float(np.quantile(expected_net, .025)),
        "assumed_expected_net_q975_um": float(np.quantile(expected_net, .975)),
        "outcome_net_q025_um": float(np.quantile(realized_net, .025)),
        "outcome_net_q975_um": float(np.quantile(realized_net, .975)),
        "profit_fraction_under_assumptions": probability,
        "profit_fraction_mc_se": float(np.sqrt(probability * (1 - probability) / draws)),
        "parameter_expected_net_sd_um": assumption_sd,
        "conditional_outcome_rms_sd_um": conditional_sd,
        "combined_sd_from_variance_decomposition_um": float(np.hypot(assumption_sd, conditional_sd))}
    table = pd.DataFrame({"success_assumption": u, "cost_assumption_um": cost,
        "benefit_assumption_um": benefit, "scenario_expected_net_um": expected_net,
        "saved_draw": saved, "outcome_net_um": realized_net})
    return {"summary": summary, "draws": table, "assumptions": {
        "success_uniform": list(u_range), "cost_uniform_um": list(c_range),
        "benefit_uniform_um": list(b_range), "fixed_cost_um": fixed,
        "parameters_independent": True, "outcomes_independent_given_parameters": True,
        "parameter_ranges_are_user_assumptions": True,
        "model_calibration_and_population_shift_uncertainty_included": False,
        "baseline_risk_approximates_no_campaign": True},
        "data_context": "validation", "not_a_causal_effect": True}


def v6_total_variation(reference, incoming):
    a, b = np.asarray(reference, dtype=float), np.asarray(incoming, dtype=float)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Distribusi tidak sesuai.")
    if (a < 0).any() or (b < 0).any() or a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Frekuensi distribusi harus nonnegatif dan berjumlah positif.")
    return float(.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def v6_distribution_counts(series, bins=None):
    if bins is None:
        # Tagged tuples keep a real string '__MISSING__' distinct from a missing value.
        keys = series.map(lambda x: ("missing", "") if pd.isna(x) else ("value", str(x)))
        return keys.value_counts().to_dict()
    numeric = pd.to_numeric(series, errors="raise")
    return np.r_[np.histogram(numeric.dropna().to_numpy(), bins=bins)[0], numeric.isna().sum()]


def v6_batch_quality(reference, incoming, model_bundle, predict_function,
                     reference_probabilities, distribution_alert=.15, data_context="external_csv"):
    alert = v6_number(distribution_alert, "Batas perubahan distribusi", high=1)
    meta = model_bundle["metadata"]
    features = meta["features"]
    invalid = []
    if not isinstance(incoming, pd.DataFrame) or incoming.empty:
        invalid.append("Input harus DataFrame dengan minimal satu baris.")
    elif incoming.columns.duplicated().any():
        invalid.append("Header duplikat.")
    else:
        missing = sorted(set(features) - set(incoming.columns))
        extra = sorted(set(incoming.columns) - set(features) - {"CustomerId", "Exited"})
        if missing:
            invalid.append(f"Kolom wajib hilang: {missing}")
        if extra:
            invalid.append(f"Kolom tambahan: {extra}")
        if not incoming.index.is_unique:
            invalid.append("Index baris harus unik.")
        if "CustomerId" in incoming and (incoming.CustomerId.isna().any() or incoming.CustomerId.duplicated().any()):
            invalid.append("CustomerId harus lengkap dan unik.")
        if "Exited" in incoming:
            label = pd.to_numeric(incoming.Exited, errors="coerce")
            if label.isna().any() or not label.isin([0, 1]).all():
                invalid.append("Label opsional Exited harus lengkap dan 0/1.")
    base = {"data_context": data_context, "distribution_alert_assumption": alert,
            "reference_features": "training", "reference_scores": "validation",
            "threshold_is_illustrative_not_statistical_significance": True,
            "model_version": meta.get("revision"), "model_run_id": meta.get("run_id"),
            "rows": len(incoming) if isinstance(incoming, pd.DataFrame) else 0}
    scores = pd.DataFrame(columns=["estimated_churn_probability", "flag_for_review", "rank_within_batch", "policy", "decision_threshold"])
    if not invalid:
        try:
            incoming = incoming.copy()
            if "CustomerId" in incoming:
                incoming.index = pd.Index(incoming.CustomerId.astype(str), name="CustomerId")
            for name in meta["numeric_features"] + meta["binary_features"]:
                incoming[name] = pd.to_numeric(incoming[name], errors="raise").astype(float)
            for name in meta["categorical_features"]:
                incoming[name] = incoming[name].map(lambda x: str(x).strip() if pd.notna(x) else np.nan).replace("", np.nan)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores = predict_function(incoming.loc[:, features], model_bundle)
            base["prediction_warnings"] = [str(w.message) for w in caught]
        except (TypeError, ValueError) as exc:
            invalid.append(str(exc))
    if invalid:
        return {"summary": {**base, "status": "blocked_invalid_schema", "errors": invalid,
                "scored_rows": 0}, "features": pd.DataFrame(), "scores": scores.iloc[:0].copy()}

    rows = []
    for name in features:
        ref, new = reference[name], incoming[name]
        is_numeric = name in meta["numeric_features"] or name in meta["binary_features"]
        if is_numeric:
            ref_num = pd.to_numeric(ref, errors="raise")
            new_num = pd.to_numeric(new, errors="raise")
            unique = np.sort(ref_num.dropna().unique())
            if len(unique) <= 20:
                # Quantile bins can merge both binary values into one bucket.
                # Midpoints preserve every observed value for discrete numeric features.
                cuts = unique[:-1] + (unique[1:] - unique[:-1]) / 2
            else:
                cuts = np.unique(np.quantile(ref_num.dropna(), np.linspace(0, 1, 11)))[1:-1]
            bins = np.r_[-np.inf, cuts, np.inf]
            a, b = v6_distribution_counts(ref_num, bins), v6_distribution_counts(new_num, bins)
            distance = v6_total_variation(a, b)
            outside = float(((new_num < ref_num.min()) | (new_num > ref_num.max())).mean())
            unseen = 0.
        else:
            a, b = v6_distribution_counts(ref), v6_distribution_counts(new)
            keys = set(a) | set(b)
            distance = v6_total_variation([a.get(k, 0) for k in keys], [b.get(k, 0) for k in keys])
            unseen = float((new.notna() & ~new.isin(ref.dropna().unique())).mean())
            outside = 0.
        reasons = []
        if new.isna().mean() > ref.isna().mean() + 1e-12:
            reasons.append("missing_bertambah")
        if unseen:
            reasons.append("kategori_baru")
        if outside:
            reasons.append("di_luar_rentang_train")
        if distance > alert:
            reasons.append("distribusi_berubah")
        rows.append({"feature": name, "reference_missing_rate": float(ref.isna().mean()),
            "incoming_missing_rate": float(new.isna().mean()), "unseen_category_rate": unseen,
            "out_of_train_range_rate": outside, "total_variation_distance": distance,
            "flagged": bool(reasons), "reasons": "; ".join(reasons)})
    table = pd.DataFrame(rows)
    ref_p, _ = v6_order(reference_probabilities)
    new_p = scores.estimated_churn_probability.to_numpy()
    score_shift = v6_total_variation(np.histogram(ref_p, np.linspace(0, 1, 11))[0],
                                    np.histogram(new_p, np.linspace(0, 1, 11))[0])
    summary = {**base, "status": "review_needed" if table.flagged.any() or score_shift > alert else "no_alert_at_selected_threshold",
        "errors": [], "scored_rows": len(scores), "flagged_features": table.loc[table.flagged, "feature"].tolist(),
        "score_distribution_tv": score_shift, "mean_estimated_risk": float(new_p.mean()),
        "flagged_for_review": int(scores.flag_for_review.sum()),
        "accuracy_not_estimated_without_labels": "Exited" not in incoming,
        "distribution_alert_proves_performance_drop": False}
    if "Exited" in incoming:
        y = pd.to_numeric(incoming.Exited).to_numpy()
        pred = scores.flag_for_review.to_numpy()
        tp = int(((pred == 1) & (y == 1)).sum())
        summary["label_audit"] = {"positive_labels": int(y.sum()), "true_positive": tp,
            "precision": tp / int(pred.sum()) if pred.sum() else None,
            "recall": tp / int(y.sum()) if y.sum() else None,
            "is_external_validation": data_context == "external_csv",
            "labels_used_to_refit_or_select_policy": False}
    return {"summary": summary, "features": table, "scores": scores}


def v6_compare_versions(baseline, current):
    fields = ["data_sha256", "split_fingerprints", "features", "seed", "default_policy", "capacity_fraction"]
    differences = [k for k in fields if baseline.get(k) != current.get(k)]
    versions_differ = baseline.get("versions") != current.get("versions")
    rows = []
    for name, old_value in baseline["metrics"].items():
        value = current["metrics"].get(name)
        comparable = not differences and old_value is not None and value is not None
        rows.append({"metric": name, "v5_reference": old_value, "v6_current": value,
                     "delta_v6_minus_v5": value - old_value if comparable else None,
                     "data_and_protocol_match": not differences})
    return {"table": pd.DataFrame(rows), "summary": {"baseline_run_id": baseline["run_id"],
        "current_run_id": current["run_id"], "baseline_environment": baseline.get("environment"),
        "current_environment": current.get("environment"), "protocol_mismatches": differences,
        "package_versions_differ": versions_differ,
        "timing_is_hardware_dependent": True,
        "comparison_is_descriptive_not_independent_validation": True,
        "selection_uses_this_comparison": False}}


def v6_test_narrative(context):
    d, m = context["default_metrics"], context["test_scores"]
    return (
        f"# Interpretasi otomatis · run {context['run_id']}\n\n"
        f"Model **{context['model']}**, dipilih pada train, memberi ROC-AUC **{m['roc_auc']:.3f}** "
        f"dan AP **{m['average_precision']:.3f}** pada test berisi {m['test_rows']:,} baris "
        f"dengan {m['test_churn']:,} label churn. Prevalensi test {m['test_prevalence']:.2%}.\n\n"
        f"Default **{context['default_policy']}**, threshold **{context['threshold']:.6f}** dari validation: "
        f"{int(d['contacts']):,} prioritas ({d['contact_rate']:.2%}), {int(d['tp'])} churn terdeteksi, "
        f"{int(d['fp'])} alarm keliru, dan {int(d['fn'])} kasus terlewat. "
        f"Precision **{d['precision']:.1%}**, recall **{d['recall']:.1%}**.\n\n"
        "ROC-AUC menjelaskan ranking; AP merangkum precision–recall. Keduanya berbeda dari accuracy "
        "dan precision pada satu threshold. Flag merupakan prioritas peninjauan; deteksi label historis "
        "belum mengukur pelanggan yang terselamatkan. Cocokkan jumlah prioritas dengan kapasitas dan "
        "biaya operasional melalui validation. Test lama ini tetap audit internal yang telah dilihat "
        "selama pengembangan; validasi pada data baru dan pilot terkontrol masih diperlukan.\n\n"
        f"Sumber SHA-256: `{context['data_sha256']}`. Angka dihitung dari run ini.\n"
    )


def v6_refresh_report(directory, context):
    directory = Path(directory)
    sections = {}
    for name in ["advisor", "uncertainty", "comparison", "batch"]:
        path = directory / f"v6_{name}_latest.json"
        if path.is_file():
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["run_id"] != context["run_id"] or item["source_sha256"] != context["data_sha256"]:
                raise ValueError("Snapshot berasal dari run/sumber lain.")
            sections[name] = item
    report = v6_test_narrative(context)
    if "advisor" in sections:
        a = sections["advisor"]["summary"]
        best = a["recommended"]
        report += "\n## Keputusan skenario terbaru · validation\n\n"
        if a["status"] == "conditional_candidate_found":
            report += (f"Kandidat bersyarat: **{best['policy']}**, {best['contacts']} kontak, "
                       f"biaya {best['cost_um']:,.2f} UM, expected net {best['expected_net_um']:,.2f} UM. ")
        else:
            report += "**Tidak ada kampanye bernilai positif yang memenuhi batas input.** Pilihan skenario: tidak menjalankan kampanye. "
        report += "Keputusan memakai asumsi biaya, manfaat, dan keberhasilan; tidak mengganti kebijakan audit model.\n"
    if "uncertainty" in sections:
        s = sections["uncertainty"]["summary"]["results"]
        report += (f"\n## Ketidakpastian skenario terbaru\n\nDengan {s['contacts']} kontak tetap, rentang kuantil "
                   f"outcome net 2,5%–97,5% adalah {s['outcome_net_q025_um']:,.2f} sampai "
                   f"{s['outcome_net_q975_um']:,.2f} UM; proporsi draw positif "
                   f"{s['profit_fraction_under_assumptions']:.1%}. Rentang parameter adalah asumsi pengguna, "
                   "sehingga proporsi ini bukan peluang keuntungan yang terverifikasi dari kampanye nyata.\n")
    if "comparison" in sections:
        s = sections["comparison"]["summary"]
        report += (f"\n## Pembanding V5\n\nBaseline {s['baseline_run_id']}; perbedaan protokol: "
                   f"{s['protocol_mismatches'] or 'tidak ada'}. Paket berbeda: {s['package_versions_differ']}. "
                   "Tabel perbandingan bersifat deskriptif; pemilihan model tetap pada train.\n")
    if "batch" in sections:
        s = sections["batch"]["summary"]
        report += (f"\n## Pemeriksaan batch terbaru\n\nKonteks **{s['data_context']}**, status **{s['status']}**, "
                   f"{s['scored_rows']} baris berhasil diskor. ")
        if s.get("flagged_features"):
            report += "Perlu tinjauan: " + ", ".join(s["flagged_features"]) + ". "
        if s.get("errors"):
            report += "Kesalahan input: " + "; ".join(s["errors"]) + ". "
        report += "Perubahan distribusi tidak langsung membuktikan penurunan akurasi.\n"
    report += ("\n## Tindakan berikutnya\n\nKonfirmasi horizon churn dan satuan manfaat; kumpulkan data baru "
               "yang sesuai; uji penawaran terhadap kelompok kontrol sebelum memperluas program. "
               "Contoh validation dan data stres sintetis di notebook berfungsi sebagai demonstrasi.\n")
    payload = v6_json({"run_id": context["run_id"], "source_sha256": context["data_sha256"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "model_context": context,
        "latest_sections": sections})
    (directory / "current_report.md").write_text(report, encoding="utf-8")
    (directory / "current_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def v6_save_section(kind, inputs, summary, directory, context, tables=None):
    if kind not in {"advisor", "uncertainty", "comparison", "batch"}:
        raise ValueError("Bagian V6 tidak dikenal.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = v6_json({"kind": kind, "run_id": context["run_id"],
        "source_sha256": context["data_sha256"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "summary": summary})
    target = directory / f"v6_{kind}_latest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    for name, table in (tables or {}).items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Nama tabel ekspor tidak valid.")
        table.to_csv(directory / f"v6_{kind}_{name}_latest.csv", index=True if name == "scores" else False)
    report = v6_refresh_report(directory, context)
    base = directory.parent.parent
    manifest_path = base / "latest_run.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") == context["run_id"] and manifest.get("status") == "complete":
            import shutil
            shutil.make_archive(str(base / context["run_id"]), "zip", root_dir=directory)
    return target, report
