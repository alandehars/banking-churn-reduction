"""Shared scenario calculations from the validated V6 lab."""
import html
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, clear_output, display
from matplotlib.ticker import FuncFormatter
SIM_COLORS = {'navy': '#18324D', 'teal': '#138B8D', 'amber': '#C97A28', 'red': '#BD4852'}

def _sim_scalar(value, name, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} harus angka, bukan boolean.')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} harus angka yang valid.') from exc
    if not np.isfinite(number) or (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        raise ValueError(f'{name} di luar rentang yang diizinkan.')
    return number

def _sim_probabilities(probabilities):
    result = np.asarray(probabilities, dtype=float)
    if result.ndim != 1 or result.size == 0 or (not np.isfinite(result).all()) or ((result < 0) | (result > 1)).any():
        raise ValueError('Skor simulasi harus vektor probabilitas finite antara 0 dan 1.')
    return result

def _sim_portfolio(labels, probabilities):
    score = _sim_probabilities(probabilities)
    target = np.asarray(labels)
    if target.ndim != 1 or len(target) != len(score) or (not np.isin(target, [0, 1]).all()):
        raise ValueError('Label simulasi harus vektor 0/1 dengan panjang yang sama seperti skor.')
    return (target.astype(int), score)

def _sim_order(score, index=None):
    keys = np.arange(len(score)) if index is None else np.asarray(index)
    if keys.ndim != 1 or len(keys) != len(score) or pd.isna(keys).any():
        raise ValueError('Index simulasi harus lengkap dan sesuai jumlah skor.')
    text_keys = np.asarray([str(key) for key in keys])
    if not pd.Index(text_keys).is_unique:
        raise ValueError('Index simulasi harus unik setelah diubah menjadi teks.')
    return (np.lexsort((text_keys, -score)), keys)

def _sim_counts(target, decision):
    tp = int(((target == 1) & (decision == 1)).sum())
    fp = int(((target == 0) & (decision == 1)).sum())
    fn = int(((target == 1) & (decision == 0)).sum())
    tn = int(((target == 0) & (decision == 0)).sum())
    contacts, positives = (tp + fp, tp + fn)
    denominator = 2 * tp + fp + fn
    return {'rows': len(target), 'contacts': contacts, 'contact_rate': contacts / len(target), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': tp / contacts if contacts else np.nan, 'recall': tp / positives if positives else np.nan, 'f1': 2 * tp / denominator if denominator else np.nan, 'accuracy': (tp + tn) / len(target), 'prevalence': positives / len(target)}

def profile_what_if(baseline, changes, model_bundle, threshold, predictor):
    cutoff = _sim_scalar(threshold, 'threshold', 0, 1)
    if not isinstance(baseline, pd.DataFrame) or len(baseline) != 1 or (not isinstance(changes, dict)):
        raise ValueError('Baseline harus DataFrame satu baris dan changes harus dictionary.')
    unknown = set(changes) - set(baseline.columns)
    if unknown:
        raise ValueError(f'Nama fitur skenario tidak tersedia: {sorted(unknown)}')
    scenario = baseline.copy(deep=True)
    for name, value in changes.items():
        scenario[name] = value
    combined = pd.concat([baseline, scenario], ignore_index=True)
    combined.index = pd.Index(['Baseline', 'Skenario'], name='profile')
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        prediction = predictor(combined, model_bundle)
    score = _sim_probabilities(prediction['estimated_churn_probability'].to_numpy())
    if len(score) != 2:
        raise ValueError('Predictor harus mengembalikan satu skor untuk setiap profil.')
    equal = baseline.eq(scenario).iloc[0] | (baseline.isna() & scenario.isna()).iloc[0]
    changed = pd.DataFrame({'baseline': baseline.iloc[0], 'scenario': scenario.iloc[0]}).loc[~equal]
    comparison = pd.DataFrame({'estimated_risk': score, 'threshold': cutoff, 'flag_for_review': (score >= cutoff).astype(int)}, index=combined.index)
    return {'comparison': comparison, 'feature_changes': changed, 'risk_change_pp': float((score[1] - score[0]) * 100), 'warnings': [str(item.message) for item in recorded], 'scenario_input': scenario}

def threshold_scenario(labels, probabilities, threshold):
    target, score = _sim_portfolio(labels, probabilities)
    cutoff = _sim_scalar(threshold, 'threshold', 0, 1)
    decision = (score >= cutoff).astype(int)
    return {'metrics': dict(_sim_counts(target, decision), threshold=cutoff), 'decision': decision}

def capacity_scenario(labels, probabilities, fraction, index=None):
    target, score = _sim_portfolio(labels, probabilities)
    share = _sim_scalar(fraction, 'fraction', 0, 1)
    order, keys = _sim_order(score, index)
    k = min(len(score), math.ceil(share * len(score)))
    chosen = order[:k]
    decision = np.zeros(len(score), dtype=int)
    decision[chosen] = 1
    metrics = _sim_counts(target, decision)
    metrics.update({'capacity_fraction': share, 'predicted_churn_sum': float(score[chosen].sum()), 'lift': metrics['precision'] / metrics['prevalence'] if k and metrics['prevalence'] else np.nan})
    ranked = pd.DataFrame({'rank': np.arange(1, len(score) + 1), 'sample_row': keys[order], 'estimated_risk': score[order], 'historical_label': target[order], 'selected': np.arange(len(score)) < k})
    return {'metrics': metrics, 'decision': decision, 'ranking': ranked, 'selected_positions': chosen}

def _campaign_components(probabilities, fraction, contact_cost, value_per_saved, success_rate, fixed_cost=0.0, index=None):
    """Shared analytic calculation; comparison tables do not need Monte Carlo draws."""
    score = _sim_probabilities(probabilities)
    share = _sim_scalar(fraction, 'fraction', 0, 1)
    cost = _sim_scalar(contact_cost, 'contact_cost', 0)
    benefit = _sim_scalar(value_per_saved, 'value_per_saved', 0)
    success = _sim_scalar(success_rate, 'success_rate', 0, 1)
    fixed = _sim_scalar(fixed_cost, 'fixed_cost', 0)
    order, _ = _sim_order(score, index)
    k = min(len(score), math.ceil(share * len(score)))
    selected = order[:k]
    risk_sum = float(score[selected].sum())
    launched_fixed = fixed if k else 0.0
    total_cost = k * cost + launched_fixed
    expected_saved = risk_sum * success
    expected_gross = expected_saved * benefit
    expected_net = expected_gross - total_cost
    success_probability = score[selected] * success
    break_even = total_cost / (benefit * risk_sum) if benefit > 0 and risk_sum > 0 else np.nan
    summary = {'contacts': k, 'risk_sum': risk_sum, 'success_rate_assumption': success, 'expected_saved': expected_saved, 'expected_gross_um': expected_gross, 'total_cost_um': total_cost, 'expected_net_um': expected_net, 'expected_roi': expected_net / total_cost if total_cost > 0 else np.nan, 'break_even_success_rate': break_even, 'break_even_feasible': bool(np.isfinite(break_even) and break_even <= 1)}
    return {'summary': summary, 'selected_positions': selected, 'success_probability': success_probability, 'assumptions': {'fraction': share, 'contact_cost_um': cost, 'value_per_saved_um': benefit, 'success_rate': success, 'fixed_cost_um': fixed, 'fixed_cost_charged_only_if_contacts': True, 'baseline_risk_approximates_no_campaign': True, 'independent_outcomes': True}}

def retention_expectation(probabilities, fraction, contact_cost, value_per_saved, success_rate, fixed_cost=0.0, index=None):
    """Deterministic expected outcome under explicitly supplied business assumptions."""
    result = _campaign_components(probabilities, fraction, contact_cost, value_per_saved, success_rate, fixed_cost, index)
    return {'summary': result['summary'], 'assumptions': result['assumptions'], 'selected_positions': result['selected_positions']}

def retention_scenario(probabilities, fraction, contact_cost, value_per_saved, success_rate, fixed_cost=0.0, n_draws=2000, seed=42, index=None):
    result = _campaign_components(probabilities, fraction, contact_cost, value_per_saved, success_rate, fixed_cost, index)
    draws_number = _sim_scalar(n_draws, 'n_draws', 1, 20000)
    random_seed = _sim_scalar(seed, 'seed', 0, 2 ** 32 - 1)
    if not draws_number.is_integer() or not random_seed.is_integer():
        raise ValueError('n_draws dan seed harus bilangan bulat.')
    summary = result['summary']
    k, total_cost = (summary['contacts'], summary['total_cost_um'])
    benefit = result['assumptions']['value_per_saved_um']
    success_probability = result['success_probability']
    random = np.random.default_rng(int(random_seed))
    saved = np.zeros(int(draws_number), dtype=int)
    if k:
        for start in range(0, len(saved), 256):
            end = min(start + 256, len(saved))
            saved[start:end] = (random.random((end - start, k)) < success_probability).sum(axis=1)
    net = saved * benefit - total_cost
    probability_profit = float(np.mean(net > 0))
    expected_variance = float(np.sum(success_probability * (1 - success_probability)))
    summary.update({'profit_probability_mc': probability_profit, 'profit_probability_mc_se': float(np.sqrt(probability_profit * (1 - probability_profit) / len(net))), 'net_q025_um': float(np.quantile(net, 0.025)), 'net_q50_um': float(np.quantile(net, 0.5)), 'net_q975_um': float(np.quantile(net, 0.975)), 'saved_mean_mc': float(saved.mean()), 'saved_mean_mc_se_theoretical': float(np.sqrt(expected_variance / len(saved))), 'n_draws': len(saved), 'seed': int(random_seed)})
    return {'summary': summary, 'draws': pd.DataFrame({'saved_customers': saved, 'net_benefit_um': net}), 'selected_positions': result['selected_positions'], 'assumptions': result['assumptions']}
