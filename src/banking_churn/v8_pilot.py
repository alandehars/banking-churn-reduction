"""Independent two-arm pilot planning; all effects are assumptions, not churn-model estimates."""
from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

NORMAL = NormalDist()


def _number(value, name, low, high, inclusive=True):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} harus angka, bukan boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} harus angka.") from exc
    valid = low <= value <= high if inclusive else low < value < high
    if not math.isfinite(value) or not valid:
        raise ValueError(f"{name} di luar rentang yang diizinkan.")
    return value


def _integer(value, name, low, high):
    result = _number(value, name, low, high)
    if result != int(result):
        raise ValueError(f"{name} harus bilangan bulat.")
    return int(result)


def wilson_interval(successes, trials, confidence=.95):
    """Wilson score interval, including zero/all events; undefined for no trials."""
    trials = _integer(trials, "trials", 0, 100_000_000)
    successes = _integer(successes, "successes", 0, trials)
    confidence = _number(confidence, "confidence", 0, 1, False)
    if not trials:
        return None, None
    p = successes / trials
    z = NORMAL.inv_cdf((1 + confidence) / 2)
    denominator = 1 + z*z/trials
    centre = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return (0. if successes==0 else max(0., centre-radius),
        1. if successes==trials else min(1., centre+radius))


def approximate_power(control_rate, treatment_rate, per_arm, alpha=.05):
    """Two-sided normal power with Cohen's arcsine h and equal group sizes."""
    p0 = _number(control_rate, "control_rate", 0, 1, False)
    p1 = _number(treatment_rate, "treatment_rate", 0, 1, False)
    n = _integer(per_arm, "per_arm", 2, 10_000_000)
    alpha = _number(alpha, "alpha", .00001, .2)
    h = abs(2*math.asin(math.sqrt(p0)) - 2*math.asin(math.sqrt(p1)))
    shift = h*math.sqrt(n/2)
    critical = NORMAL.inv_cdf(1-alpha/2)
    return NORMAL.cdf(shift-critical) + NORMAL.cdf(-shift-critical)


@dataclass(frozen=True)
class PilotPlan:
    control_rate: float = .20
    absolute_reduction: float = .03
    power: float = .80
    alpha: float = .05
    missing_outcome_fraction: float = .10

    def calculate(self):
        p0 = _number(self.control_rate, "control_rate", 0, 1, False)
        reduction = _number(self.absolute_reduction, "absolute_reduction", 0, p0, False)
        target_power = _number(self.power, "power", .5, .999)
        alpha = _number(self.alpha, "alpha", .00001, .2)
        loss = _number(self.missing_outcome_fraction, "missing_outcome_fraction", 0, .5)
        p1 = p0 - reduction
        low, high = 2, 10_000_000
        if approximate_power(p0, p1, high, alpha) < target_power:
            raise ValueError("Efek terlalu kecil untuk batas perencanaan 10 juta per kelompok.")
        while low < high:
            middle = (low+high)//2
            if approximate_power(p0, p1, middle, alpha) >= target_power:
                high = middle
            else:
                low = middle+1
        enrolled = math.ceil(low/(1-loss))
        return {"control_rate_assumption":p0,"treatment_rate_assumption":p1,
            "absolute_reduction_assumption":reduction,"reduction_percentage_points":100*reduction,
            "relative_reduction_assumption":reduction/p0,"target_power":target_power,"alpha":alpha,
            "analyzable_per_arm":low,"enrolment_per_arm":enrolled,"total_enrolment":2*enrolled,
            "missing_outcome_fraction_assumption":loss,"approximate_power":approximate_power(p0,p1,low,alpha),
            "allocation":"1:1","test":"two_sided","method":"normal_independent_arcsine_h",
            "effect_estimated_from_churn_model":False,"causal_result_observed":False,
            "limitations":["Independent customers, one pre-specified binary outcome and fixed horizon",
                "No clustering, repeated testing or multiple-outcome adjustment",
                "Inflation allows for missing outcomes; it does not remove missingness bias",
                "Normal approximation; confirm the design before starting an experiment"]}


def simulate_pilot(plan, per_arm=None, repetitions=5000, seed=42, null_effect=False):
    """Simulate binomial outcomes and a pooled two-proportion z test at the chosen sample size."""
    spec = plan.calculate()
    n = spec["analyzable_per_arm"] if per_arm is None else _integer(per_arm,"per_arm",2,10_000_000)
    repetitions = _integer(repetitions,"repetitions",100,100_000)
    seed = _integer(seed,"seed",0,2**32-1)
    if not isinstance(null_effect,(bool,np.bool_)):
        raise ValueError("null_effect harus boolean.")
    p0 = spec["control_rate_assumption"]
    p1 = p0 if null_effect else spec["treatment_rate_assumption"]
    rng = np.random.default_rng(seed)
    control = rng.binomial(n,p0,repetitions)
    treatment = rng.binomial(n,p1,repetitions)
    difference = (control-treatment)/n
    pooled = (control+treatment)/(2*n)
    se = np.sqrt(pooled*(1-pooled)*(2/n))
    z = np.divide(difference,se,out=np.zeros_like(difference),where=se>0)
    critical = NORMAL.inv_cdf(1-spec["alpha"]/2)
    significant = abs(z)>critical
    lower,upper = wilson_interval(int(significant.sum()),repetitions)
    summary = {"synthetic_experiments":True,"null_effect_simulation":bool(null_effect),
        "assumed_control_rate":p0,"assumed_treatment_rate":p1,"analyzable_per_arm":n,
        "repetitions":repetitions,"seed":seed,"rejection_rate":float(significant.mean()),
        "monte_carlo_interval_95":[lower,upper],
        "benefit_direction_detection_rate":float(((z>critical)&significant).mean()),
        "observed_reduction_pp_quantiles":(100*np.quantile(difference,[.025,.5,.975])).tolist(),
        "test":"two_sided_pooled_proportions_z","causal_result_observed":False}
    draws = pd.DataFrame({"control_events":control,"treatment_events":treatment,
        "observed_reduction_pp":100*difference,"significant_two_sided":significant})
    return summary,draws
