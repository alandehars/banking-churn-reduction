"""Deterministic synthetic fixtures; no real customers, CSV, training or persisted model."""
import ast
import csv
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd


class SyntheticEstimator:
    """Fixed formula for testing inference plumbing, not a learned churn model."""
    classes_=np.array([0,1])

    def predict_proba(self, frame):
        score=np.clip(.15+.003*(frame.Age.fillna(45)-18)
            +.000001*frame.Balance.fillna(50000)+.05*frame.IsActiveMember.fillna(0),.01,.99)
        return np.column_stack([1-score,score])


def make_synthetic_bundle(notebook):
    sources={c["id"]:"".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"}
    env={"np":np,"pd":pd,"math":math,"warnings":warnings,"csv":csv,"Path":Path,"tempfile":tempfile}
    allowed={"TARGET","NUMERIC_COLS","BINARY_COLS","CATEGORY_COLS","FEATURES","NUMERIC_RULES"}
    for node in ast.parse(sources["configuration"]).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in allowed for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),"notebook:feature-schema","exec"),env)
    exec(compile(sources["core_helpers"],"notebook:core_helpers","exec"),env)
    rng=np.random.default_rng(42)
    rows=200
    frame=pd.DataFrame({"CreditScore":rng.integers(450,850,rows),"Age":rng.integers(20,76,rows),
        "Tenure":rng.integers(0,11,rows),"Balance":rng.uniform(0,200000,rows),
        "NumOfProducts":rng.integers(1,5,rows),"EstimatedSalary":rng.uniform(1000,190000,rows),
        "HasCrCard":np.tile([1,1,1,0],50),"IsActiveMember":rng.integers(0,2,rows),
        "Geography":rng.choice(["France","Germany","Spain"],rows),"Gender":rng.choice(["Female","Male"],rows)})
    frame=frame.loc[:,env["FEATURES"]]
    meta={"revision":6,"run_id":"synthetic-fixture","features":env["FEATURES"],
        "numeric_features":env["NUMERIC_COLS"],"binary_features":env["BINARY_COLS"],
        "categorical_features":env["CATEGORY_COLS"],"numeric_rules":env["NUMERIC_RULES"],
        "training_categories":{c:sorted(frame[c].unique().tolist()) for c in env["CATEGORY_COLS"]},
        "training_numeric_ranges":{c:{"minimum":float(frame[c].min()),"maximum":float(frame[c].max())} for c in env["NUMERIC_COLS"]},
        "positive_class":1,"default_policy":"balanced_f1","policies":{"balanced_f1":{"kind":"threshold","threshold":.3}}}
    return {"metadata":meta,"estimator":SyntheticEstimator()},frame,SimpleNamespace(predict_customers=env["predict_customers"])
