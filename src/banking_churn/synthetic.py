"""Deterministic synthetic data for integration tests, never evidence about a real bank."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic_data(rows=1000,seed=42):
    if isinstance(rows,bool) or not isinstance(rows,int) or rows<300:
        raise ValueError("At least 300 synthetic rows are needed for split and calibration tests")
    rng=np.random.default_rng(seed)
    age=rng.integers(18,81,rows)
    active=rng.integers(0,2,rows)
    products=rng.integers(1,4,rows)
    geography=rng.choice(["France","Germany","Spain"],rows)
    balance=np.round(rng.uniform(0,200000,rows),2)
    risk=1/(1+np.exp(-(-4.7+.065*age-.8*active+.5*(products==1)+.5*(geography=="Germany"))))
    labels=rng.binomial(1,risk)
    frame=pd.DataFrame({"RowNumber":np.arange(1,rows+1),"CustomerId":[f"SYNTH-{i:05d}" for i in range(rows)],
        "Surname":["SYNTHETIC"]*rows,"CreditScore":rng.integers(350,851,rows),"Geography":geography,
        "Gender":rng.choice(["Female","Male"],rows),"Age":age,"Tenure":rng.integers(0,11,rows),
        "Balance":balance,"NumOfProducts":products,"HasCrCard":rng.integers(0,2,rows),
        "IsActiveMember":active,"EstimatedSalary":np.round(rng.uniform(100,199000,rows),2),"Exited":labels})
    if min(frame.Exited.value_counts())<30:
        raise ValueError("Synthetic draw has too few examples of one class; change seed or row count")
    return frame


def write_synthetic_csv(path,rows=1000,seed=42):
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError("Synthetic generator does not overwrite an existing file")
    make_synthetic_data(rows,seed).to_csv(target,index=False,lineterminator="\n")
    return target
