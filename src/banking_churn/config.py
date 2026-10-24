"""Explicit training configuration with bounded runtime and no mutable module settings."""
from dataclasses import asdict,dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    seed: int=42
    n_jobs: int=2
    outer_folds: int=3
    inner_folds: int=3
    search_trials: int=6
    calibrate: bool=True
    default_policy: str="balanced_f1"
    capacity_fraction: float=.2
    min_validation_precision: float|None=None
    max_validation_contact_rate: float|None=None
    bootstrap_resamples: int=500
    smoke: bool=False

    def validate(self):
        for name,low,high in [("seed",0,2**32-1),("n_jobs",1,8),("outer_folds",2,10),
                              ("inner_folds",2,10),("search_trials",1,100),("bootstrap_resamples",20,2000)]:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
                raise ValueError(f"Invalid {name}")
        if self.default_policy not in {"threshold_0_5","precision_f05","balanced_f1","recall_f2"}:
            raise ValueError("Unsupported default policy")
        if not math.isfinite(self.capacity_fraction) or not 0<self.capacity_fraction<=1:
            raise ValueError("Invalid capacity fraction")
        for value in [self.min_validation_precision,self.max_validation_contact_rate]:
            if value is not None and (not math.isfinite(value) or not 0<=value<=1):
                raise ValueError("Validation constraints must be None or within [0,1]")
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def quick(cls,seed=42):
        return cls(seed=seed,outer_folds=2,inner_folds=2,search_trials=1,bootstrap_resamples=30,smoke=True)
