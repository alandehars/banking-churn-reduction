"""Preprocessing fitted only inside training/CV pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from .schema import NUMERIC_COLS,BINARY_COLS,CATEGORY_COLS


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), NUMERIC_COLS),
        ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), BINARY_COLS),
        ("category", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORY_COLS),
    ], remainder="drop", verbose_feature_names_out=True)


def make_pipeline(estimator):
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
