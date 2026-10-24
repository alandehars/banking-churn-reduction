"""Reference schema and file provenance; no customer records."""
DATA_SOURCE_URL = "https://drive.google.com/file/d/1_mpH_uzTGHaYIF33pz9Is6KW_wj8cv8-/view?usp=sharing"
DATA_DRIVE_ID = DATA_SOURCE_URL.split("/d/", 1)[1].split("/", 1)[0]
REFERENCE_SHA256 = "3996cd1fa372e0db0cd9c0ebac35bbd4e8e3c65fb942bb010c826e7b1eeef0a0"
SOURCE_NOTEBOOK_ID = "1j2HGhm_mdRzkCASodkoyy102L7qOZ_m9"
TARGET = "Exited"
NUMERIC_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]
CATEGORY_COLS = ["Geography", "Gender"]
FEATURES = NUMERIC_COLS + BINARY_COLS + CATEGORY_COLS
NUMERIC_RULES = {
    "CreditScore": {"minimum": 0, "integer": True},
    "Age": {"minimum": 1, "maximum": 120, "integer": True},
    "Tenure": {"minimum": 0, "integer": True},
    "Balance": {"minimum": 0}, "NumOfProducts": {"minimum": 1, "integer": True},
    "EstimatedSalary": {"minimum": 0},
}
