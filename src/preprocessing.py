"""
Data cleaning for the laotse/credit-risk-dataset (credit_risk_dataset.csv).

Every decision here is grounded in EDA findings documented in
notebooks/01_eda_and_preprocessing.ipynb — nothing here is arbitrary. Kept as a
module (not inline notebook code) so the exact same cleaning logic can be reused by
the training pipeline and the FastAPI serving code, avoiding train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "loan_status"

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

# EDA-derived sanity bounds.
# person_age: dataset contains a handful of physically implausible values (94, 123,
# 144) — clear data-entry errors, not a coded sentinel like Home Credit's
# DAYS_EMPLOYED=365243, since there's no repeated single value. Capping is the
# defensible choice; capping is documented and small enough to matter little.
MAX_PLAUSIBLE_AGE = 80

# person_emp_length: max observed is 123 years, and two rows have emp_length >
# person_age, which is impossible (can't have worked longer than you've been alive).
# Cap employment length by age minus 14 (a reasonable minimum legal working age)
# rather than dropping — keeps the row's other information usable.
MIN_WORKING_AGE = 14


def load_raw(path: str = "data/raw/credit_risk_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning steps. Returns a new, cleaned DataFrame.

    Steps (each justified in notebooks/01_eda_and_preprocessing.ipynb):
    1. Drop exact duplicate rows (165 found in EDA) — same applicant/loan record
       appearing more than once inflates whatever split it lands in.
    2. Cap implausible person_age values (>80) at the 80th-percentile-safe bound —
       6 rows had ages of 94-144, physically impossible.
    3. Fix person_emp_length values that exceed what's physically possible given
       person_age (emp_length > age - 14) — 2 rows had emp_length=123 with age
       21-22. Capped at (age - MIN_WORKING_AGE), floored at 0.
    4. Leave missingness in person_emp_length (~2.7%) and loan_int_rate (~9.6%)
       for the imputation step (imputers must be fit on train only — see notebook).
    """
    out = df.copy()

    n_before = len(out)
    out = out.drop_duplicates()
    n_dupes = n_before - len(out)

    out.loc[out["person_age"] > MAX_PLAUSIBLE_AGE, "person_age"] = np.nan
    out["person_age"] = out["person_age"].fillna(out["person_age"].median())

    impossible_emp = out["person_emp_length"] > (out["person_age"] - MIN_WORKING_AGE)
    out.loc[impossible_emp, "person_emp_length"] = (
        out.loc[impossible_emp, "person_age"] - MIN_WORKING_AGE
    ).clip(lower=0)

    out.attrs["n_duplicates_dropped"] = n_dupes
    out.attrs["n_age_capped"] = int((df["person_age"] > MAX_PLAUSIBLE_AGE).sum())
    out.attrs["n_emp_length_fixed"] = int(impossible_emp.sum())

    return out


def get_feature_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
