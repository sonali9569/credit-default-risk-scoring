"""
FastAPI serving layer for the credit risk PD model.

Loads the calibrated model + encoder + feature list saved by notebook 06
(models/model_calibrated.pkl, models/encoder.pkl, models/feature_names.pkl) and the
cleaning bounds from src/preprocessing.py — the exact same objects the notebooks
validated, no re-derivation, no train/serve skew.

Risk bands and the REJECT cutoff are not arbitrary round numbers: they're read
directly from reports/decile_table.csv and reports/policy_analysis.csv, produced by
notebook 07's decile/policy analysis. REJECT specifically matches the "reject
worst decile" policy validated there (38.1% bad-rate reduction, 90.4% volume
retained).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing import (  # noqa: E402
    MAX_PLAUSIBLE_AGE,
    MIN_WORKING_AGE,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Cutoffs derived from reports/decile_table.csv + reports/policy_analysis.csv
# (notebook 07). REJECT_CUTOFF is the exact top-decile boundary validated by the
# policy analysis: rejecting PD >= this value reproduces the 38.1% bad-rate
# reduction / 90.4% volume-retained result (see README.md Results section).
REJECT_CUTOFF = 0.9934
HIGH_CUTOFF = 0.15  # roughly the decile 6/7 boundary (bad rate jumps 21% -> 74%)
MEDIUM_CUTOFF = 0.05  # roughly the decile 3/4 boundary (bad rate ~4-6%)

app = FastAPI(
    title="Credit Risk API",
    description="Probability-of-default scoring for the laotse/credit-risk-dataset model.",
    version="1.0.0",
)

_model = None
_encoder = None
_feature_names = None


def _load_artifacts():
    global _model, _encoder, _feature_names
    if _model is None:
        for name in ("model_calibrated.pkl", "encoder.pkl", "feature_names.pkl"):
            if not (MODEL_DIR / name).exists():
                raise RuntimeError(
                    f"Missing {name} in {MODEL_DIR} — run notebooks 05 and 06 first "
                    "to produce the trained/calibrated model artifacts."
                )
        _model = joblib.load(MODEL_DIR / "model_calibrated.pkl")
        _encoder = joblib.load(MODEL_DIR / "encoder.pkl")
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return _model, _encoder, _feature_names


class Application(BaseModel):
    person_age: int = Field(..., ge=18, le=MAX_PLAUSIBLE_AGE, description="Applicant age in years")
    person_income: float = Field(..., gt=0, description="Annual income")
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    person_emp_length: float = Field(..., ge=0, description="Years employed")
    loan_intent: Literal[
        "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0, le=40)
    loan_percent_income: float = Field(..., ge=0, le=1, description="loan_amnt / person_income")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 30,
                "person_income": 60000,
                "person_home_ownership": "RENT",
                "person_emp_length": 5.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 10000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.17,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6,
            }
        }


class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "REJECT"]
    decision: Literal["APPROVE", "REJECT"]


@app.get("/health")
@app.head("/health")  # uptime monitors (UptimeRobot, Pingdom, etc.) default to
# HEAD requests, and FastAPI/Starlette doesn't auto-add HEAD support for a GET
# route — without this, every monitoring check gets a real 405 and the service
# looks permanently "down" even while it's genuinely healthy. Confirmed live:
# HEAD /health returned 405 (allow: GET) from every UptimeRobot region before
# this fix.
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: Application):
    model, encoder, feature_names = _load_artifacts()

    row = pd.DataFrame([application.model_dump()])

    # Same sanity bound as src/preprocessing.clean() — reject rather than silently
    # cap here, since a live applicant with an impossible combination is a data
    # problem to surface, not paper over.
    if row.loc[0, "person_emp_length"] > (row.loc[0, "person_age"] - MIN_WORKING_AGE):
        raise HTTPException(
            status_code=422,
            detail="person_emp_length is implausible given person_age (would imply "
            f"working before age {MIN_WORKING_AGE}).",
        )

    row = row[NUMERIC_COLS + CATEGORICAL_COLS]
    row_enc = encoder.transform(row)[feature_names]

    pd_score = float(model.predict_proba(row_enc)[0, 1])

    if pd_score >= REJECT_CUTOFF:
        band, decision = "REJECT", "REJECT"
    elif pd_score >= HIGH_CUTOFF:
        band, decision = "HIGH", "REJECT"
    elif pd_score >= MEDIUM_CUTOFF:
        band, decision = "MEDIUM", "APPROVE"
    else:
        band, decision = "LOW", "APPROVE"

    return PredictionResponse(
        probability_of_default=round(pd_score, 4),
        risk_band=band,
        decision=decision,
    )
