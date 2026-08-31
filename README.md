# Credit Risk Default Prediction

A calibrated, explainable probability-of-default (PD) model for loan applications,
built on the [Credit Risk Dataset (laotse)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
from Kaggle. The project covers the full pipeline — EDA, feature analysis,
imbalance handling, hyperparameter tuning, probability calibration, SHAP
explainability, and a live deployed API — with an emphasis on metrics that
actually matter for imbalanced classification, not just leaderboard AUC.

**Live API:** https://credit-risk-api-ysb0.onrender.com
([`/docs`](https://credit-risk-api-ysb0.onrender.com/docs) for the interactive
Swagger UI — try it directly in the browser, no setup needed.)

## Results

| Model | AUC | KS | Brier |
|---|---|---|---|
| Logistic regression baseline | 0.870 | 0.604 | 0.138 |
| WOE-encoded logistic regression | 0.885 | 0.653 | — |
| XGBoost, default params (`scale_pos_weight`) | 0.946 | 0.756 | 0.068 |
| XGBoost, Optuna-tuned (80 trials) | 0.948 | 0.757 | 0.062 |
| **XGBoost, tuned + isotonic-calibrated (final)** | **0.948** | **0.756** | **0.051** |

- **Ablation** (model retrained without `loan_grade`/`loan_int_rate`): AUC 0.909 —
  confirms real predictive signal in applicant-level attributes alone, independent
  of the lender's own risk grade.
- **IV analysis:** 4 of 11 features exceed IV 0.5 (legitimate, underwriting-time
  information — cross-checked against the ablation model, not leakage). Full table:
  [`reports/iv_table.csv`](reports/iv_table.csv).
- **SHAP:** top drivers are `person_income`, `loan_int_rate`, `loan_percent_income` —
  consistent with the IV ranking and the ablation finding.
- **Business impact:** rejecting just the worst-scored decile cuts the approved
  population's bad rate from 21.9% to 13.5% (**38.1% reduction**) while retaining
  **90.4%** of applicants. See [`reports/policy_analysis.csv`](reports/policy_analysis.csv).

## Documentation

- [`docs/analysis_and_modeling_plan.md`](docs/analysis_and_modeling_plan.md) —
  the reasoning behind every modeling decision: why each metric, why XGBoost, why
  the imbalance technique was tested rather than assumed, why the ablation model,
  why calibration, why SHAP.
- [`docs/deployment_render.md`](docs/deployment_render.md) — how the API is
  deployed and kept alive.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Raw data isn't committed to this repo (not ours to redistribute).

1. Go to https://www.kaggle.com/datasets/laotse/credit-risk-dataset
   (sign in with your Kaggle account if prompted).
2. Download **`credit_risk_dataset.csv`**.
3. Place it at:
   ```
   data/raw/credit_risk_dataset.csv
   ```
4. Verify:
   ```bash
   python3 -c "import pandas as pd; df = pd.read_csv('data/raw/credit_risk_dataset.csv'); print(df.shape); print(df['loan_status'].value_counts(normalize=True))"
   ```
   Expect `(32581, 12)` and a `loan_status` split of roughly 78% / 22%.

12 raw columns: `person_age`, `person_income`, `person_home_ownership`,
`person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`,
`loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`,
`loan_status` (target). Mix of numeric and categorical columns, with real missingness
in `person_emp_length` (~2.7%) and `loan_int_rate` (~9.6%).

## Repo layout

```
data/raw/          raw CSV (gitignored)
data/processed/    cleaned/engineered datasets, train/val/test splits (gitignored)
notebooks/         numbered notebooks, one per pipeline stage, run in order
src/               reusable functions shared across notebooks and the API (cleaning, metrics)
models/            trained model + encoder artifacts used by the API
reports/           figures, tables, decile analysis, SHAP plots
app/               FastAPI serving code
Dockerfile         container build (works locally or on Render)
render.yaml         Render Blueprint config
```

## Pipeline

1. **EDA & preprocessing** (`notebooks/01_eda_and_preprocessing.ipynb`) — cleaning,
   missing-value handling, 60/20/20 stratified split.
2. **Baseline** (`notebooks/02_baseline_model.ipynb`) — logistic regression, AUC 0.870.
3. **XGBoost** (`notebooks/03_xgboost_imbalance_comparison.ipynb`) — `scale_pos_weight`
   vs. SMOTE comparison, plus the `loan_grade` ablation model. AUC 0.946.
4. **WOE/IV analysis** (`notebooks/04_woe_iv_analysis.ipynb`) — full 11-feature IV
   table, WOE-encoded logistic regression.
5. **Hyperparameter tuning** (`notebooks/05_optuna_tuning.ipynb`) — 80-trial Optuna
   study, parallelized. AUC 0.948.
6. **Calibration** (`notebooks/06_calibration.ipynb`) — isotonic regression on a
   held-out validation split. Brier 0.062 → 0.051.
7. **Explainability & policy** (`notebooks/07_shap_and_policy.ipynb`) — SHAP
   analysis, decile table, lending-policy trade-off.
8. **Serving** (`app/`, `Dockerfile`) — FastAPI service, deployed to Render.

See [`docs/analysis_and_modeling_plan.md`](docs/analysis_and_modeling_plan.md) for
why each of these steps happened, not just what they did.

To re-run the notebooks yourself (each writes files the next one reads):

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

## How to run the API locally

```bash
source .venv/bin/activate
uvicorn app.app:app --reload --port 8000
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

## How to get a prediction

**Against the live deployment** (no setup required):

```bash
curl -X POST https://credit-risk-api-ysb0.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "cb_person_cred_hist_length": 6
  }'
```

Response:
```json
{"probability_of_default": 0.0159, "risk_band": "LOW", "decision": "APPROVE"}
```

`risk_band` is one of `LOW` / `MEDIUM` / `HIGH` / `REJECT`, with cutoffs derived
directly from the decile analysis in `reports/decile_table.csv` — not arbitrary
round numbers. Field constraints (valid categories, ranges) are documented at
`/docs` and enforced by the API — invalid input returns a `422` with the reason.

**Locally**, same request against `http://localhost:8000/predict` once the server
from the previous section is running.

## Deployment

Deployed on [Render](https://render.com)'s free tier with a
[UptimeRobot](https://uptimerobot.com) monitor keeping it warm. Full details,
including a real bug that came up during deployment (uptime monitors check via
`HEAD` requests, which the `/health` endpoint didn't originally support) and how
it was fixed, are in [`docs/deployment_render.md`](docs/deployment_render.md).

`render.yaml` lets Render auto-configure the service from this repo; the
`Dockerfile` at the repo root is an equally valid alternative if deploying via
Docker instead of Render's native Python build.
