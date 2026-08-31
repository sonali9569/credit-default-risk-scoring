# Deployment: Render + UptimeRobot

**Live:** https://credit-risk-api-ysb0.onrender.com

Deployed on [Render](https://render.com)'s free tier using native Python (no
Docker required on Render's end, though the repo's `Dockerfile` works too if
preferred), kept warm with a free [UptimeRobot](https://uptimerobot.com) monitor.

## Setting up the Render service

1. Go to [render.com](https://render.com), sign up/log in, **New → Blueprint**.
2. Connect the GitHub repo. Render detects `render.yaml` automatically and
   proposes the `credit-risk-api` web service with the build/start commands and
   free plan already filled in.
3. Click **Apply**. First build takes a few minutes (installing
   pandas/xgboost/scikit-learn etc.).

To use the Dockerfile instead of native Python: switch the runtime to Docker in
the service settings — Render builds the repo-root `Dockerfile` directly, no
config changes needed either way since it already reads Render's `$PORT`.

## Verifying it's live

```bash
curl https://credit-risk-api-ysb0.onrender.com/health

curl -X POST https://credit-risk-api-ysb0.onrender.com/predict -H "Content-Type: application/json" -d '{
  "person_age": 30, "person_income": 60000, "person_home_ownership": "RENT",
  "person_emp_length": 5.0, "loan_intent": "EDUCATION", "loan_grade": "B",
  "loan_amnt": 10000, "loan_int_rate": 11.5, "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6
}'
```

Expect `{"probability_of_default":0.0159,"risk_band":"LOW","decision":"APPROVE"}`.
Also see `/docs` for the live Swagger UI.

The first request after any idle period is slow (~30-60s) — that's the cold-start
behavior the UptimeRobot monitor below fixes.

## Keeping it warm with UptimeRobot

Render's free tier sleeps a service after ~15 minutes with no traffic. A free
UptimeRobot monitor pinging `/health` every 5 minutes (comfortably inside that
window) keeps it from ever sleeping.

1. Go to [uptimerobot.com](https://uptimerobot.com), sign up (free).
2. **Add New Monitor** → Monitor Type: **HTTP(s)**.
3. Friendly Name: `credit-risk-api`.
4. URL: `https://credit-risk-api-ysb0.onrender.com/health`.
5. Monitoring Interval: **5 minutes**.
6. Save.

Pinging every 5 minutes, 24/7, keeps the Render service running essentially the
whole month — roughly 720 of Render's 750 free instance-hours. Still fits inside
the free allowance, just uses nearly all of it for this one service.

## A bug this caught

Right after setting up the monitor, UptimeRobot showed the service as
permanently down — despite Render's dashboard showing "Live" and direct `GET`
requests to `/health` succeeding in under a second. The cause, found via
UptimeRobot's incident detail panel and confirmed with `curl -I`: uptime
monitors check via `HEAD` requests by default, and `/health` only had a `GET`
handler registered, so every check returned a real `405 Method Not Allowed`.
Fixed by adding `HEAD` support alongside `GET` on that route.

## Updating the deployed service

Push to GitHub — Render auto-deploys on push by default (Blueprint-managed
services), or trigger a manual deploy from the dashboard.
