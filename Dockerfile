# Built from the repo root (not app/) because the app imports src/preprocessing.py
# and loads models/ — both live outside app/. Build with:
#   docker build -t credit-risk-api .
# Run locally with:
#   docker run -p 8000:8000 credit-risk-api
# On Render (or any platform that assigns its own port via $PORT), no extra flags
# needed — the CMD below reads $PORT automatically and falls back to 8000 when
# it isn't set (i.e. running locally).
FROM python:3.11-slim

WORKDIR /code

# Copy requirements first and install before copying the rest of the code, so
# code-only edits don't invalidate this (slow) layer on rebuild.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# --host 0.0.0.0 is required: 127.0.0.1 inside a container is unreachable from
# outside it — the classic beginner bug this project's plan specifically calls out.
# Shell form (not exec-array form) so $PORT actually gets expanded by the shell.
CMD uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}
