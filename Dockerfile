FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RETAINIQ_ROOT=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY sql ./sql
COPY data ./data
RUN pip install --no-cache-dir --no-deps . \
    && python -m retainiq.train

# APP=api (default) serves the scoring API; APP=a2a serves the Retention Strategist agent.
ENV APP=api PORT=8080
CMD if [ "$APP" = "a2a" ]; then python -m retainiq.agents.a2a_server; \
    else uvicorn retainiq.api:app --host 0.0.0.0 --port ${PORT}; fi
