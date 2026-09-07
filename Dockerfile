FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY api/ ./api/
COPY data/processed/ ./data/processed/
COPY mlruns/ ./mlruns/
COPY mlflow.db ./mlflow.db

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
