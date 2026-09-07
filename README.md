# spurs-match-predictor

A project to predict the outcome of Spurs (Tottenham Hotspur) matches based on historical Premier League match results.

## Status

Data preparation, feature engineering, and model training are done. A FastAPI prediction service is built, containerized, pushed to DockerHub, and deployed to a local Kubernetes cluster. CI is the only piece not yet started.

- [x] Raw data (Premier League seasons 2010/11 – 2025/26)
- [x] Half-time score imputation ([src/impute_ht_helper.py](src/impute_ht_helper.py))
- [x] Spurs match extraction ([src/pull_data.py](src/pull_data.py))
- [x] Feature engineering ([src/build_features.py](src/build_features.py) + feature sets A/B/C/D, see below)
- [x] Model training ([src/train.py](src/train.py)) — 15 runs across logistic regression, random forest, and XGBoost, tracked in MLflow across feature sets A–D (see `training_diary.md`)
- [x] Prediction API ([api/main.py](api/main.py)) — FastAPI app serving win/draw/loss probabilities for a given opponent and date
- [x] Containerization & deployment ([Dockerfile](Dockerfile), [k8s/](k8s/)) — built, pushed to DockerHub, deployed to a local minikube cluster
- [ ] CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))

## Project structure

```
data/
  raw/          Season-by-season Premier League results (JSON, 2010/11-2025/26)
  processed/    Cleaned and derived Spurs match data (see Data section below)
src/
  pull_data.py               Extracts Spurs matches from data/raw into data/processed/spurs_matches.csv
  impute_ht_helper.py        One-off helper: fills missing HT scores with 0-0 when FT was 0-0.
                             Mutates data/raw/*.json in place — already run, don't re-run casually.
  build_features.py          Derives spurs_score/opponent/venue/result/points -> spurs_matches_features.csv
  build_rolling_features.py  Rolling 5-match form only -> spurs_matches_model_ready.csv
  build_h2h_features.py      Head-to-head history only -> spurs_matches_h2h.csv
  build_feature_set_b.py     Feature set B: rolling form + h2h -> spurs_matches_feature_set_b.csv
  build_feature_set_c.py     Feature set C: rolling form + h2h + home/away split -> spurs_matches_feature_set_c.csv
  build_feature_set_d.py     Feature set D: rolling form + h2h + rest days -> spurs_matches_feature_set_d.csv
  visualize_rest_days.py     Prints matches ordered by days since last match and plots the gap over time
  train.py                   Trains and cross-validates a model, logs params/metrics/artifacts to MLflow
api/
  main.py           FastAPI app — /health and /predict (opponent + date -> win/draw/loss probabilities),
                     loads the trained model directly from its MLflow artifact folder
k8s/
  deployment.yaml   1 replica, readiness/liveness probes against /health
  service.yaml      NodePort service exposing the deployment on a stable address
Dockerfile          Builds the API image (uv-managed deps, copies in api/, processed data, and mlruns/)
.dockerignore       Excludes .venv, raw data, and other build-context clutter from the Docker image
mlflow.db, mlruns/  MLflow tracking store and logged run artifacts (models, confusion matrices)
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Data pipeline & training

Each stage reads the previous stage's output, so run them in order:

```bash
uv run src/pull_data.py              # data/raw -> spurs_matches.csv
uv run src/build_features.py         # spurs_matches.csv -> spurs_matches_features.csv
uv run src/build_feature_set_b.py    # spurs_matches_features.csv -> spurs_matches_feature_set_b.csv
uv run src/build_feature_set_c.py    # spurs_matches_features.csv -> spurs_matches_feature_set_c.csv
uv run src/build_feature_set_d.py    # spurs_matches_features.csv -> spurs_matches_feature_set_d.csv
uv run src/train.py                  # trains a model on the configured feature set, logs to MLflow
```

`build_rolling_features.py` and `build_h2h_features.py` are earlier, narrower stages kept for reference — each also reads `spurs_matches_features.csv` and applies only one of the two feature families. Feature sets B, C, and D each build their features from scratch independently, so they don't chain off these or off each other.

`train.py` currently points at feature set D; edit `FEATURE_COLUMNS` and the `df = pd.read_csv(...)` path at the top of the file to switch feature sets, and update `RUN_NUMBER`/`RUN_NAME` and the model block before each run (see the checklist in the file's module docstring).

To inspect logged runs (params, metrics, confusion matrices):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Running the API locally

```bash
uv run uvicorn api.main:app --reload
```

Then `GET /health` for a liveness check, or `POST /predict` with a JSON body like `{"opponent": "Liverpool FC", "date": "2026-10-01"}` to get win/draw/loss probabilities.

### Docker

```bash
docker build -t <image-name> .
docker run -p 8000:8000 <image-name>
```

The image is built from a slim Python base, installs dependencies with `uv`, and copies in `api/`, `data/processed/`, and the MLflow `mlruns/`/`mlflow.db` needed to load the trained model at runtime.

### Kubernetes

Requires [minikube](https://minikube.sigs.k8s.io/docs/start/) and `kubectl`.

```bash
minikube start
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service spurs-match-predictor-service --url
```

`kubectl get pods` should show the pod as `1/1 Running`; the deployment's readiness/liveness probes hit `/health`.

## Data

Raw match data is Premier League season results in JSON format (one file per season, 2010/11 through 2025/26), each containing full-time and half-time scores for every match.

Processed feature sets, in increasing order of feature richness:

- **Set A** (`spurs_matches_model_ready.csv`): rolling 5-match form (points, goals scored/conceded, goal difference) only.
- **Set B** (`spurs_matches_feature_set_b.csv`): Set A's rolling form plus head-to-head history against the specific opponent (prior matches played, win rate, goal difference).
- **Set C** (`spurs_matches_feature_set_c.csv`): Set B plus rolling form split by venue (home-only and away-only 5-match windows, unified into a single `_matching_venue` column per stat).
- **Set D** (`spurs_matches_feature_set_d.csv`): Set B plus `days_since_last_match` (capped at 30 days) and `is_extended_break` (a flag for gaps beyond that — season boundaries, COVID, the 2022 World Cup break).

Each feature set drops its own warm-up rows (matches without enough prior history to compute its features), so row counts shrink slightly from Set A onward. Per `training_diary.md` and `feature_rankings.md`, head-to-head history (Set B) is the one addition that produced a real, measured improvement — venue split (Set C) and rest days (Set D) have both come back flat or negative in matched comparisons; Set B's balanced logistic regression (run 6, F1 macro 0.415) remains the best result logged.
