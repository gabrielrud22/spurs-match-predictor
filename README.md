# spurs-match-predictor

A project to predict the outcome of Spurs (Tottenham Hotspur) matches based on historical Premier League match results.

## Status

Data preparation and feature engineering are done, and baseline models have been trained and tracked in MLflow. The prediction API and deployment are scaffolded but not yet implemented.

- [x] Raw data (Premier League seasons 2010/11 – 2025/26)
- [x] Half-time score imputation ([src/impute_ht_helper.py](src/impute_ht_helper.py))
- [x] Spurs match extraction ([src/pull_data.py](src/pull_data.py))
- [x] Feature engineering ([src/build_features.py](src/build_features.py) + feature sets A/B/C, see below)
- [x] Model training ([src/train.py](src/train.py)) — logistic regression, random forest, and XGBoost baselines tracked in MLflow across feature sets A and B
- [ ] Prediction API ([api/main.py](api/main.py))
- [ ] Containerization & deployment ([Dockerfile](Dockerfile), [k8s/](k8s/))
- [ ] CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))

## Project structure

```
data/
  raw/          Season-by-season Premier League results (JSON, 2010/11-2025/26)
  processed/    Cleaned and derived Spurs match data (see Data section below)
src/
  pull_data.py             Extracts Spurs matches from data/raw into data/processed/spurs_matches.csv
  impute_ht_helper.py      One-off helper: fills missing HT scores with 0-0 when FT was 0-0.
                           Mutates data/raw/*.json in place — already run, don't re-run casually.
  build_features.py        Derives spurs_score/opponent/venue/result/points -> spurs_matches_features.csv
  build_rolling_features.py  Rolling 5-match form only -> spurs_matches_model_ready.csv
  build_h2h_features.py      Head-to-head history only -> spurs_matches_h2h.csv
  build_feature_set_b.py     Feature set B: rolling form + h2h -> spurs_matches_feature_set_b.csv
  build_feature_set_c.py     Feature set C: rolling form + h2h + home/away split -> spurs_matches_feature_set_c.csv
  train.py                 Trains and cross-validates a model, logs params/metrics/artifacts to MLflow
api/
  main.py         Prediction API (not yet implemented)
k8s/              Kubernetes manifests for deployment (not yet implemented)
Dockerfile        Container build (not yet implemented)
mlflow.db, mlruns/  MLflow tracking store and logged run artifacts (models, confusion matrices)
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Each stage reads the previous stage's output, so run them in order:

```bash
uv run src/pull_data.py              # data/raw -> spurs_matches.csv
uv run src/build_features.py         # spurs_matches.csv -> spurs_matches_features.csv
uv run src/build_feature_set_b.py    # spurs_matches_features.csv -> spurs_matches_feature_set_b.csv
uv run src/build_feature_set_c.py    # spurs_matches_features.csv -> spurs_matches_feature_set_c.csv
uv run src/train.py                  # trains a model on the configured feature set, logs to MLflow
```

`build_rolling_features.py` and `build_h2h_features.py` are earlier, narrower stages kept for reference — each also reads `spurs_matches_features.csv` and applies only one of the two feature families. Feature sets B and C build both from scratch independently, so they don't chain off these or off each other.

`train.py` currently points at feature set B; edit `FEATURE_COLUMNS` and the `df = pd.read_csv(...)` path at the top of the file to switch feature sets, and update `RUN_NUMBER`/`RUN_NAME` and the model block before each run (see the checklist in the file's module docstring).

To inspect logged runs (params, metrics, confusion matrices):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Data

Raw match data is Premier League season results in JSON format (one file per season, 2010/11 through 2025/26), each containing full-time and half-time scores for every match.

Processed feature sets, in increasing order of feature richness:

- **Set A** (`spurs_matches_model_ready.csv`): rolling 5-match form (points, goals scored/conceded, goal difference) only.
- **Set B** (`spurs_matches_feature_set_b.csv`): Set A's rolling form plus head-to-head history against the specific opponent (prior matches played, win rate, goal difference).
- **Set C** (`spurs_matches_feature_set_c.csv`): Set B plus rolling form split by venue (home-only and away-only 5-match windows, unified into a single `_matching_venue` column per stat).

Each feature set drops its own warm-up rows (matches without enough prior history to compute its features), so row counts shrink slightly from Set A to C.
