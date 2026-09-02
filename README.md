# spurs-match-predictor

## TODO: README updates needed

- [ ] `main.py` deleted (unused `uv init` placeholder) — remove it from the project structure diagram if it's still listed elsewhere, and confirm `pyproject.toml` doesn't reference it as an entry point.
- [ ] Status checklist is stale: feature engineering and model training are implemented and have real runs behind them (11 MLflow runs across Set A/B, logreg/random forest/XGBoost) — update those two checkboxes and describe what's actually done vs. still pending (API, containerization, CI are still genuinely empty scaffolding).
- [ ] Project structure section lists `src/features.py` (doesn't exist) — replace with the real pipeline: `build_features.py`, `build_rolling_features.py`, `build_h2h_features.py`, `build_feature_set_b.py`, `build_feature_set_c.py`, `train.py`.
- [ ] Explain the feature-set naming scheme (A/B/C) — what each one adds (rolling form / +h2h / +venue split) and which processed CSV corresponds to which.
- [ ] Usage section only documents `pull_data.py` — add run instructions for the rest of the pipeline (`build_features.py` → `build_rolling_features.py`/`build_h2h_features.py` → `build_feature_set_b.py`/`build_feature_set_c.py` → `train.py`), including that they must run in order since each reads the previous stage's output.
- [ ] Document how to inspect training runs (`mlflow ui`, which backing store/db it uses) since `train.py` now logs real experiments.
- [ ] Decide and document whether `mlruns/` and `mlflow.db` are meant to be committed to git — flag this as an open question rather than silently updating, since it changes repo conventions.
- [ ] `impute_ht_helper.py` is described as a "one-off helper" already run — note that it mutates `data/raw/*.json` in place, so it should not be re-run casually.

A project to predict the outcome of Spurs (Tottenham Hotspur) matches based on historical Premier League match results.

## Status

Currently in the data-preparation stage. The pipeline pulls raw Premier League results, imputes missing half-time scores, and extracts Spurs-specific matches into a clean CSV. Feature engineering, model training, the prediction API, and deployment are scaffolded but not yet implemented.

- [x] Raw data (Premier League seasons 2010/11 – 2025/26)
- [x] Half-time score imputation ([src/impute_ht_helper.py](src/impute_ht_helper.py))
- [x] Spurs match extraction ([src/pull_data.py](src/pull_data.py))
- [ ] Feature engineering ([src/features.py](src/features.py))
- [ ] Model training ([src/train.py](src/train.py))
- [ ] Prediction API ([api/main.py](api/main.py))
- [ ] Containerization & deployment ([Dockerfile](Dockerfile), [k8s/](k8s/))
- [ ] CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))

## Project structure

```
data/
  raw/          Season-by-season Premier League results (JSON, 2010/11-2025/26)
  processed/    spurs_matches.csv - cleaned, Spurs-only match data
src/
  pull_data.py          Extracts Spurs matches from data/raw into data/processed/spurs_matches.csv
  impute_ht_helper.py   One-off helper: fills missing HT scores with 0-0 when FT was 0-0
  features.py           Feature engineering (not yet implemented)
  train.py              Model training (not yet implemented)
api/
  main.py         Prediction API (not yet implemented)
k8s/              Kubernetes manifests for deployment (not yet implemented)
Dockerfile        Container build (not yet implemented)
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Regenerate the cleaned Spurs match dataset from the raw season files:

```bash
uv run src/pull_data.py
```

This reads every `data/raw/*.json` file, filters for matches involving Tottenham Hotspur, and writes the result to `data/processed/spurs_matches.csv`, sorted chronologically.

## Data

Raw match data is Premier League season results in JSON format (one file per season, 2010/11 through 2025/26), each containing full-time and half-time scores for every match.
