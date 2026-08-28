# spurs-match-predictor

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
