from fastapi import FastAPI, HTTPException #imports the core FastAPI class used to create the app instance
from pydantic import BaseModel, Field #imports Pydantic's BaseModel class to define the shape of incoming request data
import pandas as pd #standard dataframe library, used here to load and query historical match data
import mlflow.sklearn #imports the MLflow module for loading trained model from MLflow registry

app = FastAPI() #creates the application instance that uvicorn actually runs

# loads the registered CHAMPION model; line runs once when the module is first imported
# model = mlflow.sklearn.load_model("models:/spurs-match-predictor@champion")
model = mlflow.sklearn.load_model("mlruns/1/models/m-df7fdda84f314f29a4a9ca66d6763ea3/artifacts")
#loads the trained model directly from its MLflow artifact folder

#also runs once at start, loads raw match history, converts into proper datetime type, and sorts chronologically
history = pd.read_csv("data/processed/spurs_matches_features.csv")
history["date"] = pd.to_datetime(history["date"])
history = history.sort_values("date").reset_index(drop=True)

WINDOW = 5 #matches the rolling window size used during training. Must stay consistent with whatever the champion model was trained on

#health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

#defines what a valid /predict request body must contain, three string fields
class FixtureRequest(BaseModel):
    opponent: str = Field(..., example="Liverpool FC")
    date: str = Field(..., example="YYYY-MM-DD")
    # venue: str = Field(..., example="home")  # "home" or "away", not currently used by Set B, kept for future extension

KNOWN_OPPONENTS = set(history["opponent_team"].unique())

def build_features_for_fixture(opponent: str, fixture_date: pd.Timestamp) -> pd.DataFrame:
    if opponent not in KNOWN_OPPONENTS:
        raise HTTPException(status_code=400, detail=f"{opponent} is not a recognised team. Known opponents: {', '.join(sorted(KNOWN_OPPONENTS))}")
        # raise ValueError(f"{opponent} is not a recognised team. Known opponents: {', '.join(sorted(KNOWN_OPPONENTS))}")
    # only matches strictly BEFORE the fixture date, same leak-prevention rule as training
    prior_matches = history[history["date"] < fixture_date]

    if len(prior_matches) < WINDOW: #if there aren't 5 prior matches to average over, the function can't compute and therefore fails
        # raise ValueError("Not enough match history before this date to compute rolling form")
        raise HTTPException(status_code=400, detail="Not enough match history before this date to compute rolling form")

    recent = prior_matches.tail(WINDOW) #takes the last 5 rows of prior matches to give the exact 5 matches before this fixture

    #calculates the four rolling form values used in the original build_rolling_features.py  
    rolling_points_5 = round(recent["points"].mean(), 1)
    rolling_goals_scored_5 = round(recent["spurs_score"].mean(), 1)
    rolling_goals_conceded_5 = round(recent["opponent_score"].mean(), 1)
    rolling_goal_diff_5 = round(rolling_goals_scored_5 - rolling_goals_conceded_5, 1)

    #filters prior matches more granularly to only those against this specific opponent
    h2h_matches = prior_matches[prior_matches["opponent_team"] == opponent]
    h2h_prior_matches_played = len(h2h_matches)

    #if there are no prior matches, default h2h values to 0.0
    if h2h_prior_matches_played == 0:
        h2h_win_rate = 0.0  # no history, use a neutral default rather than crashing
        h2h_goal_diff = 0.0
    else:
        h2h_win_rate = round((h2h_matches["points"].sum() / h2h_prior_matches_played) / 3, 2)
        h2h_goal_diff = round((h2h_matches["spurs_score"] - h2h_matches["opponent_score"]).mean(), 2)

    #package all calculated values into a single-row DataFrame
    return pd.DataFrame([{
        "rolling_points_5": rolling_points_5,
        "rolling_goals_scored_5": rolling_goals_scored_5,
        "rolling_goals_conceded_5": rolling_goals_conceded_5,
        "rolling_goal_diff_5": rolling_goal_diff_5,
        "h2h_win_rate": h2h_win_rate,
        "h2h_goal_diff": h2h_goal_diff,
        "h2h_prior_matches_played": h2h_prior_matches_played,
    }])

#the predict endpoint, FastAPI parses the incoming JSON into a FixtureRequest object, which is then used to compute features and make a prediction
@app.post("/predict")
def predict(fixture: FixtureRequest):
    fixture_date = pd.to_datetime(fixture.date)
    features = build_features_for_fixture(fixture.opponent, fixture_date)

    DISPLAY_ORDER = ["win", "draw", "loss"]

    probabilities = model.predict_proba(features)[0] #runs the model, returning a probability for every class, for every row passed in
    classes = model.classes_ #retrieves the actual class labels learning during training
    prob_by_class = dict(zip(classes, probabilities))

    ordered_probabilities = {label: f"{round(float(prob_by_class[label]) * 100, 1)}%" for label in DISPLAY_ORDER}

    #builds the JSON response, so each class label is paired with its corresponding probability 
    return { 
        "opponent": fixture.opponent,
        "date": fixture.date,
        # "venue": fixture.venue,
        "probabilities": ordered_probabilities
    }

# in api/main.py
@app.get("/opponents")
def get_opponents():
    return {"opponents": sorted(KNOWN_OPPONENTS)}