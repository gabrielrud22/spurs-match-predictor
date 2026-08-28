import pandas as pd

df = pd.read_csv("data/processed/spurs_matches_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)  # re-confirm chronological order, never assume it survived a save/reload

WINDOW = 5

# shift(1) first: moves every value down one row, so the window at row N only ever sees rows before N
# rolling(WINDOW): takes the average over that shifted window
df["rolling_points_5"] = df["points"].shift(1).rolling(WINDOW).mean().round(1)
df["rolling_goals_scored_5"] = df["spurs_score"].shift(1).rolling(WINDOW).mean().round(1)
df["rolling_goals_conceded_5"] = df["opponent_score"].shift(1).rolling(WINDOW).mean().round(1)
df["rolling_goal_diff_5"] = (df["rolling_goals_scored_5"] - df["rolling_goals_conceded_5"]).round(1)

print(f"Rows before dropping NaN warm-up rows: {len(df)}")
print(df[["date", "opponent_team", "points", "rolling_points_5", "rolling_goal_diff_5"]].head(10))

# first WINDOW rows of the entire dataset have no prior matches to average, they'll be NaN, drop them
df = df.dropna(subset=["rolling_points_5", "rolling_goal_diff_5"]).reset_index(drop=True)

print(f"Rows after dropping NaN warm-up rows: {len(df)}")

df.to_csv("data/processed/spurs_matches_model_ready.csv", index=False)
print("Saved model-ready dataset with rolling form features")