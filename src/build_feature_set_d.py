import pandas as pd

df = pd.read_csv("data/processed/spurs_matches_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)  # re-confirm chronological order, never assume it survived a save/reload
df["days_since_last_match"] = df["date"].diff().dt.days

# a handful of gaps are extraordinary stoppages (season boundaries, COVID,
# the 2022 Qatar World Cup break) rather than normal fixture congestion - flag
# them separately and cap the raw value so they don't dominate a linear
# model's coefficient for this feature the way an unbounded 2-104 day range
# would; 30 sits in the real gap between the largest ordinary outlier (22
# days) and the next one up (44, the World Cup restart)
REST_DAYS_CAP = 30
df["is_extended_break"] = (df["days_since_last_match"] > REST_DAYS_CAP).astype(int)
df["days_since_last_match"] = df["days_since_last_match"].clip(upper=REST_DAYS_CAP)
print(f"Flagged {df['is_extended_break'].sum()} matches as following an extended break (>{REST_DAYS_CAP} days)")

WINDOW = 5

# --- rolling form features (last WINDOW matches, all opponents combined) ---
# shift(1) first: moves every value down one row, so the window at row N only ever sees rows before N
# rolling(WINDOW): takes the average over that shifted window
df["rolling_points_5"] = df["points"].shift(1).rolling(WINDOW).mean().round(1)
df["rolling_goals_scored_5"] = df["spurs_score"].shift(1).rolling(WINDOW).mean().round(1)
df["rolling_goals_conceded_5"] = df["opponent_score"].shift(1).rolling(WINDOW).mean().round(1)
df["rolling_goal_diff_5"] = (df["rolling_goals_scored_5"] - df["rolling_goals_conceded_5"]).round(1)

# --- head-to-head features (history against this specific opponent) ---
# for each opponent, keep a running total of points and goal difference from
# matches played BEFORE the current row, so a match's own result never leaks
# into its own h2h features
matches_played = {}
points_total = {}
goal_diff_total = {}

h2h_prior_matches_played = []
h2h_win_rate = []
h2h_goal_diff = []

for _, row in df.iterrows():
    opponent = row["opponent_team"]
    prior_matches = matches_played.get(opponent, 0)

    if prior_matches == 0:
        # first ever meeting with this opponent, no history to draw on yet
        win_rate = None
        goal_diff = None
    else:
        win_rate = round((points_total[opponent] / prior_matches) / 3, 2)  # 0-3 points scale -> 0-1 win-rate-like scale
        goal_diff = round(goal_diff_total[opponent] / prior_matches, 2)

    h2h_prior_matches_played.append(prior_matches)
    h2h_win_rate.append(win_rate)
    h2h_goal_diff.append(goal_diff)

    # update running totals with this match's own result, for the next time Spurs face this opponent
    matches_played[opponent] = prior_matches + 1
    points_total[opponent] = points_total.get(opponent, 0) + row["points"]
    goal_diff_total[opponent] = goal_diff_total.get(opponent, 0) + (row["spurs_score"] - row["opponent_score"])

df["h2h_prior_matches_played"] = h2h_prior_matches_played
df["h2h_win_rate"] = h2h_win_rate
df["h2h_goal_diff"] = h2h_goal_diff

print(f"Rows before dropping NaN warm-up rows: {len(df)}")

# drop rows missing either kind of warm-up history, same treatment for both feature families
# so train.py never has to deal with NaNs
df = df.dropna(
    subset=["rolling_points_5", "rolling_goal_diff_5", "h2h_win_rate", "h2h_goal_diff", "days_since_last_match"]
).reset_index(drop=True)

print(f"Rows after dropping NaN warm-up rows: {len(df)}")
print("Days since last match summary stats:")
print(df["days_since_last_match"].describe())
print(df[["date", "days_since_last_match"]].head(10))

df.to_csv("data/processed/spurs_matches_feature_set_d.csv", index=False)
print(f"Saved feature set D (rolling + h2h + days_since_last_match) for {len(df)} matches across {df['season'].nunique()} seasons")
