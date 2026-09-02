import pandas as pd

df = pd.read_csv("data/processed/spurs_matches_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)  # re-confirm chronological order, never assume it survived a save/reload

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
opponents = df["opponent_team"].tolist()
points_list = df["points"].tolist()
spurs_scores = df["spurs_score"].tolist()
opponent_scores = df["opponent_score"].tolist()

matches_played = {}
points_total = {}
goal_diff_total = {}

h2h_prior_matches_played = []
h2h_win_rate = []
h2h_goal_diff = []

for i in range(len(df)):
    opponent = opponents[i]
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
    points_total[opponent] = points_total.get(opponent, 0) + points_list[i]
    goal_diff_total[opponent] = goal_diff_total.get(opponent, 0) + (spurs_scores[i] - opponent_scores[i])

df["h2h_prior_matches_played"] = h2h_prior_matches_played
df["h2h_win_rate"] = h2h_win_rate
df["h2h_goal_diff"] = h2h_goal_diff

print(f"Rows before dropping NaN warm-up rows: {len(df)}")

# drop rows missing either kind of warm-up history, same treatment for both feature families
# so train.py never has to deal with NaNs
df = df.dropna(
    subset=["rolling_points_5", "rolling_goal_diff_5", "h2h_win_rate", "h2h_goal_diff"]
).reset_index(drop=True)

print(f"Rows after dropping NaN warm-up rows: {len(df)}")

# --- home/away split rolling form ---
# same shift-before-rolling leak prevention as the overall rolling form, but this time
# home matches only look back at prior HOME matches, and away matches only look back at
# prior AWAY matches, so we keep a separate running history list for each venue instead
# of one shared window
venues = df["venue"].tolist()
points_list = df["points"].tolist()
spurs_scores = df["spurs_score"].tolist()
opponent_scores = df["opponent_score"].tolist()

home_points_history = []
home_scored_history = []
home_conceded_history = []
away_points_history = []
away_scored_history = []
away_conceded_history = []


def rolling_average(history):
    # need a full WINDOW of prior matches at this venue before we can average, same
    # requirement as pandas' .rolling(WINDOW) - otherwise there's no history yet
    if len(history) < WINDOW:
        return None
    return round(sum(history[-WINDOW:]) / WINDOW, 2)


rolling_points_5_home = []
rolling_points_5_away = []
rolling_goals_scored_5_home = []
rolling_goals_scored_5_away = []
rolling_goals_conceded_5_home = []
rolling_goals_conceded_5_away = []
rolling_points_5_matching_venue = []
rolling_goals_scored_5_matching_venue = []
rolling_goals_conceded_5_matching_venue = []

for i in range(len(df)):
    if venues[i] == "home":
        points_value = rolling_average(home_points_history)
        scored_value = rolling_average(home_scored_history)
        conceded_value = rolling_average(home_conceded_history)

        rolling_points_5_home.append(points_value)
        rolling_points_5_away.append(None)
        rolling_goals_scored_5_home.append(scored_value)
        rolling_goals_scored_5_away.append(None)
        rolling_goals_conceded_5_home.append(conceded_value)
        rolling_goals_conceded_5_away.append(None)

        # this match becomes part of HOME history for the next home match
        home_points_history.append(points_list[i])
        home_scored_history.append(spurs_scores[i])
        home_conceded_history.append(opponent_scores[i])
    else:
        points_value = rolling_average(away_points_history)
        scored_value = rolling_average(away_scored_history)
        conceded_value = rolling_average(away_conceded_history)

        rolling_points_5_home.append(None)
        rolling_points_5_away.append(points_value)
        rolling_goals_scored_5_home.append(None)
        rolling_goals_scored_5_away.append(scored_value)
        rolling_goals_conceded_5_home.append(None)
        rolling_goals_conceded_5_away.append(conceded_value)

        # this match becomes part of AWAY history for the next away match
        away_points_history.append(points_list[i])
        away_scored_history.append(spurs_scores[i])
        away_conceded_history.append(opponent_scores[i])

    # each row only ever gets a value from ONE of home/away (matching its own venue),
    # so this is just whichever of the two was actually computed above
    rolling_points_5_matching_venue.append(points_value)
    rolling_goals_scored_5_matching_venue.append(scored_value)
    rolling_goals_conceded_5_matching_venue.append(conceded_value)

df["rolling_points_5_home"] = rolling_points_5_home
df["rolling_points_5_away"] = rolling_points_5_away
df["rolling_goals_scored_5_home"] = rolling_goals_scored_5_home
df["rolling_goals_scored_5_away"] = rolling_goals_scored_5_away
df["rolling_goals_conceded_5_home"] = rolling_goals_conceded_5_home
df["rolling_goals_conceded_5_away"] = rolling_goals_conceded_5_away
df["rolling_points_5_matching_venue"] = rolling_points_5_matching_venue
df["rolling_goals_scored_5_matching_venue"] = rolling_goals_scored_5_matching_venue
df["rolling_goals_conceded_5_matching_venue"] = rolling_goals_conceded_5_matching_venue

print(f"Rows before dropping venue-split warm-up rows: {len(df)}")

# same treatment as the earlier dropna: the first WINDOW home matches and first WINDOW
# away matches have no same-venue history yet, so their _matching_venue columns are None
df = df.dropna(
    subset=["rolling_points_5_matching_venue", "rolling_goals_scored_5_matching_venue", "rolling_goals_conceded_5_matching_venue"]
).reset_index(drop=True)

print(f"Rows after dropping venue-split warm-up rows: {len(df)}")

print("Sample of venue-split rolling form:")
print(df[["date", "venue", "opponent_team", "points", "rolling_points_5_home", "rolling_points_5_away", "rolling_points_5_matching_venue"]].head(10))

df.to_csv("data/processed/spurs_matches_feature_set_c.csv", index=False)
print(f"Saved feature set C (rolling + h2h + venue split) for {len(df)} matches across {df['season'].nunique()} seasons")
