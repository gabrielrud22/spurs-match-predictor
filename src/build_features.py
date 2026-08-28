import pandas as pd

df = pd.read_csv("data/processed/spurs_matches.csv")

def get_spurs_score(row):
    return row["score_home"] if "Tottenham Hotspur" in row["home_team"] else row["score_away"]

def get_opponent_score(row):
    return row["score_away"] if "Tottenham Hotspur" in row["home_team"] else row["score_home"]

def get_opponent_team(row):
    return row["away_team"] if "Tottenham Hotspur" in row["home_team"] else row["home_team"]

def get_venue(row):
    return "home" if "Tottenham Hotspur" in row["home_team"] else "away"

def get_result(row):
    if row["spurs_score"] > row["opponent_score"]:
        return "win"
    elif row["spurs_score"] == row["opponent_score"]:
        return "draw"
    else:
        return "loss"

def get_points(row):
    if row["result"] == "win":
        return 3
    elif row["result"] == "draw":
        return 1
    else:
        return 0

df["spurs_score"] = df.apply(get_spurs_score, axis=1)
df["opponent_score"] = df.apply(get_opponent_score, axis=1)
df["opponent_team"] = df.apply(get_opponent_team, axis=1)
df["venue"] = df.apply(get_venue, axis=1)
df["result"] = df.apply(get_result, axis=1)
df["points"] = df.apply(get_points, axis=1)

df.to_csv("data/processed/spurs_matches_features.csv", index=False)
print(f"Derived Spurs-specific information for {len(df)} matches across {df['season'].nunique()} seasons")
print("Computed Win/Draw/Loss result for each game")
print(df["result"].value_counts())
print(f"Total points earned: {df['points'].sum()}")