import pandas as pd

df = pd.read_csv("data/processed/spurs_matches_features.csv")

def get_result(row):
    if row["spurs_score"] > row["opponent_score"]:
        return "win"
    elif row["spurs_score"] == row["opponent_score"]:
        return "draw"
    else:
        return "loss"

df["result"] = df.apply(get_result, axis=1)

df.to_csv("data/processed/spurs_matches_results.csv", index=False)