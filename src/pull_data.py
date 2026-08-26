import json
import glob
import pandas as pd

records = []
for filepath in glob.glob("data/raw/*.json"):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    season = data.get("name", "")

    for match in data.get("matches", []):
        team1 = match.get("team1", "")
        team2 = match.get("team2", "")

        if "Tottenham Hotspur" in (team1, team2):
            ft = match.get("score", {}).get("ft", [None, None])
            records.append({
                "season": season,
                "round": match.get("round"),
                "date": match.get("date"),
                "team1": team1,
                "team2": team2,
                "score_home": ft[0],
                "score_away": ft[1],
            })

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)  # true chronological order, not round order

df.to_csv("data/processed/spurs_matches.csv", index=False)
print(f"Extracted {len(df)} Spurs matches across {df['season'].nunique()} seasons")