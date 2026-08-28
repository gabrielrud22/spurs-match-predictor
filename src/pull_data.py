import json #Python library to read JSON files
import glob #finds files matching a wildcard pattern (all JSON files)
import pandas as pd #Dataframe library aliased as "pd" by convention

records = [] #empty list that will hold one dictionary per Spurs match
for filepath in glob.glob("data/raw/*.json"): # loop over every JSON file in the data/raw folder (uses glob)
    with open(filepath, encoding="utf-8") as f: #opens the file for reading, the "with" statement ensures the file's safe closure
        data = json.load(f) #parse the file contents into a Python dictionary

    season = data.get("name", "") #pull the season name; .get with a default value of "" avoids KeyError if the key is missing

    for match in data.get("matches", []): #loop over every match dict inside the relevant season
        team1 = match.get("team1", "") #home team for this match, "" if the key is missing
        team2 = match.get("team2", "") #same as above for away team

        if any("Tottenham Hotspur" in team for team in (team1, team2)): #checks if "Tottenham Hotspur" appears as a substring in either team1 or team2
            #substring match is not exact equality so it catches naming variants like "Tottenham Hotspur FC"
            ft = match.get("score", {}).get("ft", [None, None]) 
            records.append({
                "season": season, #which season the match belongs to
                "round": match.get("round"), #matchday label
                "date": match.get("date"), #fixture date as a string
                "home_team": team1, #renamed from team1 for clarity
                "away_team": team2, #renamed from team2 for clarity
                "score_home": ft[0], #first item of the ft list, home side goals
                "score_away": ft[1], #second item of the ft list, away side goals
            })

df = pd.DataFrame(records) #converts the list of dictionaries into a pandas DataFrame (tabular structure)
df["date"] = pd.to_datetime(df["date"]) #converts the "date" column from string to datetime objects for easier manipulation and sorting
df = df.sort_values("date").reset_index(drop=True)  # sorts to true chronological order, not round order

df.to_csv("data/processed/spurs_matches.csv", index=False)#writes the finished DF to the specified CSV file
print(f"Extracted {len(df)} Spurs matches across {df['season'].nunique()} seasons")