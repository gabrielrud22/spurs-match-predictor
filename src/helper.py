import json
import glob

def impute_halftime(match):
    score = match.get("score", {})
    if isinstance(score, dict):
        ft = score.get("ft")
        if ft == [0, 0] and "ht" not in score:
            score["ht"] = [0, 0]
    return match

for filepath in glob.glob("data/raw/*.json"):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    data["matches"] = [impute_halftime(match) for match in data.get("matches", [])]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

print("Imputed HT scores as 0-0 where FT scores were 0-0.")
