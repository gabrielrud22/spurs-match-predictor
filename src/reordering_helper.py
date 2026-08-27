import json
import glob

for filepath in glob.glob("data/raw/*.json"):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    for match in data.get("matches", []):
        score = match.get("score", {})
        if isinstance(score, dict):
            ht = score.get("ht", [0, 0] if score.get("ft") == [0, 0] else None)
            ft = score.get("ft")

            new_score = {}
            if ht is not None:
                new_score["ht"] = ht
            if ft is not None:
                new_score["ft"] = ft

            match["score"] = new_score

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

print("Reordered ht/ft keys consistently across all matches")