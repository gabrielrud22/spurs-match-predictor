import json #Python library for parsing JSON data
import glob #Python library for file pattern matching

def impute_halftime(match): #defines a function that takes a match dictionary as input and imputes halftime scores if they are missing
    score = match.get("score", {}) #retrieves the value stored under the key "score" in the match dictionary, or an empty dictionary if the key is not present
        #using .get("score", {}) handles cases where the "score" key is empty and returns an empty dictionary instead of raising an error
    if isinstance(score, dict): #checks that the score variable is indeed a dictionary before proceeding
        ft = score.get("ft") #extracts the FT score from the score dictionary, which is a two-item list
        ht = score.get("ht") #extracts the HT score from the score dictionary in the same way

        if ht is None and ft == [0, 0]: #this is the key imputation logic: if the HT score is missing (None) and the FT score is [0, 0], it sets the HT score to [0, 0]
            ht = [0, 0] 

        new_score = {} #creates a brand new empty dictionary to hold the updated score information
        if ht is not None: #Now, if HT has an actual value, it is inserted into the new_score dictionary under the key "ht"
            new_score["ht"] = ht 
        if ft is not None: #same logic as above, FT is inserted under the key "ft".
            new_score["ft"] = ft #the HT key will always appear first in the score dictionaries due to its ordering here

        match["score"] = new_score #replaces the match "score" value with the new, consistently ordered score dictionary
    return match

for filepath in glob.glob("data/raw/*.json"): #loops over every JSON files in the data/raw folder, using glob to find all files that match the pattern "data/raw/*.json"
    with open(filepath, encoding="utf-8") as f: #opens the current file for reading -- the with statement ensures that the file is properly closed afterwards
        data = json.load(f) #parses the contents of the open JSON file into a Python dictionary 

    data["matches"] = [impute_halftime(match) for match in data.get("matches", [])] #a list comprehension that applies the impute_halftime function to each match in the "matches" list of the data dictionary, effectively updating all matches in the file

    with open(filepath, "w", encoding="utf-8") as f: #reopens the same file in write mode (indicated by "w") to overwrite it with the updated data
        json.dump(data, f, ensure_ascii=False, indent=4) #writes the modified data dictionary back out to the file as JSON, indent=4 formats it for readability 

print("Imputed HT scores as 0-0 where FT scores were 0-0.")
