import streamlit as st
import requests
import pandas as pd 

API_URL = "http://localhost:8000"

history = pd.read_csv("data/processed/spurs_matches_features.csv")
history["date"] = pd.to_datetime(history["date"])

st.header("Match Replay Tool")
st.write("Replay a past match and observe the model's prediction for that match.")

match_options = history.apply(
    lambda row: f"{row['opponent_team']} - {row['date'].date()}", axis=1
).tolist()
selected_match = st.selectbox("Choose a real fixture", match_options)

if st.button("Replay Prediction"):
    selected_opponent, selected_date_str = selected_match.split(" - ")
    payload = {"opponent": selected_opponent, "date": selected_date_str}
    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.write(result["probabilities"])

        actual_row = history[
            (history["opponent_team"] == selected_opponent) &
            (history["date"] == pd.to_datetime(selected_date_str))
        ].iloc[0]
        st.caption(f"Actual result: {actual_row['result']}")
    else:
        st.error(f"Error: not enough match history to compute rolling form feature")