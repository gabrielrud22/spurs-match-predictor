import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"  # Update with your API endpoint

opponents_response = requests.get(f"{API_URL}/opponents")
KNOWN_OPPONENTS = opponents_response.json()["opponents"]

history = pd.read_csv("data/processed/spurs_matches_features.csv")
history["date"] = pd.to_datetime(history["date"])

st.title("Spurs Match Predictor")

st.write("This tool can be used to predict the outcome of a Tottenham Hotspur match against a specific opponent on a specific date.  " \
"The predictions are based on a machine learning model trained on historical match data.")

min_date = history["date"].min().date()
max_date = history["date"].max().date()

opponent = st.selectbox("Opponent", KNOWN_OPPONENTS)
date = st.date_input("Fixture Date", min_value=min_date, max_value=max_date)

if st.button("Predict"):
    payload = {"opponent": opponent, "date": str(date)}
    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.subheader(f"{result['opponent']} = {result['date']}")
        st.write(result["probabilities"])

        probs = result['probabilities']
        win_pct = float(probs['win'].strip('%'))
        draw_pct = float(probs['draw'].strip('%'))
        loss_pct = float(probs['loss'].strip('%'))

        if win_pct > draw_pct and win_pct > loss_pct:
            st.balloons()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")


# with st.sidebar:

# st.title("Hello, Streamlit")
# st.write("This page started as a normal Python script.")

# x = st.slider("Pick a number", min_value=1, max_value=20, value=5)
# st.write(f"The square of {x} is {x * x}.")

# if st.button("Celebrate"):
#     st.balloons()
