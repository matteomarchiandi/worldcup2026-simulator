import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(page_title="World Cup 2026 Match Predictor", page_icon="⚽", layout="centered")

st.title("⚽ 2026 World Cup Match Simulator")
st.markdown("Select two teams to simulate a single match based on their ELO ratings and the trained XGBoost model (last data update on 2026-06-10).")

# --- Load Pre-computed Data & Model ---
@st.cache_resource
def load_assets():
    # Load ELO ratings
    with open('current_elos.json', 'r') as f:
        elos = json.load(f)
    
    # Recreate and load the XGBoost model
    model = XGBClassifier()
    model.load_model('wc_model.json')
    
    return elos, model

try:
    current_elos, wc_26_predictor = load_assets()
    team_list = sorted(list(current_elos.keys()))
except FileNotFoundError:
    st.error("Missing asset files! Please make sure 'current_elos.json' and 'wc_model.json' are in the same directory as this script.")
    st.stop()

# --- Team Selection UI ---
# --- Team Selection UI ---
st.subheader("Match")

# Helper function to handle lists or strings from the JSON
def get_clean_elo(team_name):
    # 1. Safety check for empty team names on first load
    if not team_name:
        return 1500.0
        
    try:
        # 2. Safely get the value (defaulting to 1500 if missing)
        val = current_elos.get(team_name, 1500.0)
        
        # 3. Handle if the value is a list (e.g., [1400, 1450, 1500])
        if isinstance(val, list):
            val = val[-1]
            
        # 4. Handle if the value is a dictionary (e.g., {"rating": 1500, "matches": 10})
        elif isinstance(val, dict):
            # Guess the most common keys for ELO
            if "rating" in val:
                val = val["rating"]
            elif "elo" in val:
                val = val["elo"]
            else:
                # Just grab the first value in the dictionary as a fallback
                val = list(val.values())[0]
                
        # 5. Force it to a decimal
        return float(val)
        
    except Exception as e:
        # If it STILL crashes, this will print the exact reason on your app screen!
        st.error(f"🚨 **Data Error for {team_name}:**")
        st.write(f"The raw value pulled from JSON is: `{val}` (Type: {type(val)})")
        st.write(f"The exact error is: `{e}`")
        st.stop()

col1, col2 = st.columns(2)

with col1:
    team_h = st.selectbox("Home Team", team_list, index=team_list.index("Brazil") if "Brazil" in team_list else 0)
    # Use the helper function here
    elo_h = get_clean_elo(team_h)
    st.info(f"Current ELO: **{elo_h:.1f}**")

with col2:
    # Filter out home team so they can't play themselves
    away_teams = [team for team in team_list if team != team_h]
    team_a = st.selectbox("Away Team", away_teams, index=away_teams.index("Morocco") if "Morocco" in away_teams else 0)
    # Use the helper function here
    elo_a = get_clean_elo(team_a)
    st.info(f"Current ELO: **{elo_a:.1f}**")

# --- Match Settings ---
st.markdown("### Simulation Settings")
is_knockout = st.checkbox("Knockout Match (no draws allowed)", value=False)
n_sims = st.slider("Number of Match Simulations", min_value=100, max_value=10000, value=5000, step=100)

# --- Simulation Logic ---
if st.button("Simulate Match", type="primary"):
    elo_diff = elo_h - elo_a

    # Match feature row exactly matching your notebook [elo_pre_diff, neutral, k_20, k_30, k_40, k_50, k_60]
    # Set to WC conditions: neutral=1, k_60=1
    match_features = pd.DataFrame(
        [[elo_diff, 1, 0, 0, 0, 0, 1]],
        columns=['elo_pre_diff', 'neutral', 'k_20', 'k_30', 'k_40', 'k_50', 'k_60']
    )

    # Get raw probabilities from your XGBoost classifier [0: Draw, 1: Home Win, 2: Away Win]
    probs = wc_26_predictor.predict_proba(match_features)[0]

    win_team_1 = 0
    win_team_2 = 0
    draws = 0

    # Run the simulation loop
    for _ in range(n_sims):
        rand_roll = np.random.rand()
        if is_knockout:
            # Redistribute draws equally to both sides for knockout rules
            win_h_prob = probs[1] + (probs[0] / 2)
            if rand_roll < win_h_prob:
                win_team_1 += 1
            else:
                win_team_2 += 1
        else:
            if rand_roll < probs[1]:
                win_team_1 += 1
            elif rand_roll < probs[0] + probs[1]:
                draws += 1
            else:
                win_team_2 += 1

    # --- Display Results ---
    st.markdown("---")
    st.subheader("📊 Simulation Results")
    
    p_team_1 = (win_team_1 / n_sims) * 100
    p_team_2 = (win_team_2 / n_sims) * 100
    p_draw = (draws / n_sims) * 100

    # Quick metric callouts
    if is_knockout:
        m1, m2 = st.columns(2)
        m1.metric(f"{team_h} Win Prob", f"{p_team_1:.1f}%")
        m2.metric(f"{team_a} Win Prob", f"{p_team_2:.1f}%")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_h} Win Prob", f"{p_team_1:.1f}%")
        m2.metric("Draw Prob", f"{p_draw:.1f}%")
        m3.metric(f"{team_a} Win Prob", f"{p_team_2:.1f}%")

    # Visual Chart
    fig, ax = plt.subplots(figsize=(6, 3))
    if is_knockout:
        data = {f"{team_h} Wins": p_team_1, f"{team_a} Wins": p_team_2}
    else:
        data = {f"{team_h} Wins": p_team_1, "Draws": p_draw, f"{team_a} Wins": p_team_2}
        
    sns.barplot(x=list(data.values()), y=list(data.keys()), palette="magma", ax=ax)
    ax.set_xlabel("Probability (%)")
    st.pyplot(fig)
