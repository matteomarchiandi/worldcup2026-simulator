import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(page_title="World Cup 2026 Match Predictor", page_icon="⚽", layout="centered")

st.title("🌍 2026 World Cup Match Simulator")
st.markdown("Select two teams to simulate a single match based on their ELO ratings and the trained XGBoost model (data last updated on 2026-06-27).")

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


# --- Feature: Group Finder ---
wc_26_groups = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "Group B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Group D": ["United States", "Paraguay", "Australia", "Turkey"],
    "Group E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "Group I": ["France", "Senegal", "Iraq", "Norway"],
    "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"]
}
st.markdown("---")# --- App Navigation (Tabs) ---
tab1, tab2, tab3 = st.tabs(["⚽ Match Simulator", "🏆 Winning Probability", "📈 Power Rankings"])


# ==========================================
# TAB 1: MATCH SIMULATOR
# ==========================================
with tab1:

    st.subheader("🔍 Group Finder")

    search_team = st.selectbox(
        "Select a team to find their World Cup Group:", 
        team_list, 
        index=team_list.index("Mexico") if "Mexico" in team_list else 0
    )

    if search_team:
        found_group = next((group for group, teams in wc_26_groups.items() if search_team in teams), None)
        
        if found_group:
            group_teams = wc_26_groups[found_group]
            
            st.markdown(f"### {found_group} Teams")
            cols = st.columns(4)
            for idx, team in enumerate(group_teams):
                with cols[idx]:
                    if team == search_team:
                        st.info(f" **{team}**")
                    else:
                        st.info(team)
        else:
            st.warning("This team was not found in the 2026 World Cup groups definition.")

    st.subheader("⚽ Match Setup")

    if not team_list:
        st.error("⚠️ No teams found! Please check your `current_elos.json` file.")
        st.stop()

    col1, col2 = st.columns(2)

    selectable_teams = ["Select a team"] + team_list

    with col1:
        h_selection = st.selectbox("Home Team", selectable_teams, index=0)
        
        if h_selection != "Select a team":
            team_h = h_selection
            elo_h = get_clean_elo(team_h)
            st.info(f"Current ELO: **{elo_h:.1f}**")
        else:
            team_h = None
            st.info("Select a team to see their ELO")

    with col2:
        away_options = ["Select a team"] + [t for t in team_list if t != team_h]
        a_selection = st.selectbox("Away Team", away_options, index=0)
        
        if a_selection != "Select a team":
            team_a = a_selection
            elo_a = get_clean_elo(team_a)
            st.info(f"Current ELO: **{elo_a:.1f}**")
        else:
            team_a = None
            st.info("Select a team to see their ELO")

    st.markdown("---")
    # --- Match Settings ---
    st.markdown("### Simulation Settings")
    is_knockout = st.checkbox("Knockout Match (no draws allowed)", value=False)
    n_sims = st.slider("Number of Match Simulations", min_value=1000, max_value=10000, value=5000, step=1000)

    # --- Simulation Logic ---
    if st.button("Simulate Match", type="primary"):
        if not team_h or not team_a:
            st.warning("⚠️ Please select both a Home Team and an Away Team before simulating!")
        else:
           # 1. Build BOTH feature arrays (Forward and Backward)
            match_forward = pd.DataFrame(
                [[elo_h - elo_a, 1, 0, 0, 0, 0, 1]],
                columns=['elo_pre_diff', 'neutral', 'k_20', 'k_30', 'k_40', 'k_50', 'k_60']
            )
            
            match_backward = pd.DataFrame(
                [[elo_a - elo_h, 1, 0, 0, 0, 0, 1]],
                columns=['elo_pre_diff', 'neutral', 'k_20', 'k_30', 'k_40', 'k_50', 'k_60']
            )

            # 2. Get probabilities for both scenarios 
            probs_forward = wc_26_predictor.predict_proba(match_forward)[0]
            probs_backward = wc_26_predictor.predict_proba(match_backward)[0]

            # 3. Inference Averaging: Eliminate bias
            win_h_avg = (probs_forward[1] + probs_backward[2]) / 2
            win_a_avg = (probs_forward[2] + probs_backward[1]) / 2
            draw_avg = (probs_forward[0] + probs_backward[0]) / 2

            win_team_1 = 0
            win_team_2 = 0
            draws = 0

            # 4. Run the Monte Carlo Loop 
            for _ in range(n_sims):
                rand_roll = np.random.rand()
                
                if is_knockout:
                    # Redistribute draws equally to both sides for knockout rules
                    win_h_prob = win_h_avg + (draw_avg / 2)
                    if rand_roll < win_h_prob:
                        win_team_1 += 1
                    else:
                        win_team_2 += 1
                else:
                    # Standard group stage rules
                    if rand_roll < win_h_avg:
                        win_team_1 += 1
                    elif rand_roll < win_h_avg + draw_avg:
                        draws += 1
                    else:
                        win_team_2 += 1

            # --- Display Results ---
            st.markdown("---")
            st.subheader("📊 Simulation Results")
            
            p_team_1 = (win_team_1 / n_sims) * 100
            p_team_2 = (win_team_2 / n_sims) * 100
            p_draw = (draws / n_sims) * 100

            if is_knockout:
                m1, m2 = st.columns(2)
                m1.metric(f"{team_h} Win Probability", f"{p_team_1:.1f}%")
                m2.metric(f"{team_a} Win Probability", f"{p_team_2:.1f}%")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{team_h} Win Probability", f"{p_team_1:.1f}%")
                m2.metric("Draw Probability", f"{p_draw:.1f}%")
                m3.metric(f"{team_a} Win Probability", f"{p_team_2:.1f}%")

            fig, ax = plt.subplots(figsize=(6, 3))
            if is_knockout:
                data = {f"{team_h} Win": p_team_1, f"{team_a} Win": p_team_2}
            else:
                data = {f"{team_h} Win": p_team_1, "Draw": p_draw, f"{team_a} Win": p_team_2}
                
            sns.barplot(x=list(data.values()), y=list(data.keys()), palette="magma", ax=ax)
            ax.set_xlabel("Probability (%)")
            st.pyplot(fig)

# ==========================================
# TAB 2: STATIC SIMULATION RESULTS
# ==========================================
with tab2:
    st.subheader("🏆 Full Tournament Prediction Results")
    st.markdown("These graphics show the winning probability of a team calculated across **50,000 Monte Carlo simulations** of the actual knockout stage from the semifinals using the model trained.")
    
    # Hard-coded results based on your XGBoost model's output
    # Numbers represent total tournament wins out of 10,000 simulations
    static_probabilities = {
        "Argentina": 35.37, "France": 27.56, "Spain": 27.17, "England": 10.08
    }
        
    # Convert the direct probabilities directly to a DataFrame
    df_static = pd.DataFrame(list(static_probabilities.items()), columns=["Team", "Win Probability (%)"])
    
    # Sort from highest to lowest probability
    df_static = df_static.sort_values(by="Win Probability (%)", ascending=False).reset_index(drop=True)
    df_static.index += 1
    
    # Create the Layout
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.markdown("**Teams Leaderboard**")
        st.dataframe(
            df_static.style.format({"Win Probability (%)": "{:.2f}%"}), 
            use_container_width=True,
            height=150
        )
        
    with col_right:
        st.markdown("**Winning Probabilities (Top 15)**")
        fig_static, ax_static = plt.subplots(figsize=(8, 8))
        
        # Show the top 15 teams in the chart for a better overview
        sns.barplot(
            data=df_static.head(15), 
            x="Win Probability (%)", 
            y="Team", 
            palette="flare", 
            ax=ax_static
        )
        ax_static.set_xlabel("Probability to Win the World Cup (%)")
        ax_static.set_ylabel("")
        st.pyplot(fig_static)


# ==========================================
# TAB 3: POWER RANKINGS
# ==========================================
with tab3:
    st.subheader("📈 Power Rankings")
    st.markdown("Current ELO scores for the 2026 World Cup teams.")
    
    # 1. Create a clean list of all teams and their ELOs using our safe function
    ranking_data = []
    for t in team_list:
        ranking_data.append({"Team": t, "ELO Rating": get_clean_elo(t)})
        
    # 2. Convert to a Pandas DataFrame and sort it from highest to lowest
    df_ranks = pd.DataFrame(ranking_data)
    df_ranks = df_ranks.sort_values(by="ELO Rating", ascending=False).reset_index(drop=True)
    
    # 3. Make the index start at 1 (for Rank 1, Rank 2, etc.)
    df_ranks.index += 1
    
    colA, colB = st.columns([1, 1.5])
    
    with colA:
        # Display the full scrollable leaderboard
        st.dataframe(
            df_ranks.style.format({"ELO Rating": "{:.1f}"}), 
            use_container_width=True,
            height=1000
        )
        
    with colB:
        st.markdown("**Top 15 Teams**")
        # Plot a horizontal bar chart of just the Top 15 teams
        fig_rank, ax_rank = plt.subplots(figsize=(8, 10))
        sns.barplot(data=df_ranks.head(15), x="ELO Rating", y="Team", palette="viridis", ax=ax_rank)
        
        # Adjust the X-axis so the differences are more visible (zoomed in)
        min_elo = df_ranks.head(15)["ELO Rating"].min() - 50
        ax_rank.set_xlim(left=min_elo)
        ax_rank.set_xlabel("ELO Rating")
        ax_rank.set_ylabel("")
        
        st.pyplot(fig_rank)

st.markdown("---")  # Adds a clean separating line
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        ⚠️ <strong>Disclaimer:</strong> this application is an independent data science project developed for educational and demonstration purposes. 
        All predictions and probabilities are generated using a machine learning model based on historical data and do not constitute official forecasts, 
        guarantees of match outcomes or betting advice in any way, shape or form.
    </div>
    """,
    unsafe_allow_html=True
)
