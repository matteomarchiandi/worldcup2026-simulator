# 🏆⚽ 2026 World Cup Simulator

An end-to-end simulator designed to forecast the 2026 World Cup matches and the full tournament.

## 🧠 Architecture & Methodology
This project is divided into three distinct engines:

1. The Data Engineering Pipeline

Elo Score Tracker implemented to compute teams' current Elo ratings using dynamic factors based on tournament importance and goal difference in matches.

Historical international matches data (from 1872 to present) is pulled and feed into the Feature Engineering process.

2. The Machine Learning Model (XGBoost)

The core predictive engine is a 3-class XGBoostClassifier trained on historical international matches between 2000-01-01 and 2026-06-10, handling neutral venue flags and World Cup-specific tournament weight.

3. 🏆 The Monte Carlo Simulation Framework
   
Simulates the full tournament multiple times from the group stage to the final, using the official knockout stage scheme.
Default is 10,000 simulations of the entire tournament returning the estimated probability of each team lifting the trophy.

## 💻 The Streamlit Web App
https://worldcup2026-simulator.streamlit.app/
The project includes an interactive web application built with Streamlit:

- Head to Head matchups, where the user can select any two national teams to get the model's predicted probabilities for a Win/Draw/Loss in a single match

- Full Tournament Simulation Results, displaying the results of 10,000-iterations simulations of the full tournament

- Power Rankings, displaying the scoreboard of the Elo ranking of World Cup teams.

## 📊 Data Source
International Matches: Historical match results since 1872.

# 📌 Further Improvements 
One could also consider additional features such as teams market value, World Ranking and recent form.


⚠️ Disclaimer: this project is an independent data science project developed for educational and demonstration purposes. All predictions and probabilities are generated using a machine learning model based on historical data and do not constitute official forecasts, guarantees of match outcomes or betting advice in any way, shape or form.
