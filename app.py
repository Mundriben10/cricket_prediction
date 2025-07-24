import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# Define teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the trained model
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'pipe.pkl' not found. Please ensure it is in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Streamlit app title and layout
st.title('IPL Win Predictor')
st.markdown('Real-time win probability prediction inspired by live cricket analytics.')

# Create two columns for team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Ensure batting and bowling teams are different
if batting_team == bowling_team:
    st.warning("Batting and bowling teams cannot be the same!")
    st.stop()

# Select host city and pitch condition (simulating impact)
selected_city = st.selectbox('Select host city', sorted(cities))
pitch_condition = st.selectbox('Pitch Condition', ['Normal', 'Bowler-friendly', 'Batsman-friendly'])

# Input for target score and recent form
target = st.number_input('Target Runs', min_value=1, max_value=500, step=1, format="%d")
recent_form = st.slider('Recent Form (0-10, 10 being best)', 0, 10, 5)  # Simulate team momentum

# Toss decision inputs
col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox('Select toss winner', sorted(teams))
with col4:
    toss_decision = st.selectbox('Select toss decision', ['bat', 'field'])

# Create three columns for score, overs, and wickets
col5, col6, col7 = st.columns(3)
with col5:
    score = st.number_input('Current Score', min_value=0, max_value=500, step=1, format="%d")
with col6:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col7:
    wickets = st.number_input('Wickets Lost', min_value=0, max_value=10, step=1, format="%d")

# Reset button
if st.button('Reset Inputs'):
    st.experimental_rerun()

# Predict button
if st.button('Predict Probability'):
    # Input validations
    if score > target:
        st.error("Current score cannot be greater than the target score!")
    elif overs == 0 and score > 0:
        st.error("If overs completed is 0, score must be 0!")
    elif overs > 20:
        st.error("Overs cannot exceed 20!")
    elif balls_left := 120 - (overs * 6) <= 0:
        st.error("Overs completed result in negative or zero balls left!")
    elif wickets > 10:
        st.error("Wickets lost cannot exceed 10!")
    else:
        # Show progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        # Calculate derived features
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_remaining = 10 - wickets
        current_run_rate = score / overs if overs > 0 else 0
        required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')

        # Simulate momentum based on recent form and run rate trend
        momentum_factor = recent_form / 10.0  # Normalize to 0-1
        run_rate_trend = current_run_rate / 6.0 if current_run_rate > 0 else 0.5  # Normalize to 0-1 (avg ~6)
        momentum = min(momentum_factor * run_rate_trend * 1.5, 1.0)  # Cap at 1

        # Adjust pitch impact
        pitch_impact = {'Normal': 1.0, 'Bowler-friendly': 0.8, 'Batsman-friendly': 1.2}[pitch_condition]

        # Set team1 as batting team and team2 as bowling team
        team1 = batting_team  # Chasing team
        team2 = bowling_team  # Defending team

        # Create input DataFrame with enhanced features
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'wickets_remaining': [wickets_remaining],
            'total_runs_x': [target],
            'total_runs': [target],
            'current_run_rate': [current_run_rate],
            'required_run_rate': [required_run_rate],
            'team1': [team1],
            'team2': [team2],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'result': ['normal'],
            'momentum': [momentum],  # New feature
            'pitch_impact': [pitch_impact]  # New feature
        })

        # Make prediction
        try:
            result = pipe.predict_proba(input_df)
            # Assume result[0][0] is probability for team1 (batting team) winning
            # and adjust bowling team probability to sum to 100%
            batting_win_prob = result[0][0]  # Probability of batting team (team1) winning
            bowling_win_prob = 1.0 - batting_win_prob  # Ensure binary outcome

            # Format required_run_rate for display
            required_run_rate_display = f"{required_run_rate:.2f}" if required_run_rate != float('inf') else "N/A"

            # Display match situation summary
            st.subheader('Match Situation')
            st.write(f"**Runs Left**: {runs_left}")
            st.write(f"**Balls Left**: {balls_left}")
            st.write(f"**Wickets Remaining**: {wickets_remaining}")
            st.write(f"**Current Run Rate**: {current_run_rate:.2f}")
            st.write(f"**Required Run Rate**: {required_run_rate_display}")
            st.write(f"**Momentum Factor**: {momentum:.2f}")
            st.write(f"**Pitch Condition Impact**: {pitch_condition}")

            # Display results
            st.subheader('Win Probabilities')
            st.write(f"{batting_team} Win Probability: {batting_win_prob * 100:.2f}%")
            st.write(f"{bowling_team} Win Probability: {bowling_win_prob * 100:.2f}%")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")