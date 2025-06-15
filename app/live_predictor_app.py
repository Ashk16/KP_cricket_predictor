#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Web Interface for Live Cricket Match Predictor
Uses KP Astrology + Machine Learning for real-time predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os
import altair as alt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the live predictor
from scripts.live_predictor import LiveCricketTimelinePredictor, get_venue_coordinates

# Page configuration
st.set_page_config(
    page_title="🔮 KP Cricket Timeline Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .winner-text {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .kp-analysis {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .team-prob {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Assuming the project structure is consistent
# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the directory of the current script (app/)
app_dir = os.path.dirname(current_script_path)
# Get the parent directory (the project root)
ROOT_DIR = os.path.dirname(app_dir)
# Add the project root to the Python path
sys.path.insert(0, ROOT_DIR)

# --- Configuration & Setup ---
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Caching ---
@st.cache_data
def get_coordinates(location_name):
    """Cached function to get lat/lon from a location name."""
    try:
        geolocator = Nominatim(user_agent="kp_cricket_predictor")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None, None

@st.cache_resource
def load_predictor():
    """Load the timeline predictor once and cache it"""
    return LiveCricketTimelinePredictor()

# --- Core Functions ---
def parse_time_string(time_str):
    """Parse time string in various formats."""
    formats = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Time data '{time_str}' is not in a recognized format (expected HH:MM or HH:MM:SS)")

def create_timeline_charts(timeline_data):
    """Create interactive charts using Altair"""
    
    # Prepare data for plotting
    chart_data = []
    for period in timeline_data:
        period_info = period['period_info']
        favorability = period['team_favorability']
        dynamics = period['match_dynamics']
        
        chart_data.append({
            'period_num': period_info['period_num'],
            'start_time': period_info['start_time'],
            'duration': period_info['duration_minutes'],
            'period_type': period_info['period_type'],
            'kp_score': favorability['kp_score'],
            'favorability_strength': favorability['favorability_strength'] * 100,
            'ascendant_favored': favorability['ascendant_favored'],
            'high_scoring': dynamics['high_scoring_probability'],
            'wicket_pressure': dynamics['wicket_pressure_probability'],
            'momentum_shift': dynamics['momentum_shift_probability'],
            'collapse_risk': dynamics['collapse_probability']
        })
    
    df = pd.DataFrame(chart_data)
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    # Chart 1: KP Score Timeline
    kp_chart = alt.Chart(df).mark_line(
        point=True, 
        color='blue',
        strokeWidth=2
    ).encode(
        x=alt.X('start_time:T', title='Time'),
        y=alt.Y('kp_score:Q', title='KP Score'),
        tooltip=['start_time:T', 'kp_score:Q', 'period_type:N']
    ).properties(
        width='container',
        height=200,
        title='KP Score Timeline'
    )
    
    # Chart 2: Favorability Strength
    favorability_chart = alt.Chart(df).mark_area(
        opacity=0.7,
        color='green'
    ).encode(
        x=alt.X('start_time:T', title='Time'),
        y=alt.Y('favorability_strength:Q', title='Favorability Strength (%)'),
        tooltip=['start_time:T', 'favorability_strength:Q']
    ).properties(
        width='container',
        height=200,
        title='Team Favorability Strength'
    )
    
    # Chart 3: Match Dynamics - Melt data for multi-line chart
    dynamics_data = []
    for _, row in df.iterrows():
        dynamics_data.extend([
            {'start_time': row['start_time'], 'metric': 'High Scoring', 'value': row['high_scoring']},
            {'start_time': row['start_time'], 'metric': 'Wicket Pressure', 'value': row['wicket_pressure']},
            {'start_time': row['start_time'], 'metric': 'Momentum Shift', 'value': row['momentum_shift']},
            {'start_time': row['start_time'], 'metric': 'Collapse Risk', 'value': row['collapse_risk']}
        ])
    
    dynamics_df = pd.DataFrame(dynamics_data)
    
    dynamics_chart = alt.Chart(dynamics_df).mark_line(
        point=True,
        strokeWidth=2
    ).encode(
        x=alt.X('start_time:T', title='Time'),
        y=alt.Y('value:Q', title='Probability (%)'),
        color=alt.Color('metric:N', 
                       scale=alt.Scale(range=['orange', 'red', 'purple', 'darkred']),
                       title='Match Dynamics'),
        tooltip=['start_time:T', 'metric:N', 'value:Q']
    ).properties(
        width='container',
        height=250,
        title='Match Dynamics Probabilities'
    )
    
    return kp_chart, favorability_chart, dynamics_chart

def display_period_details(timeline_data, match_info):
    """Display detailed period-by-period analysis"""
    
    st.subheader("📋 Period-by-Period Analysis")
    
    for i, period in enumerate(timeline_data):
        period_info = period['period_info']
        favorability = period['team_favorability']
        dynamics = period['match_dynamics']
        notes = period['astrological_notes']
        
        # Create expandable section for each period
        with st.expander(f"🌟 Period {period_info['period_num']} - {period_info['start_time']} ({period_info['period_type']})"):
            
            # Period Information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**⏰ Period Info**")
                st.write(f"**Start:** {period_info['start_time']}")
                st.write(f"**End:** {period_info['end_time']}")
                st.write(f"**Duration:** {period_info['duration_minutes']} minutes")
                st.write(f"**Type:** {period_info['period_type']}")
            
            with col2:
                st.markdown("**🌙 Planetary Lords**")
                st.write(f"**Star Lord:** {period_info['moon_nakshatra']}")
                st.write(f"**Sub Lord:** {period_info['moon_sub_lord']}")
                st.write(f"**Sub-Sub Lord:** {period_info['moon_sub_sub_lord']}")
            
            with col3:
                st.markdown("**🎯 Team Favorability**")
                favored_team = match_info['team1'] if favorability['ascendant_favored'] else match_info['team2']
                st.write(f"**Favored:** {favored_team}")
                st.write(f"**Strength:** {favorability['confidence_level']}")
                st.write(f"**KP Score:** {favorability['kp_score']:.2f}")
            
            # Match Dynamics with Progress Bars
            st.markdown("**📊 Match Dynamics**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("High Scoring", f"{dynamics['high_scoring_probability']:.1f}%")
                st.progress(dynamics['high_scoring_probability'] / 100)
                
                st.metric("Momentum Shift", f"{dynamics['momentum_shift_probability']:.1f}%")
                st.progress(dynamics['momentum_shift_probability'] / 100)
            
            with col2:
                st.metric("Wicket Pressure", f"{dynamics['wicket_pressure_probability']:.1f}%")
                st.progress(dynamics['wicket_pressure_probability'] / 100)
                
                st.metric("Collapse Risk", f"{dynamics['collapse_probability']:.1f}%")
                st.progress(dynamics['collapse_probability'] / 100)
            
            # Astrological Notes
            if notes:
                st.markdown("**🔮 Astrological Notes**")
                for note in notes:
                    st.write(f"• {note}")

def create_summary_metrics(timeline_data, match_info):
    """Create summary metrics for the entire match"""
    
    # Calculate averages and key insights
    total_periods = len(timeline_data)
    avg_high_scoring = np.mean([p['match_dynamics']['high_scoring_probability'] for p in timeline_data])
    avg_wicket_pressure = np.mean([p['match_dynamics']['wicket_pressure_probability'] for p in timeline_data])
    avg_collapse_risk = np.mean([p['match_dynamics']['collapse_probability'] for p in timeline_data])
    
    # Count favorable periods for each team
    team1_favorable = sum(1 for p in timeline_data if p['team_favorability']['ascendant_favored'])
    team2_favorable = total_periods - team1_favorable
    
    # Display metrics
    st.subheader("📈 Match Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Periods", total_periods)
        st.metric(f"{match_info['team1']} Favorable", f"{team1_favorable} ({team1_favorable/total_periods*100:.1f}%)")
    
    with col2:
        st.metric(f"{match_info['team2']} Favorable", f"{team2_favorable} ({team2_favorable/total_periods*100:.1f}%)")
        st.metric("Avg High Scoring", f"{avg_high_scoring:.1f}%")
    
    with col3:
        st.metric("Avg Wicket Pressure", f"{avg_wicket_pressure:.1f}%")
        st.metric("Avg Collapse Risk", f"{avg_collapse_risk:.1f}%")
    
    with col4:
        # Overall match prediction
        if team1_favorable > team2_favorable:
            predicted_winner = match_info['team1']
            win_confidence = team1_favorable / total_periods * 100
        else:
            predicted_winner = match_info['team2']
            win_confidence = team2_favorable / total_periods * 100
        
        st.metric("Predicted Winner", predicted_winner)
        st.metric("Win Confidence", f"{win_confidence:.1f}%")

# --- UI Layout ---

# --- Session State Initialization ---
if 'match_name' not in st.session_state:
    st.session_state.match_name = "TeamA_vs_TeamB"
if 'match_date' not in st.session_state:
    st.session_state.match_date = datetime.today()
if 'match_time_str' not in st.session_state:
    st.session_state.match_time_str = "19:30"
if 'latitude' not in st.session_state:
    st.session_state.latitude = 0.0
if 'longitude' not in st.session_state:
    st.session_state.longitude = 0.0
if 'team1_name' not in st.session_state:
    st.session_state.team1_name = "Team A"
if 'team2_name' not in st.session_state:
    st.session_state.team2_name = "Team B"
if 'duration_hours' not in st.session_state:
    st.session_state.duration_hours = 4
if 'timeline_results' not in st.session_state:
    st.session_state.timeline_results = {}

st.title("🔮 KP Cricket Timeline Predictor - ML Enhanced")

# Load predictor
predictor = load_predictor()

if not predictor.models:
    st.error("❌ No trained models found. Please train the multi-target models first.")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Match Settings")
    
    # Section: Generate New Timeline
    st.subheader("1. Generate New Timeline")
    
    # Match Details
    team1_name = st.text_input("Team 1 Name", value=st.session_state.team1_name)
    team2_name = st.text_input("Team 2 Name", value=st.session_state.team2_name)
    match_date = st.date_input("Match Date", value=st.session_state.match_date)
    match_time_str = st.text_input("Match Time (HH:MM)", value=st.session_state.match_time_str)
    duration_hours = st.slider("Match Duration (hours)", min_value=2, max_value=8, value=st.session_state.duration_hours)
    
    # Location Search
    st.subheader("Location")
    location_query = st.text_input("Search for Match Venue (e.g., 'Mumbai, India')")
    if st.button("Find Coordinates"):
        if location_query:
            with st.spinner("Finding location..."):
                coords = get_coordinates(location_query)
            if coords and coords[0] is not None and coords[1] is not None:
                st.session_state.latitude, st.session_state.longitude = coords
                st.success(f"Found: Lat {coords[0]:.4f}, Lon {coords[1]:.4f}")
                st.rerun()
            else:
                st.error("Location not found or service unavailable.")
        else:
            st.warning("Please enter a location to search for.")
    
    # Manual Location Input
    lat = st.number_input("Latitude", value=st.session_state.latitude, format="%.4f")
    lon = st.number_input("Longitude", value=st.session_state.longitude, format="%.4f")
    
    # Generate Button
    if st.button("🔮 Generate Timeline Prediction", type="primary"):
        if lat == 0.0 and lon == 0.0:
            st.error("Please provide valid coordinates for the match venue.")
        else:
            try:
                # Parse match time
                match_time = parse_time_string(match_time_str)
                match_datetime = datetime.combine(match_date, match_time)
                
                # Add timezone (assuming local timezone)
                local_tz = pytz.timezone('Asia/Kolkata')  # Default to IST, can be made configurable
                match_dt = local_tz.localize(match_datetime)
                
                # Generate timeline prediction
                with st.spinner("🔮 Generating astrological timeline predictions..."):
                    timeline_result = predictor.predict_timeline(
                        match_dt, lat, lon, team1_name, team2_name, duration_hours
                    )
                
                if 'error' not in timeline_result:
                    # Store in session state
                    match_key = f"{team1_name}_vs_{team2_name}_{match_date}"
                    st.session_state.timeline_results[match_key] = timeline_result
                    
                    # Update session state
                    st.session_state.team1_name = team1_name
                    st.session_state.team2_name = team2_name
                    st.session_state.match_date = match_date
                    st.session_state.match_time_str = match_time_str
                    st.session_state.latitude = lat
                    st.session_state.longitude = lon
                    st.session_state.duration_hours = duration_hours
                    
                    st.success(f"✅ Timeline prediction generated for {team1_name} vs {team2_name}!")
                    st.rerun()
                else:
                    st.error(f"❌ Error generating timeline: {timeline_result['error']}")
                
            except Exception as e:
                st.error(f"❌ Error generating timeline: {str(e)}")
    
    # Model Information
    st.subheader("🤖 Model Information")
    st.write("**Loaded Models:**")
    for target, info in predictor.models.items():
        target_name = target.replace('_target', '').replace('_', ' ').title()
        st.write(f"• {target_name}: {info['accuracy']:.1%}")

# --- Display Section ---
if st.session_state.timeline_results:
    # Create tabs for each match
    match_keys = list(st.session_state.timeline_results.keys())
    tabs = st.tabs([key.replace('_', ' ') for key in match_keys])
    
    for tab, match_key in zip(tabs, match_keys):
        with tab:
            timeline_result = st.session_state.timeline_results[match_key]
            match_info = timeline_result['match_info']
            timeline_data = timeline_result['timeline']
            
            # Match Header
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f"### 🏏 {match_info['team1']} vs {match_info['team2']}")
                st.markdown(f"**📅 Date:** {match_info['match_datetime']} | **📍 Location:** {match_info['location']}")
            with col2:
                if st.button("❌ Close", key=f"close_{match_key}"):
                    del st.session_state.timeline_results[match_key]
                    st.rerun()
            
            # Summary Metrics
            create_summary_metrics(timeline_data, match_info)
            
            st.divider()
            
            # Interactive Charts
            st.subheader("📊 Timeline Visualization")
            kp_chart, favorability_chart, dynamics_chart = create_timeline_charts(timeline_data)
            
            # Display charts
            st.altair_chart(kp_chart, use_container_width=True)
            st.altair_chart(favorability_chart, use_container_width=True)
            st.altair_chart(dynamics_chart, use_container_width=True)
            
            st.divider()
            
            # Detailed Period Analysis
            display_period_details(timeline_data, match_info)

else:
    st.info("🔮 Generate a new timeline prediction using the controls in the sidebar to see comprehensive astrological analysis with ML-powered match dynamics predictions.")
    
    # Show example of what the output will contain
    st.subheader("📋 What You'll Get:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Team Favorability Analysis:**
        - Ascendant vs Descendant favorability for each period
        - KP prediction strength and confidence levels
        - Overall match winner prediction
        
        **📊 Match Dynamics Predictions:**
        - High scoring probability (ML-powered)
        - Wicket pressure probability (ML-powered)
        - Momentum shift probability (ML-powered)
        - Collapse risk assessment (ML-powered)
        """)
    
    with col2:
        st.markdown("""
        **🌟 Astrological Timeline:**
        - Dynamic periods based on planetary movements
        - Moon's nakshatra, sub lord, and sub-sub lord changes
        - Period duration and timing analysis
        
        **🔮 Authentic KP Insights:**
        - Traditional KP astrological principles
        - Planetary influence analysis
        - Retrograde planet effects
        - Nakshatra significance notes
        """)

# Main execution handled by Streamlit 