import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta, datetime, time
import os
import pytz
import altair as alt
import sys
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import swisseph as swe

# Assuming the project structure is consistent
# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the directory of the current script (app/)
app_dir = os.path.dirname(current_script_path)
# Get the parent directory (the project root)
ROOT_DIR = os.path.dirname(app_dir)
# Add the project root to the Python path
sys.path.insert(0, ROOT_DIR)

from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability
from scripts.unified_kp_predictor import UnifiedKPPredictor

# --- Configuration & Setup ---
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NAKSHATRA_DATA_PATH = os.path.join(ROOT_DIR, "config", "nakshatra_sub_lords_longitudes.csv")

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

@st.cache_data
def cached_load_nakshatra_data():
    """Cached function to load nakshatra data once."""
    return pd.read_csv(NAKSHATRA_DATA_PATH)

# Load data once at startup for the favorability rules
NAKSHATRA_DF = cached_load_nakshatra_data()

# --- Core Functions ---
def parse_team_names(match_name: str):
    """Parse team names from match name with multiple separator support"""
    team_a = "TeamA"
    team_b = "TeamB"
    
    # Try different common separators
    separators = [" vs ", " v ", " VS ", " V ", " Vs "]
    for sep in separators:
        if sep in match_name:
            team_parts = match_name.split(sep, 1)  # Split only on first occurrence
            if len(team_parts) == 2:
                team_a = team_parts[0].strip()
                team_b = team_parts[1].strip()
                break
    
    return team_a, team_b
def find_next_ssl_change(current_dt: datetime, lat: float, lon: float, nakshatra_df: pd.DataFrame):
    """
    Calculates the exact datetime of the next Sub-Sub Lord change.
    Enhanced with duplicate prevention and boundary validation.
    """
    try:
        # 1. Get current moon position
        jd = swe.julday(current_dt.year, current_dt.month, current_dt.day, current_dt.hour + current_dt.minute/60 + current_dt.second/3600 - 5.5)
        pos = swe.calc_ut(jd, swe.MOON)
        moon_long = pos[0][0] % 360  # Ensure 0-360 range
        moon_speed_deg_per_day = pos[0][3]
        moon_speed_deg_per_sec = moon_speed_deg_per_day / (24 * 3600)

        # 2. Find the current SSL's end degree
        current_row = nakshatra_df[(nakshatra_df['Start_Degree'] <= moon_long) & (nakshatra_df['End_Degree'] > moon_long)]
        
        # Handle edge cases
        if current_row.empty:
            if moon_long >= 359.99:
                current_row = nakshatra_df.iloc[-1:]
            else:
                # Fallback to fixed interval
                return current_dt + timedelta(minutes=45)

        if current_row.empty:
            return current_dt + timedelta(minutes=45)
            
        end_degree = current_row.iloc[0]['End_Degree']

        # 3. Calculate time to reach the end degree
        if end_degree < moon_long:  # Handles the 360 -> 0 degree crossover
            degrees_to_travel = (360 - moon_long) + end_degree
        else:
            degrees_to_travel = end_degree - moon_long
        
        # Add buffer to ensure we cross the boundary, but not too much to avoid precision issues
        degrees_to_travel += 0.001

        if moon_speed_deg_per_sec <= 0:  # Should not happen
            return current_dt + timedelta(minutes=45)

        seconds_to_change = degrees_to_travel / moon_speed_deg_per_sec
        
        # Calculate next change time
        next_change_time = current_dt + timedelta(seconds=seconds_to_change)
        
        # Enforce minimum and maximum bounds to prevent edge cases
        min_next_time = current_dt + timedelta(minutes=1)   # Minimum 1 minute gap
        max_next_time = current_dt + timedelta(minutes=120) # Maximum 2 hours gap
        
        # Apply bounds
        if next_change_time < min_next_time:
            next_change_time = min_next_time
        elif next_change_time > max_next_time:
            next_change_time = max_next_time
        
        return next_change_time
        
    except Exception as e:
        # Fallback on any calculation error
        print(f"Error in find_next_ssl_change: {str(e)}")
        return current_dt + timedelta(minutes=45)


def generate_and_save_timeline(start_dt, lat, lon, match_name, model_type="comprehensive", duration_hours=4):
    """Generates the timeline DataFrame and saves it to a CSV."""
    
    # Use unified predictor for both models
    model_info = {"comprehensive": "🚀 Comprehensive KP Model", "legacy": "📊 Legacy KP Model"}
    st.info(f"Using {model_info.get(model_type, model_type)} with unified architecture")
    
    predictor = UnifiedKPPredictor(model_type)
    
    # Convert datetime components
    match_date = start_dt.strftime("%Y-%m-%d")
    start_time = start_dt.strftime("%H:%M:%S")
    
    # Parse team names with multiple separator support
    team_a, team_b = parse_team_names(match_name)
    
    # Generate timeline using unified predictor
    df = predictor.predict_match_timeline(
        team_a=team_a, 
        team_b=team_b, 
        match_date=match_date,
        start_time=start_time,
        lat=lat, 
        lon=lon, 
        duration_hours=duration_hours
    )
    
    # Save results using unified file manager
    filepath = predictor.save_results(df)
    
    # Convert datetime column back to string for display
    df['datetime'] = df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return df, filepath


def get_verdict_label(score):
    """Creates meaningful labels for score ranges."""
    if score > 25: return "Strongly Favors Ascendant"
    if score > 10: return "Clearly Favors Ascendant"
    if score > 2: return "Slightly Favors Ascendant"
    if score < -25: return "Strongly Favors Descendant"
    if score < -10: return "Clearly Favors Descendant"
    if score < -2: return "Slightly Favors Descendant"
    return "Neutral / Too Close to Call"

def get_saved_timelines():
    """Returns a list of saved CSV files in the results directory."""
    if not os.path.exists(RESULTS_DIR):
        return []
    return sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")], reverse=True)

def load_metadata_from_csv(file_path):
    """Reads metadata from commented lines at the top of a CSV file."""
    metadata = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
    except Exception:
        pass # Ignore errors if metadata can't be read
    return metadata

def display_df_colored(df):
    """
    Displays the dataframe with nuanced, multi-column colored scores.
    Shows all score columns after verdict and final_score, and uses full width.
    """
    def get_color_style(score, max_abs_score=50.0):
        if pd.isna(score): return 'color: black;'
        score = max(-max_abs_score, min(max_abs_score, score))
        norm_score = score / max_abs_score
        if norm_score > 0:
            lightness = 90 - (norm_score * 40)
            return f'background-color: hsl(120, 80%, {lightness}%); color: black;'
        elif norm_score < 0:
            lightness = 90 - (abs(norm_score) * 40)
            return f'background-color: hsl(0, 90%, {lightness}%); color: black;'
        else:
            return 'color: black;'

    # Desired column order
    display_cols = [
        'datetime', 'moon_star_lord', 'sub_lord', 'sub_sub_lord',
        'verdict', 'final_score', 'moon_sl_score', 'moon_sub_score', 'moon_ssl_score'
    ]
    df_display = df[display_cols].copy()

    styler = df_display.style

    def row_styler(row):
        styles = pd.Series('', index=row.index)
        styles['moon_star_lord'] = get_color_style(row['moon_sl_score'])
        styles['sub_lord'] = get_color_style(row['moon_sub_score'])
        styles['sub_sub_lord'] = get_color_style(row['moon_ssl_score'])
        styles['final_score'] = get_color_style(row['final_score'])
        styles['verdict'] = get_color_style(row['final_score'])
        styles['moon_sl_score'] = get_color_style(row['moon_sl_score'])
        styles['moon_sub_score'] = get_color_style(row['moon_sub_score'])
        styles['moon_ssl_score'] = get_color_style(row['moon_ssl_score'])
        return styles
    styler.apply(row_styler, axis=1)
    styler.hide(axis="index")
    st.dataframe(styler, use_container_width=True)

def parse_time_string(time_str):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Time data '{time_str}' is not in a recognized format (expected HH:MM or HH:MM:SS)")

# --- UI Layout ---

# --- Session State Initialization (must be at the very top, before any widgets) ---
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
if 'active_matches' not in st.session_state:
    st.session_state.active_matches = {}  # Dictionary to store match timelines
if 'closed_matches' not in st.session_state:
    st.session_state.closed_matches = set()  # Set to track closed matches

st.title("🪐 KP Cricket Predictor Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Match Settings")
    
    # Model Selection
    model_type = st.selectbox(
        "Select KP Model",
        ["comprehensive", "legacy"],
        index=0,
        help="Choose between different KP prediction models"
    )
    
    # Model Info
    if model_type == "comprehensive":
        st.success("🚀 Comprehensive Model Active")
        st.caption("✓ Dynamic weighting\n✓ Enhanced contradictions\n✓ Fixed planetary scores")
    else:
        st.info("📊 Legacy Model Active")
        st.caption("✓ Fixed 50%-30%-20% weighting\n✓ Original KP calculations")
    
    # Section: Generate New Timeline
    st.subheader("1. Generate New Timeline")
    
    # Match Details
    match_name = st.text_input("Match Name", value=st.session_state.match_name)
    match_date = st.date_input("Match Date", value=st.session_state.match_date)
    match_time_str = st.text_input("Match Time (HH:MM)", value=st.session_state.match_time_str)
    
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
    
    # Timeline Duration Control
    st.subheader("Timeline Duration")
    duration_hours = st.slider("Match Duration (Hours)", 
                              min_value=2, 
                              max_value=10, 
                              value=4, 
                              step=1,
                              help="Select how many hours of timeline to generate from match start time")
    st.caption(f"Will generate timeline for {duration_hours} hours from match start")
    
    # Generate Button
    if st.button("Generate Timeline"):
        try:
            # Parse match time
            match_time = parse_time_string(match_time_str)
            match_datetime = datetime.combine(match_date, match_time)
            
            # Generate timeline
            timeline_df, filepath = generate_and_save_timeline(
                match_datetime, 
                lat, 
                lon, 
                match_name,
                model_type,
                duration_hours
            )
            
            if timeline_df is not None:
                # Store in session state
                st.session_state.active_matches[match_name] = timeline_df
                
                # Update session state
                st.session_state.match_name = match_name
                st.session_state.match_date = match_date
                st.session_state.match_time_str = match_time_str
                st.session_state.latitude = lat
                st.session_state.longitude = lon
                
                st.success(f"Timeline generated for {match_name}!")
                st.rerun()
            else:
                st.warning("No valid timeline data generated.")
            
        except Exception as e:
            st.error(f"Error generating timeline: {str(e)}")
    
    # Section: Load Saved Timeline
    st.subheader("2. Load Saved Timeline")
    saved_timelines = get_saved_timelines()
    
    if not saved_timelines:
        st.write("No saved timelines found.")
    else:
        selected_file = st.selectbox("Select a timeline", [""] + saved_timelines)
        
        if st.button("Load Selected Timeline"):
            if selected_file:
                try:
                    file_path = os.path.join(RESULTS_DIR, selected_file)
                    df = pd.read_csv(file_path, comment='#')
                    
                    # Extract match name from filename
                    match_name = os.path.splitext(selected_file)[0]
                    
                    # Store in active matches
                    st.session_state.active_matches[match_name] = df
                    
                    # Load metadata and populate input fields
                    metadata = load_metadata_from_csv(file_path)
                    if metadata:
                        st.session_state.match_name = metadata.get("Match Name", st.session_state.match_name)
                        try:
                            st.session_state.match_date = datetime.strptime(metadata.get("Match Date"), "%Y-%m-%d")
                        except (ValueError, TypeError):
                            pass
                        st.session_state.match_time_str = metadata.get("Start Time", st.session_state.match_time_str)
                        try:
                            st.session_state.latitude = float(metadata.get("Latitude", st.session_state.latitude))
                            st.session_state.longitude = float(metadata.get("Longitude", st.session_state.longitude))
                        except (ValueError, TypeError):
                            pass
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            else:
                st.warning("Please select a file from the dropdown first.")

# --- Display Section ---
if st.session_state.active_matches:
    # Create tabs for each active match
    tabs = st.tabs(list(st.session_state.active_matches.keys()))
    
    # Display each match in its tab
    for tab, (match_name, df) in zip(tabs, st.session_state.active_matches.items()):
        with tab:
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.markdown(f"### {match_name} Timeline")
            with col2:
                if st.button("Close", key=f"close_{match_name}"):
                    st.session_state.closed_matches.add(match_name)
                    st.rerun()
            
            # Time Chart for Final Score
            st.markdown("#### Favorability Timeline")
            chart_data = df[['datetime', 'final_score']].copy()
            chart_data['datetime'] = pd.to_datetime(chart_data['datetime'])
            chart_data['above_zero'] = chart_data['final_score'].apply(lambda x: max(x, 0))
            chart_data['below_zero'] = chart_data['final_score'].apply(lambda x: min(x, 0))
            
            import altair as alt
            base = alt.Chart(chart_data).encode(x=alt.X('datetime:T', title='Time'))
            area_green = base.mark_area(opacity=0.5, color='green').encode(
                y=alt.Y('above_zero:Q', title='Final Score')
            )
            area_red = base.mark_area(opacity=0.5, color='red').encode(
                y=alt.Y('below_zero:Q', title='Final Score')
            )
            line = base.mark_line(color='black').encode(
                y=alt.Y('final_score:Q', title='Final Score')
            )
            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[5,5]).encode(y='y')
            
            chart = (area_green + area_red + line + zero_line).properties(
                width='container',
                height=300
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Display the color-coded table
            display_df_colored(df)
else:
    st.info("Generate a new timeline or load a saved one using the controls in the sidebar.")

# Clean up closed matches
for match_name in st.session_state.closed_matches:
    if match_name in st.session_state.active_matches:
        del st.session_state.active_matches[match_name]
st.session_state.closed_matches.clear()

# Add custom CSS to reduce margins and make the table full width
st.markdown("""
    <style>
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100vw;
    }
    .stDataFrameContainer {
        width: 100vw !important;
        min-width: 100vw !important;
    }
    </style>
""", unsafe_allow_html=True)
