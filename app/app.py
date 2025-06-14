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
def find_next_ssl_change(current_dt: datetime, lat: float, lon: float, nakshatra_df: pd.DataFrame):
    """
    Calculates the exact datetime of the next Sub-Sub Lord change.
    This is a more efficient approach than iterating minute-by-minute.
    """
    # 1. Get current moon position
    jd = swe.julday(current_dt.year, current_dt.month, current_dt.day, current_dt.hour + current_dt.minute/60 + current_dt.second/3600 - 5.5)
    pos = swe.calc_ut(jd, swe.MOON)
    moon_long = pos[0][0]
    moon_speed_deg_per_day = pos[0][3]
    moon_speed_deg_per_sec = moon_speed_deg_per_day / (24 * 3600)

    # 2. Find the current SSL's end degree
    # Handle the wrap-around for the final entry in the table
    current_row = nakshatra_df[(nakshatra_df['Start_Degree'] <= moon_long) & (nakshatra_df['End_Degree'] > moon_long)]
    if current_row.empty and moon_long > 359.99:
         current_row = nakshatra_df.iloc[-1:]

    if current_row.empty:
        # Fallback if something goes wrong, though it shouldn't
        return current_dt + timedelta(minutes=1)
        
    end_degree = current_row.iloc[0]['End_Degree']

    # 3. Calculate time to reach the end degree
    if end_degree < moon_long: # Handles the 360 -> 0 degree crossover
        degrees_to_travel = (360 - moon_long) + end_degree
    else:
        degrees_to_travel = end_degree - moon_long
    
    # Add a tiny amount to degrees to ensure we are PAST the boundary
    degrees_to_travel += 0.00001

    if moon_speed_deg_per_sec <= 0: # Should not happen
        return current_dt + timedelta(minutes=1)

    seconds_to_change = degrees_to_travel / moon_speed_deg_per_sec
    
    # Return the precise time of the next change
    return current_dt + timedelta(seconds=seconds_to_change)


def generate_and_save_timeline(start_dt, lat, lon, match_name):
    """Generates the timeline DataFrame and saves it to a CSV."""
    timeline_data = []
    current_dt = start_dt
    end_dt = start_dt + timedelta(hours=4) # 4-hour match window
    
    # Use the pre-loaded Nakshatra data
    nakshatra_df = NAKSHATRA_DF
    
    with st.spinner(f"Generating timeline from {start_dt.strftime('%H:%M:%S')} to {end_dt.strftime('%H:%M:%S')}"):
        while current_dt < end_dt:
            chart = generate_kp_chart(current_dt, lat, lon, nakshatra_df)
            if "error" in chart:
                st.error(f"Error generating chart for {current_dt}: {chart['error']}")
                # On error, increment by a minute to avoid getting stuck in a loop
                current_dt += timedelta(minutes=1)
                continue

            favorability_data = evaluate_favorability(chart, nakshatra_df)
            
            timeline_row = {
                "datetime": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "moon_star_lord": chart.get("moon_star_lord"),
                "sub_lord": chart.get("moon_sub_lord"),
                "sub_sub_lord": chart.get("moon_sub_sub_lord"),
                "moon_sl_score": favorability_data.get("moon_sl_score"),
                "moon_sub_score": favorability_data.get("moon_sub_score"),
                "moon_ssl_score": favorability_data.get("moon_ssl_score"),
                "final_score": favorability_data.get("final_score"),
            }
            timeline_data.append(timeline_row)
            
            # Intelligently jump to the next SSL change time
            current_dt = find_next_ssl_change(current_dt, lat, lon, nakshatra_df)


    if not timeline_data:
        st.warning("Could not generate any timeline data.")
        return None

    df = pd.DataFrame(timeline_data)
        
    # Set the verdict based on the final score
    df['verdict'] = df['final_score'].apply(get_verdict_label)
    
    # Metadata for saving in the file
    metadata_lines = [
        f"# Match: {match_name}\n",
        f"# Date: {start_dt.strftime('%Y-%m-%d')}\n",
        f"# Start Time: {start_dt.strftime('%H:%M:%S')}\n",
        f"# Location: {lat}, {lon}\n"
    ]
    
    # Save to CSV
    filename = f"{match_name.replace(' ', '_')}_{start_dt.strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, 'w') as f:
        f.writelines(metadata_lines)
    
    df.to_csv(filepath, index=False, mode='a') # Append after metadata
    
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

# --- UI Layout ---

st.title("🪐 KP Cricket Predictor Dashboard")

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
if 'df_to_display' not in st.session_state:
    st.session_state.df_to_display = None
if 'file_being_displayed' not in st.session_state:
    st.session_state.file_being_displayed = None

# --- Sidebar ---
st.sidebar.title("Controls")

# Section: Generate New Timeline
st.sidebar.header("1. Generate New Timeline")
st.sidebar.text_input("Match Title", key="match_name")
st.sidebar.date_input("Match Date (YYYY-MM-DD)", key="match_date")
st.sidebar.text_input("Match Start Time (HH:MM, 24-hr)", key="match_time_str")

# --- Location Search ---
st.sidebar.markdown("---")
location_query = st.sidebar.text_input("Search for Match Venue (e.g., 'Mumbai, India')")
if st.sidebar.button("Find Coordinates", key="find_coords_btn"):
    if location_query:
        with st.spinner("Finding location..."):
            coords = get_coordinates(location_query)
        if coords:
            st.session_state.latitude, st.session_state.longitude = coords
            st.sidebar.success(f"Found: Lat {coords[0]:.4f}, Lon {coords[1]:.4f}")
            # Use st.rerun() to immediately update the number_input widgets below
            st.rerun()
        else:
            st.sidebar.error("Location not found or service unavailable.")
    else:
        st.sidebar.warning("Please enter a location to search for.")
st.sidebar.markdown("---")

st.sidebar.number_input("Latitude", key="latitude", format="%.4f")
st.sidebar.number_input("Longitude", key="longitude", format="%.4f")

if st.sidebar.button("Generate Timeline", type="primary", key="generate_timeline_btn"):
    try:
        # Use a timezone-aware object for start_datetime
        # This is a placeholder; a more robust app would let the user specify the timezone
        local_tz = pytz.timezone("Asia/Kolkata") 
        match_time = datetime.strptime(st.session_state.match_time_str, "%H:%M").time()
        
        # Combine date and time, then localize
        naive_datetime = datetime.combine(st.session_state.match_date, match_time)
        start_datetime = local_tz.localize(naive_datetime)

        with st.spinner("Generating astrological timeline... This is now much faster!"):
            timeline_df, filepath = generate_and_save_timeline(
                start_datetime, 
                st.session_state.latitude, 
                st.session_state.longitude, 
                st.session_state.match_name
            )
        
        if timeline_df is not None:
            st.success(f"Generated and saved: {os.path.basename(filepath)}")
            st.session_state.df_to_display = timeline_df
            st.session_state.file_being_displayed = os.path.basename(filepath)
            st.rerun()
        else:
            st.warning("No valid timeline data generated.")

    except Exception as e:
        st.error(f"Error during generation: {e}")

# Section: View Saved Timeline
st.sidebar.header("2. View Saved Timeline")
saved_timelines = get_saved_timelines()

if not saved_timelines:
    st.sidebar.write("No saved timelines found.")
else:
    selected_file = st.sidebar.selectbox("Select a timeline", [""] + saved_timelines)
    
    if st.sidebar.button("Load Selected Timeline"):
        if selected_file:
            try:
                file_path = os.path.join(RESULTS_DIR, selected_file)
                df = pd.read_csv(file_path, comment='#')
                st.session_state.df_to_display = df
                st.session_state.file_being_displayed = selected_file
                
                # Load metadata and populate input fields
                metadata = load_metadata_from_csv(file_path)
                if metadata:
                    st.session_state.match_name = metadata.get("Match Name", st.session_state.match_name)
                    try:
                        st.session_state.match_date = datetime.strptime(metadata.get("Match Date"), "%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass # Keep default on error
                    st.session_state.match_time_str = metadata.get("Start Time", st.session_state.match_time_str)
                    try:
                        st.session_state.latitude = float(metadata.get("Latitude", st.session_state.latitude))
                        st.session_state.longitude = float(metadata.get("Longitude", st.session_state.longitude))
                    except (ValueError, TypeError):
                        pass # Keep default on error
                st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.sidebar.warning("Please select a file from the dropdown first.")

# --- Display Section ---
if 'df_to_display' in st.session_state and st.session_state.df_to_display is not None:
    df = st.session_state.df_to_display

    # --- Time Chart for Final Score ---
    st.markdown("### Favorability Timeline")
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
        y='below_zero:Q'
    )
    line = base.mark_line(color='black', strokeWidth=2).encode(
        y='final_score:Q'
    )
    zero_line = base.mark_rule(color='gray', strokeDash=[4,4]).encode(y=alt.datum(0))
    chart = (area_green + area_red + line + zero_line).properties(height=300, width='container')
    st.altair_chart(chart, use_container_width=True)

    # --- Table ---
    st.markdown("### Timeline Table")
    display_df_colored(df)
else:
    st.info("Generate a new timeline or load a saved one using the controls in the sidebar.")

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
