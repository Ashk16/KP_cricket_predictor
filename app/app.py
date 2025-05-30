import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

# ✅ Fix import path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ✅ Local imports
from scripts.kp_favorability_rules import (
    create_chart,
    get_significator_houses,
    get_conjunctions,
    get_ruling_planets,
    evaluate_period
)
from scripts.kp_sub_lord_calculator import get_moon_periods

# 🧠 App Title
st.title("🏏 KP Astrology Match Predictor")
st.markdown("Enter match details to generate Sub Lord favorability timeline.")

# 📅 Match Inputs
col1, col2 = st.columns(2)

with col1:
    match_date = st.date_input(
        "Match Date", value=st.session_state.get("match_date", datetime.today().date())
    )
    st.session_state["match_date"] = match_date

    default_time = st.session_state.get("match_time", datetime.now().time())
    match_time = st.time_input("Match Time", value=default_time)
    st.session_state["match_time"] = match_time

with col2:
    stadium_name = st.text_input("Stadium (Optional)", placeholder="E.g. Eden Gardens, Kolkata")

# 🌐 Manual Latitude & Longitude
col3, col4 = st.columns(2)
with col3:
    latitude = st.number_input("Latitude", format="%.6f", value=23.022500)
with col4:
    longitude = st.number_input("Longitude", format="%.6f", value=72.571400)

# 🔍 Generate Timeline
if st.button("Generate Timeline"):
    start_dt = datetime.combine(match_date, match_time)
    end_dt = start_dt + timedelta(hours=4)

    periods = get_moon_periods(start_dt, end_dt, latitude, longitude)

    result_rows = []
    for p in periods:
        dt = p["datetime"]
        chart = create_chart(dt, latitude, longitude)
        asc = chart.get("Asc").sign

        sl = p["Sub_Lord"]
        ssl = p["Sub_Sub_Lord"]

        data = {
            "Sub_Lord": sl,
            "Sub_Sub_Lord": ssl,
            "Sub_Lord_Houses": get_significator_houses(sl, chart),
            "Sub_Sub_Lord_Houses": get_significator_houses(ssl, chart),
            "Sub_Lord_Retrograde": p.get("Sub_Lord_Retrograde", False),
            "Sub_Sub_Lord_Retrograde": p.get("Sub_Sub_Lord_Retrograde", False),
            "Sub_Lord_Combust": p.get("Sub_Lord_Combust", False),
            "Sub_Sub_Lord_Combust": p.get("Sub_Sub_Lord_Combust", False),
            "Sub_Lord_Conjunct": get_conjunctions(sl, chart),
            "Sub_Sub_Lord_Conjunct": get_conjunctions(ssl, chart),
            "Sub_Lord_Star_Lord": p.get("Sub_Lord_Star_Lord"),
            "Sub_Sub_Lord_Star_Lord": p.get("Sub_Sub_Lord_Star_Lord"),
            "Ruling_Planets": get_ruling_planets(dt, latitude, longitude),
        }

        favor = evaluate_period(data, asc)

        result_rows.append({
            "Time": dt.strftime("%H:%M"),
            "Sub Lord": sl,
            "Sub-Sub Lord": ssl,
            "Ascendant": asc,
            "Favorability": favor,
        })

    df = pd.DataFrame(result_rows)
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "kp_favorability_timeline.csv")
