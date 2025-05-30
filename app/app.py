import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("PYTHON PATH:", sys.path)
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd


from scripts.chart_generator import create_chart
from scripts.kp_favorability_rules import (
    get_significator_houses,
    get_conjunctions,
    get_ruling_planets,
    evaluate_period
)

# UI Title
st.title("🧠 KP Astrology Cricket Match Predictor")
st.markdown("Enter match start time and location to generate Sub Lord favorability timeline.")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    match_date = st.date_input("Match Date", value=datetime.today())
    match_time = st.time_input("Start Time", value=datetime.now().time())
with col2:
    latitude = st.number_input("Latitude", value=23.0225)
    longitude = st.number_input("Longitude", value=72.5714)

if st.button("🔍 Generate Timeline"):
    from scripts.kp_sub_lord_calculator import get_moon_periods  # your existing function

    start_dt = datetime.combine(match_date, match_time)
    end_dt = start_dt + timedelta(hours=4)

    # Get moon sub lord periods
    periods = get_moon_periods(start_dt, end_dt, latitude, longitude)

    result_rows = []
    for p in periods:
        dt = p['datetime']
        chart = create_chart(dt, latitude, longitude)
        asc_sign = chart.get('Asc').sign

        sub_lord = p['Sub_Lord']
        sub_sub_lord = p['Sub_Sub_Lord']

        period_data = {
            'Sub_Lord': sub_lord,
            'Sub_Sub_Lord': sub_sub_lord,
            'Sub_Lord_Houses': get_significator_houses(sub_lord, chart),
            'Sub_Sub_Lord_Houses': get_significator_houses(sub_sub_lord, chart),
            'Sub_Lord_Retrograde': p.get('Sub_Lord_Retrograde', False),
            'Sub_Sub_Lord_Retrograde': p.get('Sub_Sub_Lord_Retrograde', False),
            'Sub_Lord_Combust': p.get('Sub_Lord_Combust', False),
            'Sub_Sub_Lord_Combust': p.get('Sub_Sub_Lord_Combust', False),
            'Sub_Lord_Conjunct': get_conjunctions(sub_lord, chart),
            'Sub_Sub_Lord_Conjunct': get_conjunctions(sub_sub_lord, chart),
            'Sub_Lord_Star_Lord': p.get('Sub_Lord_Star_Lord'),
            'Sub_Sub_Lord_Star_Lord': p.get('Sub_Sub_Lord_Star_Lord'),
            'Ruling_Planets': get_ruling_planets(dt, latitude, longitude)
        }

        result = evaluate_period(period_data, asc_sign)

        result_rows.append({
            'Time': dt.strftime('%H:%M:%S'),
            'Sub Lord': sub_lord,
            'Sub-Sub Lord': sub_sub_lord,
            'Ascendant': asc_sign,
            'Favorability': result
        })

    df = pd.DataFrame(result_rows)
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Timeline CSV", csv, file_name="kp_favorability_timeline.csv")
