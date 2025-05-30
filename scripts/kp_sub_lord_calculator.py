import pandas as pd
from datetime import datetime, timedelta
from scripts.chart_generator import generate_kp_chart

def get_moon_periods(start_dt: datetime, end_dt: datetime, lat: float, lon: float):
    current_dt = start_dt
    timeline = []

    last_sub_lord = last_sub_sub_lord = None

    while current_dt <= end_dt:
        try:
            chart = generate_kp_chart(current_dt, lat, lon)
            sub_lord = chart["sub_lord"]
            sub_sub_lord = chart["sub_sub_lord"]

            if (sub_lord != last_sub_lord) or (sub_sub_lord != last_sub_sub_lord):
                timeline.append({
                    "datetime": current_dt,
                    "Sub_Lord": sub_lord,
                    "Sub_Sub_Lord": sub_sub_lord,
                    "Sub_Lord_Star_Lord": None,
                    "Sub_Sub_Lord_Star_Lord": None,
                    "Sub_Lord_Retrograde": False,
                    "Sub_Sub_Lord_Retrograde": False,
                    "Sub_Lord_Combust": False,
                    "Sub_Sub_Lord_Combust": False,
                })
                last_sub_lord, last_sub_sub_lord = sub_lord, sub_sub_lord
        except Exception as e:
            print(f"Error at {current_dt}: {e}")

        current_dt += timedelta(seconds=30)

    return timeline
