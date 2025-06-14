import swisseph as swe
import datetime
import pandas as pd

# Zodiac signs for ascendant sign derivation
SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# KP Planets
PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,  # Mean Rahu
}

def get_sign(moon_long):
    index = int(moon_long // 30) % 12
    return SIGNS[index]

def generate_kp_chart(dt: datetime.datetime, lat: float, lon: float, nakshatra_df: pd.DataFrame = None):
    try:
        # Calculate Julian Day with IST offset (-5.5 hrs)
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute / 60 + dt.second / 3600 - 5.5)

        # Get house cusps (Placidus system is common for KP)
        houses, ascmc = swe.houses(jd, lat, lon, b'P')
        asc_deg = ascmc[0]

        # Get planet longitudes
        planets_data = {}
        for name, p_id in PLANETS.items():
            pos = swe.calc_ut(jd, p_id)
            planets_data[name] = {
                "longitude": round(pos[0][0] % 360, 4),
                "speed": pos[0][3],
                "retrograde": pos[0][3] < 0
            }
        
        # Ketu is 180 degrees opposite Rahu
        rahu_long = planets_data["Rahu"]["longitude"]
        ketu_long = (rahu_long + 180) % 360
        planets_data["Ketu"] = {
            "longitude": round(ketu_long, 4),
            "speed": planets_data["Rahu"]["speed"], # Ketu moves with Rahu
            "retrograde": True # Rahu/Ketu are always retrograde
        }

        # Load the mapping CSV if not provided
        if nakshatra_df is None:
            nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")

        # Match moon longitude
        moon_long = planets_data["Moon"]["longitude"]
        match = nakshatra_df[(nakshatra_df['Start_Degree'] <= moon_long) & (nakshatra_df['End_Degree'] > moon_long)]

        if match.empty:
            # Handle edge case for 360 degrees
            if moon_long >= 359.999:
                match = nakshatra_df.iloc[-1:]
            if match.empty:
                return {"error": f"Moon longitude {moon_long:.4f} not found in any mapped range."}

        row = match.iloc[0]

        return {
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ascendant_degree": round(asc_deg, 4),
            "houses": houses,
            "planets": planets_data,
            "moon_longitude": moon_long,
            "moon_nakshatra": row["Nakshatra"],
            "moon_pada": int(row["Pada"]),
            "moon_sub_lord": row["Sub_Lord"],
            "moon_sub_sub_lord": row["Sub_Sub_Lord"],
            "moon_sign": row["Sign"],
            "moon_sign_lord": row["Sign_Lord"],
            "moon_star_lord": row["Star_Lord"]
        }

    except Exception as e:
        return {"error": f"Chart generation failed: {str(e)}"}
