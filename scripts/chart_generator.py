import swisseph as swe
import datetime
import pandas as pd

# Set ephemeris path if needed
# swe.set_ephe_path("/path/to/ephemeris")

# List of 27 Nakshatras
nakshatras = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira",
    "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha",
    "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
    "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
    "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

def get_nakshatra_pada(moon_long):
    nakshatra_index = int(moon_long // 13.3333)
    nakshatra_name = nakshatras[nakshatra_index]
    pada_number = int((moon_long % 13.3333) // 3.3333) + 1
    return nakshatra_name, pada_number

def generate_kp_chart(dt: datetime.datetime, lat: float, lon: float):
    jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60 - 5.5)
    houses, ascmc = swe.houses(jd, lat, lon)
    asc_deg = ascmc[0]

    moon_long = swe.calc_ut(jd, swe.MOON)[0][0]
    nakshatra, pada = get_nakshatra_pada(moon_long)

    # Load CSV and find matching row
    df = pd.read_csv("config/nakshatra_sub_lords.csv")
    row = df[(df["Nakshatra"] == nakshatra) & (df["Pada"] == pada)].iloc[0]

    return {
        "datetime": dt.strftime("%Y-%m-%d %H:%M"),
        "ascendant_degree": round(asc_deg, 4),
        "moon_longitude": round(moon_long, 4),
        "nakshatra": nakshatra,
        "pada": pada,
        "sub_lord": row["Sub_Lord"],
        "sub_sub_lord": row["Sub_Sub_Lord"]
    }

# Example usage
if __name__ == "__main__":
    dt = datetime.datetime(2025, 5, 2, 19, 30)
    lat = 23.0225
    lon = 72.5714

    result = generate_kp_chart(dt, lat, lon)
    print(result)
