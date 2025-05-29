import swisseph as swe
import datetime

# Set path to ephemeris data if needed (or comment out if already working)
# swe.set_ephe_path('/path/to/ephemeris')

def generate_kp_chart(dt: datetime.datetime, lat: float, lon: float):
    """
    Generates basic KP chart information for a given datetime and location.
    """
    # Julian Day calculation (adjust for IST: -5.5h offset from UTC)
    jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60 - 5.5)

    # Get Ascendant (Lagna degree)
    houses, ascmc = swe.houses(jd, lat, lon)
    asc_deg = ascmc[0]

    # Get Moon longitude
    moon_long = swe.calc_ut(jd, swe.MOON)[0][0]

    # Print raw output for now
    return {
        "datetime": dt.strftime("%Y-%m-%d %H:%M"),
        "ascendant_degree": round(asc_deg, 4),
        "moon_longitude": round(moon_long, 4)
    }

# Example usage
if __name__ == "__main__":
    # 2nd May 2025, 19:30 IST, Ahmedabad
    dt = datetime.datetime(2025, 5, 2, 19, 30)
    lat = 23.0225
    lon = 72.5714

    result = generate_kp_chart(dt, lat, lon)
    print(result)
