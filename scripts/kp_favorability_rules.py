from scripts.chart_generator import generate_kp_chart

# Dummy house significators (should be replaced with KP logic)
def get_significator_houses(planet: str, chart: dict) -> list:
    house_map = {
        "Sun": [1, 5, 9],
        "Moon": [4, 8, 12],
        "Mars": [3, 6, 11],
        "Mercury": [2, 10],
        "Jupiter": [1, 7],
        "Venus": [4, 11],
        "Saturn": [6, 8],
        "Rahu": [5, 12],
        "Ketu": [3, 9]
    }
    return house_map.get(planet, [])

# Dummy conjunctions
def get_conjunctions(planet: str, chart: dict) -> list:
    # Replace with actual planet positions logic
    return []

# Dummy ruling planets
def get_ruling_planets(dt, lat, lon) -> list:
    # This could return Lagna lord, Moon sign lord, day lord, etc.
    return ["Moon", "Venus"]

# Chart creation wrapper
def create_chart(dt, lat, lon):
    return generate_kp_chart(dt, lat, lon)

# Dummy evaluator function
def evaluate_period(data: dict, asc_sign: str) -> int:
    score = 0

    favored_houses = [1, 3, 6, 10, 11]
    weak_houses = [5, 8, 12]

    for planet in ["Sub_Lord", "Sub_Sub_Lord"]:
        houses = data.get(f"{planet}_Houses", [])
        for h in houses:
            if h in favored_houses:
                score += 10
            elif h in weak_houses:
                score -= 10

    # Apply bonus for ruling planet match
    if data["Sub_Lord"] in data["Ruling_Planets"]:
        score += 15
    if data["Sub_Sub_Lord"] in data["Ruling_Planets"]:
        score += 10

    # Penalties
    if data["Sub_Lord_Retrograde"]:
        score -= 5
    if data["Sub_Sub_Lord_Retrograde"]:
        score -= 3

    if data["Sub_Lord_Combust"]:
        score -= 7
    if data["Sub_Sub_Lord_Combust"]:
        score -= 4

    return max(-100, min(100, score))  # Clamp between -100 and +100
