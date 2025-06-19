import sys
import os
sys.path.append(os.path.dirname(__file__))
from chart_generator import generate_kp_chart
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from kp_contradiction_system import KPContradictionSystem

# Zodiac signs
SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Sign rulers
SIGN_RULERS = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
    "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
    "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
}

# --- Configuration Constants ---

# 1. House Weights for Cricket (Competitive Game) - V2 (Refined for Competition)
# Favorable for Ascendant (batting team)
HOUSE_WEIGHTS_ASC = {
    1: 10,   # Self, physical body, initiative
    2: 8,    # Wealth, accumulation of runs (Increased)
    3: 10,   # Courage, enterprise, aggressive batting (Increased)
    6: 12,   # Victory over opponents (Increased)
    10: 8,   # High-status actions, profession
    11: 12   # Gains, fulfillment of desires (Most Important)
}

# Favorable for Descendant (bowling/opposing team) - Weights are positive
HOUSE_WEIGHTS_DESC = {
    4: 4,   # Home ground / Opposition comfort
    5: 8,   # Loss of gains for ascendant (11th from 7th)
    7: 10,  # The opponent
    8: 10,  # Obstacles, transformations, collapse
    9: 4,   # Luck of the opponent
    12: 12  # Losses, expenditure of energy for ascendant
}

# 2. Planet Natural Strength (A-political Rank)
PLANET_STRENGTH = {
    "Sun": 1.2, "Moon": 1.2, "Saturn": 1.1, "Jupiter": 1.1,
    "Mars": 1.0, "Venus": 1.0, "Mercury": 0.9,
    "Rahu": 1.0, "Ketu": 1.0
}

# 3. Planet Dignity Modifiers
DIGNITY_MULTIPLIERS = {
    "Exaltation": 1.5,
    "Moolatrikona": 1.3,
    "Own House": 1.2,
    "Neutral": 1.0,
    "Debilitation": 0.6
}

# 4. Aspect System (Nuanced)
PLANET_NATURES = {
    "benefic": ["Jupiter", "Venus"],
    "malefic": ["Saturn", "Mars", "Sun", "Rahu", "Ketu"],
    "neutral": ["Moon", "Mercury"]
}

ASPECT_IMPACT = {
    "trine":      {"benefic": 1.20, "malefic": 1.05, "neutral": 1.10},
    "conjunction":{"benefic": 1.15, "malefic": 0.85, "neutral": 1.00},
    "square":     {"benefic": 0.95, "malefic": 0.80, "neutral": 0.90},
    "opposition": {"benefic": 0.90, "malefic": 0.75, "neutral": 0.85}
}

# 5. Other State Modifiers
COMBUST_MODIFIER = 0.5        # Severely weakened when too close to the Sun

# Planet-specific retrograde modifiers
RETROGRADE_MODIFIERS = {
    "Mercury": 0.85, # Disruptive, weakens positive outcomes
    "Venus": 0.90,   # Turns benefic nature inward, less effective
    "Mars": 1.15,    # Intensifies aggression and conflict
    "Jupiter": 0.90, # Expansive nature is restricted
    "Saturn": 1.20,  # Delays and obstacles become more pronounced
}

# 6. Ruling Planet Aspect Bonus
RULING_PLANET_ASPECT_BONUS = 1.25 # Aspects from RPs are 25% more impactful

# 7. Lord Hierarchy Weights (Star > Sub > Sub-Sub) - CORRECTED KEYS
LORD_WEIGHTS = {
    "sl": 0.50, # Corresponds to moon_sl
    "sub": 0.30, # Corresponds to moon_sub
    "ssl": 0.20  # Corresponds to moon_ssl
}

# 8. Hierarchical weights for planet scoring
HIERARCHICAL_WEIGHTS = {
    "planet": 0.25,
    "star_lord": 0.40,
    "slsl": 0.35 # Star Lord's Star Lord
}

def get_sign(longitude: float) -> str:
    return SIGNS[int(longitude // 30)]

def get_house_for_planet(planet_long: float, houses: list) -> int:
    """Finds which house a planet is in."""
    for i in range(12):
        cusp_start = houses[i]
        cusp_end = houses[(i + 1) % 12]
        if cusp_start < cusp_end:
            if cusp_start <= planet_long < cusp_end:
                return i + 1
        else:  # Wrap around 360 degrees
            if planet_long >= cusp_start or planet_long < cusp_end:
                return i + 1
    return -1 # Should not happen

def get_astro_details(longitude: float, df: pd.DataFrame) -> dict:
    """Gets nakshatra, star lord, etc., for a given longitude."""
    match = df[(df['Start_Degree'] <= longitude) & (df['End_Degree'] > longitude)]
    if match.empty:
        if longitude >= 359.999:
            match = df.iloc[-1:]
        if match.empty:
            return None
    return match.iloc[0].to_dict()

def get_significator_houses(planet: str, chart: dict) -> list:
    """Calculates house significations for a planet."""
    if planet not in chart["planets"]:
        return []

    planet_long = chart["planets"][planet]["longitude"]
    houses = chart["houses"]
    
    # 1. House occupied by the planet
    occupied_house = get_house_for_planet(planet_long, houses)
    
    significators = {occupied_house}
    
    # 2. Houses ruled by the planet
    house_lords = {}
    for i in range(12):
        house_cusp = houses[i]
        sign_of_house = get_sign(house_cusp)
        lord = SIGN_RULERS[sign_of_house]
        if lord not in house_lords:
            house_lords[lord] = []
        house_lords[lord].append(i + 1)

    if planet in house_lords:
        for house in house_lords[planet]:
            significators.add(house)
            
    return sorted(list(significators))

def get_conjunctions(planet: str, chart: dict, orb: float = 5.0) -> list:
    """Finds planets in conjunction with a given planet."""
    if planet not in chart["planets"]:
        return []

    conjunct_planets = []
    planet_long = chart["planets"][planet]["longitude"]
    planet_house = get_house_for_planet(planet_long, chart["houses"])

    for other_planet, other_planet_data in chart["planets"].items():
        if other_planet == planet:
            continue

        other_planet_long = other_planet_data["longitude"]
        other_planet_house = get_house_for_planet(other_planet_long, chart["houses"])

        if other_planet_house == planet_house:
            # Check for degree difference
            diff = abs(planet_long - other_planet_long)
            if min(diff, 360 - diff) <= orb:
                conjunct_planets.append(other_planet)
    
    return conjunct_planets

def get_ruling_planets(chart: dict, nakshatra_df: pd.DataFrame) -> List[str]:
    """Determines the KP Ruling Planets for a given chart."""
    if "error" in chart: return []
    
    # 1. Get Lords of Ascendant and Moon
    asc_details = get_astro_details(chart["ascendant_degree"], nakshatra_df)
    moon_details = get_astro_details(chart["moon_longitude"], nakshatra_df)

    # 2. Get Day Lord
    datetime_obj = datetime.strptime(chart["datetime"], "%Y-%m-%d %H:%M:%S")
    day_lords = ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Sun"]
    day_lord = day_lords[datetime_obj.weekday()]

    ruling_planets = {
        asc_details.get("Sign_Lord"),
        asc_details.get("Star_Lord"),
        moon_details.get("Sign_Lord"),
        moon_details.get("Star_Lord"),
        day_lord
    }
    # Remove None if any lord was not found
    ruling_planets.discard(None)
    
    # 3. Check for Rahu/Ketu agency
    final_rps = set(ruling_planets)
    for rp in ruling_planets:
        # Check if Rahu/Ketu are in the sign of an RP
        rp_sign = get_sign(chart["planets"][rp]["longitude"])
        rahu_sign = get_sign(chart["planets"]["Rahu"]["longitude"])
        ketu_sign = get_sign(chart["planets"]["Ketu"]["longitude"])
        
        if rahu_sign == rp_sign: final_rps.add("Rahu")
        if ketu_sign == rp_sign: final_rps.add("Ketu")

    return list(final_rps)

def is_combust(planet: str, chart: dict) -> bool:
    """Checks if a planet is combust."""
    if planet == "Sun" or planet not in chart["planets"]:
        return False

    sun_long = chart["planets"]["Sun"]["longitude"]
    planet_long = chart["planets"][planet]["longitude"]

    orb = 12 if planet == "Moon" else 8.5
    
    diff = abs(sun_long - planet_long)
    return min(diff, 360 - diff) <= orb

# Chart creation wrapper
def create_chart(dt, lat, lon):
    return generate_kp_chart(dt, lat, lon)

# --- Dignity (Strength) System ---
PLANET_DIGNITIES = {
    "Sun": {"exaltation": "Aries", "debilitation": "Libra", "moolatrikona": "Leo", "own_house": ["Leo"]},
    "Moon": {"exaltation": "Taurus", "debilitation": "Scorpio", "moolatrikona": "Taurus", "own_house": ["Cancer"]},
    "Mars": {"exaltation": "Capricorn", "debilitation": "Cancer", "moolatrikona": "Aries", "own_house": ["Aries", "Scorpio"]},
    "Mercury": {"exaltation": "Virgo", "debilitation": "Pisces", "moolatrikona": "Virgo", "own_house": ["Gemini", "Virgo"]},
    "Jupiter": {"exaltation": "Cancer", "debilitation": "Capricorn", "moolatrikona": "Sagittarius", "own_house": ["Sagittarius", "Pisces"]},
    "Venus": {"exaltation": "Pisces", "debilitation": "Virgo", "moolatrikona": "Libra", "own_house": ["Taurus", "Libra"]},
    "Saturn": {"exaltation": "Libra", "debilitation": "Aries", "moolatrikona": "Aquarius", "own_house": ["Capricorn", "Aquarius"]},
    # Rahu and Ketu dignities are complex and debated; treated as neutral for now.
    "Rahu": {}, 
    "Ketu": {}
}

def get_planet_dignity_strength(planet: str, chart: dict) -> float:
    """Calculates a strength multiplier for a planet based on its dignity."""
    if not planet or planet not in chart["planets"]:
        return 1.0 # Neutral strength if planet not found

    planet_sign = get_sign(chart["planets"][planet]["longitude"])
    dignities = PLANET_DIGNITIES.get(planet, {})

    if not dignities:
        return 1.0 # Neutral for Rahu/Ketu

    if dignities.get("exaltation") == planet_sign:
        return DIGNITY_MULTIPLIERS["Exaltation"]
    if dignities.get("debilitation") == planet_sign:
        return DIGNITY_MULTIPLIERS["Debilitation"]
    if dignities.get("moolatrikona") == planet_sign:
        return DIGNITY_MULTIPLIERS["Moolatrikona"]
    if planet_sign in dignities.get("own_house", []):
        return DIGNITY_MULTIPLIERS["Own House"]
        
    return DIGNITY_MULTIPLIERS["Neutral"]

# --- Natural Rank System ---
# Based on traditional astrological hierarchy and influence
NATURAL_RANK_MODIFIER = {
    "Sun": 1.3,    # Increased from 1.2
    "Moon": 1.25,  # Increased from 1.15
    "Saturn": 1.2, # Increased from 1.1
    "Jupiter": 1.2, # Increased from 1.1
    "Mars": 1.1,   # Increased from 1.0
    "Venus": 1.1,  # Increased from 1.0
    "Mercury": 1.0, # Increased from 0.9
    "Rahu": 1.0,   # Kept same
    "Ketu": 1.0    # Kept same
}

# --- Aspects System ---
ASPECTS = {
    "conjunction": {"angle": 0, "orb": 10, "type": "neutral"},
    "opposition": {"angle": 180, "orb": 10, "type": "hard"},
    "trine": {"angle": 120, "orb": 8, "type": "soft"},
    "square": {"angle": 90, "orb": 8, "type": "hard"},
}

def get_planet_aspects(planet: str, chart: dict) -> list:
    """
    Calculate aspects received by a planet.
    Returns a list of dictionaries containing aspecting planet and aspect type.
    """
    if not planet or planet not in chart["planets"]:
        return []

    aspects = []
    planet_long = chart["planets"][planet]["longitude"]

    for other_planet, other_data in chart["planets"].items():
        if other_planet == planet:
            continue

        other_long = other_data["longitude"]
        diff = abs(planet_long - other_long)
        diff = min(diff, 360 - diff)  # Get smallest angle

        # Check for different aspect types with their specific orbs
        if diff <= ASPECTS["conjunction"]["orb"]:  # Conjunction
            aspects.append({
                "planet": other_planet,
                "type": "Conjunction",
                "angle": diff
            })
        elif abs(diff - 180) <= ASPECTS["opposition"]["orb"]:  # Opposition
            aspects.append({
                "planet": other_planet,
                "type": "Opposition",
                "angle": diff
            })
        elif abs(diff - 120) <= ASPECTS["trine"]["orb"]:  # Trine
            aspects.append({
                "planet": other_planet,
                "type": "Trine",
                "angle": diff
            })
        elif abs(diff - 90) <= ASPECTS["square"]["orb"]:  # Square
            aspects.append({
                "planet": other_planet,
                "type": "Square",
                "angle": diff
            })

    return aspects

def get_planet_aspect_modifier(planet: str, chart: dict) -> float:
    """
    Calculate aspect modifier for a planet.
    Returns a single modifier that can be positive (favoring asc) or negative (favoring desc)
    """
    if not planet or planet not in chart["planets"]:
        return 0.0

    # Get aspects received by the planet
    aspects = get_planet_aspects(planet, chart)
    if not aspects:
        return 0.0

    # Initialize total modifier
    total_modifier = 0.0

    for aspect in aspects:
        aspecting_planet = aspect["planet"]
        aspect_type = aspect["type"]
        
        # Get the weight and dignity of the aspecting planet
        aspecting_weight = PLANET_CRICKET_WEIGHTS.get(aspecting_planet, 1.0)
        aspecting_dignity = get_planet_dignity_strength(aspecting_planet, chart)
        
        # Get the houses ruled by the aspecting planet
        aspecting_houses = get_significator_houses(aspecting_planet, chart)
        
        # Calculate base aspect strength based on type and orb
        orb_value = ASPECTS[aspect_type.lower()]["orb"]
        orb_factor = max(0, (1 - (aspect["angle"] / orb_value))**2) # Non-linear orb factor
        
        if aspect_type == "Conjunction":
            base_strength = 0.35 # Increased from 0.3
        elif aspect_type == "Opposition":
            base_strength = 0.3  # Increased from 0.25
        elif aspect_type == "Trine":
            base_strength = 0.25 # Increased from 0.2
        elif aspect_type == "Square":
            base_strength = 0.2 # Increased from 0.15
        else:
            base_strength = 0.1

        # Calculate aspecting planet's influence on ascendant and descendant
        asc_influence = sum(ASC_HOUSE_WEIGHTS.get(h, 0) for h in aspecting_houses)
        desc_influence = sum(DESC_HOUSE_WEIGHTS.get(h, 0) for h in aspecting_houses)
        
        # Calculate net influence (positive favors asc, negative favors desc)
        net_influence = asc_influence - desc_influence
        
        # Calculate final aspect strength considering dignity
        aspect_strength = base_strength * aspecting_weight * aspecting_dignity * orb_factor

        # Apply the aspect influence
        if aspect_type in ["Conjunction", "Trine"]:  # Benefic aspects
            total_modifier += aspect_strength * net_influence
        else:  # Malefic aspects (Opposition, Square)
            total_modifier -= aspect_strength * net_influence

    # Ensure modifier stays within reasonable bounds (-1.0 to 1.0)
    return max(-1.0, min(1.0, total_modifier))

# --- Advanced KP Evaluation Rules ---

# Houses favoring the Ascendant (Team 1)
ASC_PRIMARY_FAVORED = {6, 11}  # Victory, Gains
ASC_SECONDARY_FAVORED = {1, 3, 10}  # Self, Courage, Performance

# Houses favoring the Descendant (Team 2)
DESC_PRIMARY_FAVORED = {8, 12}  # Obstacles (loss of wickets), Loss
DESC_SECONDARY_FAVORED = {4, 5, 7, 9}  # Home Comforts, Loss of victory, Opponent, Luck for opponent

# Houses favoring the Ascendant (Team 1) with cricket-specific weights
ASC_HOUSE_WEIGHTS = {
    6: 4.0,   # Victory over opponents (most important)
    11: 3.5,  # Gains and success
    10: 2.5,  # Performance and status
    2: 2.0,   # Accumulation of runs/wealth
    1: 1.5,   # Team's own strength and effort
    3: 1.0    # Courage and enterprise
}

# Houses favoring the Descendant (Team 2) with cricket-specific weights
DESC_HOUSE_WEIGHTS = {
    8: 4.0,   # Obstacles, accidents, loss of wickets for Ascendant
    12: 3.5,  # House of loss for Ascendant
    5: 2.5,   # Gains for the opponent (11th from the 7th)
    7: 2.0,   # The opponent team itself
    4: 1.5,   # Performance of Opponent (10th from 7th)
    9: 1.0    # Luck for opponent
}

# Cricket-specific planet weights based on their nature and characteristics
PLANET_CRICKET_WEIGHTS = {
    # Aggressive/Attacking Planets
    "Mars": 2.0,    # Aggression, pace, power-hitting
    "Sun": 1.8,     # Leadership, captain, consistent performance
    "Rahu": 1.7,    # Unorthodoxy, sudden game-changing events
    
    # Balanced/Strategic Planets
    "Jupiter": 1.5, # Strategy, wisdom, guidance
    "Mercury": 1.3, # Agility, speed, tactical thinking
    
    # Defensive/Controlling Planets
    "Venus": 1.1,   # Finesse, timing, partnerships
    "Saturn": 1.0,  # Defense, patience, stamina
    "Moon": 0.9,    # Team morale, emotional state
    "Ketu": 0.8     # Sudden collapses, confusion, mystery spin
}

def get_planet_details(planet_name: str, chart: dict, df: pd.DataFrame) -> dict:
    """
    Get all astrological details for a planet including its hierarchical lordships.
    The DataFrame (df) of nakshatra mappings is passed in for efficiency.
    """
    if not planet_name or planet_name not in chart["planets"]:
        return {}
    
    planet_long = chart["planets"][planet_name]["longitude"]
    astro_details = get_astro_details(planet_long, df) # df is passed here
    if not astro_details:
        return {}
    
    star_lord_name = astro_details.get("Star_Lord")
    details = {
        "Sign": astro_details.get("Sign"),
        "Sign_Lord": astro_details.get("Sign_Lord"),
        "Nakshatra": astro_details.get("Nakshatra"),
        "Star_Lord": star_lord_name,
        "Sub_Lord": astro_details.get("Sub_Lord"),
        "House": get_house_for_planet(planet_long, chart["houses"])
    }

    # Now, find the details for the star lord and the star lord's star lord
    if star_lord_name and star_lord_name in chart["planets"]:
        # CORRECTED: Use the star_lord's longitude to find its details
        star_lord_long = chart["planets"][star_lord_name]["longitude"]
        sl_details = get_astro_details(star_lord_long, df)
        if sl_details:
            details["star_lord_of_star_lord"] = sl_details.get("Star_Lord")

    return details

def get_planet_nature(planet: str) -> str:
    """Gets the nature of a planet (benefic, malefic, or neutral)."""
    for nature, planet_list in PLANET_NATURES.items():
        if planet in planet_list:
            return nature
    return "neutral"

def calculate_planet_strength(planet: str, chart: dict, ruling_planets: List[str], nakshatra_df: pd.DataFrame, corrected_scores: dict = None) -> Tuple[float, float, Dict]:
    """
    Calculates the Ascendant and Descendant scores for a single planet.
    
    Args:
        planet: Planet name
        chart: Astrological chart
        ruling_planets: List of ruling planets
        nakshatra_df: Nakshatra data
        corrected_scores: Pre-calculated Level 1 corrected scores (if available)
    """
    if not planet or planet not in chart["planets"]:
        return 0, 0, {}

    # If corrected scores are provided (Level 1 contradictions already applied), use them
    if corrected_scores and planet in corrected_scores:
        corrected_net_score = corrected_scores[planet]
        # Convert net score back to asc/desc format for compatibility
        if corrected_net_score >= 0:
            return corrected_net_score, 0, {"level1_applied": True, "corrected_score": corrected_net_score}
        else:
            return 0, -corrected_net_score, {"level1_applied": True, "corrected_score": corrected_net_score}

    # Otherwise, calculate normally
    details = get_planet_details(planet, chart, nakshatra_df)
    
    # Initialize scores
    asc_score = 0
    desc_score = 0
    
    strength_details = {
        "planet": planet,
        "initial_scores": {},
        "hierarchical_scoring": [],
        "modifiers": [],
        "significations": {
            "planet": {"houses": []},
            "star_lord": {"houses": []},
            "slsl": {"houses": []}
        },
        "weights": HIERARCHICAL_WEIGHTS
    }

    # 1. Score based on house lordship and occupation (hierarchical)
    # Get houses for each level of lordship
    planet_houses = get_significator_houses(planet, chart)
    strength_details["significations"]["planet"]["houses"] = planet_houses

    star_lord = details.get("star_lord")
    sl_houses = get_significator_houses(star_lord, chart) if star_lord else []
    strength_details["significations"]["star_lord"]["houses"] = sl_houses

    slsl = details.get("star_lord_of_star_lord")
    slsl_houses = get_significator_houses(slsl, chart) if slsl else []
    strength_details["significations"]["slsl"]["houses"] = slsl_houses

    # Apply scores with hierarchical weighting
    for house in planet_houses:
        w = HIERARCHICAL_WEIGHTS['planet']
        asc_score += HOUSE_WEIGHTS_ASC.get(house, 0) * w
        desc_score += HOUSE_WEIGHTS_DESC.get(house, 0) * w
    
    for house in sl_houses:
        w = HIERARCHICAL_WEIGHTS['star_lord']
        asc_score += HOUSE_WEIGHTS_ASC.get(house, 0) * w
        desc_score += HOUSE_WEIGHTS_DESC.get(house, 0) * w

    for house in slsl_houses:
        w = HIERARCHICAL_WEIGHTS['slsl']
        asc_score += HOUSE_WEIGHTS_ASC.get(house, 0) * w
        desc_score += HOUSE_WEIGHTS_DESC.get(house, 0) * w

    strength_details["initial_scores"] = {"asc": asc_score, "desc": desc_score}

    # 2. Apply Modifiers
    # Dignity
    dignity_multiplier = get_planet_dignity_strength(planet, chart)
    asc_score *= dignity_multiplier
    desc_score *= dignity_multiplier
    strength_details["modifiers"].append({"type": "Dignity", "multiplier": dignity_multiplier, "asc": asc_score, "desc": desc_score})
    
    # Combust check (NEWLY ADDED)
    if is_combust(planet, chart):
        asc_score *= COMBUST_MODIFIER
        desc_score *= COMBUST_MODIFIER
        strength_details["modifiers"].append({"type": "Combust", "multiplier": COMBUST_MODIFIER, "asc": asc_score, "desc": desc_score})

    # Ruling Planet Bonus
    if planet in ruling_planets:
        ruling_planet_multiplier = 1.2 # 20% bonus
        asc_score *= ruling_planet_multiplier
        desc_score *= ruling_planet_multiplier
        strength_details["modifiers"].append({"type": "Ruling Planet Bonus", "multiplier": ruling_planet_multiplier, "asc": asc_score, "desc": desc_score})

    # Aspects
    aspect_modifier = get_planet_aspect_modifier(planet, chart)
    # We only apply aspect modifier to the favorable score
    if asc_score > desc_score:
        asc_score *= aspect_modifier
    else:
        desc_score *= aspect_modifier
    strength_details["modifiers"].append({"type": "Aspects", "modifier": aspect_modifier, "asc": asc_score, "desc": desc_score})

    # Retrograde
    if chart["planets"][planet].get("retrograde", False):
        retro_multiplier = RETROGRADE_MODIFIERS.get(planet, 1.0)
        asc_score *= retro_multiplier
        desc_score *= retro_multiplier
        strength_details["modifiers"].append({"type": "Retrograde", "multiplier": retro_multiplier, "asc": asc_score, "desc": desc_score})
    
    # High-Risk High-Reward for planets in 8th or 12th house significations
    if any(h in [8, 12] for h in planet_houses) or (star_lord and any(h in [8, 12] for h in get_significator_houses(star_lord, chart))):
         hrhr_multiplier = 0.6
         if asc_score < desc_score: # only apply to negative scores
              desc_score *= hrhr_multiplier
              strength_details["modifiers"].append({"type": "High-Risk High-Reward (Planet/SL)", "multiplier": hrhr_multiplier, "asc": asc_score, "desc": desc_score})

    return asc_score, desc_score, strength_details


def evaluate_favorability(muhurta_chart: dict, current_chart: dict = None, nakshatra_df: pd.DataFrame = None, use_enhanced_rules: bool = True) -> Dict:
    """
    Evaluates the overall favorability using AUTHENTIC KP METHODOLOGY with enhanced rules:
    1. Muhurta chart provides the house system (ascendant, house cusps)
    2. Current planetary positions are analyzed against muhurta chart houses
    3. Enhanced with opposite result corrections and dynamic weighting
    
    Args:
        muhurta_chart: Chart cast for match start time (foundation)
        current_chart: Chart cast for current delivery time (if None, uses muhurta_chart)
        nakshatra_df: Nakshatra subdivision data
        use_enhanced_rules: Whether to apply opposite result corrections and dynamic weighting
    """
    if not muhurta_chart or "error" in muhurta_chart:
        return {"error": "Invalid muhurta chart provided."}
    
    # If no current chart provided, use muhurta chart (backward compatibility)
    if current_chart is None or "error" in current_chart:
        current_chart = muhurta_chart

    # Load nakshatra data once if not passed
    if nakshatra_df is None:
        nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
    
    # AUTHENTIC KP ANALYSIS:
    # 1. Use MUHURTA CHART's house system (ascendant, house cusps) as foundation
    # 2. Use CURRENT CHART's planetary positions for analysis
    # 3. Determine ruling planets from BOTH charts
    
    # Create hybrid chart for analysis
    analysis_chart = {
        "datetime": current_chart["datetime"],
        "ascendant_degree": muhurta_chart["ascendant_degree"],  # Muhurta ascendant
        "houses": muhurta_chart["houses"],  # Muhurta house system
        "planets": current_chart["planets"],  # Current planetary positions
        "moon_longitude": current_chart["moon_longitude"],
        "moon_nakshatra": current_chart["moon_nakshatra"],
        "moon_pada": current_chart["moon_pada"],
        "moon_sub_lord": current_chart["moon_sub_lord"],
        "moon_sub_sub_lord": current_chart["moon_sub_sub_lord"],
        "moon_sign": current_chart["moon_sign"],
        "moon_sign_lord": current_chart["moon_sign_lord"],
        "moon_star_lord": current_chart["moon_star_lord"]
    }
    
    # Determine Ruling Planets from BOTH charts (authentic KP method)
    muhurta_ruling_planets = get_ruling_planets(muhurta_chart, nakshatra_df)
    current_ruling_planets = get_ruling_planets(current_chart, nakshatra_df)
    # Combine and deduplicate ruling planets
    combined_ruling_planets = list(set(muhurta_ruling_planets + current_ruling_planets))

    # Get Moon's hierarchical lords from CURRENT chart
    moon_sl = current_chart.get("moon_star_lord")
    moon_sub = current_chart.get("moon_sub_lord")
    moon_ssl = current_chart.get("moon_sub_sub_lord")

    # Apply Level 1 planet-level contradictions first (if enhanced rules enabled)
    corrected_planet_scores = None
    if use_enhanced_rules:
        corrected_planet_scores = apply_chart_level_contradictions(analysis_chart, nakshatra_df)

    # Calculate individual lord scores
    lord_scores = {}
    lord_details = {}

    # Calculate strength for each lord using HYBRID analysis
    for lord_type, lord_name in [("moon_sl", moon_sl), ("moon_sub", moon_sub), ("moon_ssl", moon_ssl)]:
        if lord_name:
            asc_score, desc_score, details = calculate_planet_strength(
                lord_name, analysis_chart, combined_ruling_planets, nakshatra_df, corrected_planet_scores
            )
            
            # Store individual lord scores for enhanced rules
            net_score = asc_score - desc_score
            lord_scores[lord_type] = net_score
            lord_details[f"{lord_type}_score"] = net_score
            lord_details[f"{lord_type}_details"] = details
        else:
            lord_scores[lord_type] = 0
            lord_details[f"{lord_type}_score"] = 0

    # Apply Enhanced Rules if enabled
    if use_enhanced_rules:
        # 1. Apply opposite result corrections
        corrected_scores = apply_opposite_result_corrections(lord_scores, analysis_chart, nakshatra_df)
        
        # 2. Calculate dynamic weights based on strength and relationships
        dynamic_weights = calculate_dynamic_weights(corrected_scores, analysis_chart)
        
        # 3. Calculate final score with dynamic weighting
        final_score = (
            corrected_scores['moon_sl'] * dynamic_weights['sl'] +
            corrected_scores['moon_sub'] * dynamic_weights['sub'] +
            corrected_scores['moon_ssl'] * dynamic_weights['ssl']
        )
        
        # 4. UPDATE THE INDIVIDUAL SCORES TO SHOW CORRECTED VALUES
        lord_details['moon_sl_score'] = corrected_scores['moon_sl']
        lord_details['moon_sub_score'] = corrected_scores['moon_sub'] 
        lord_details['moon_ssl_score'] = corrected_scores['moon_ssl']
        
        # Store enhanced information
        lord_details['enhanced_corrections'] = corrected_scores
        lord_details['dynamic_weights'] = dynamic_weights
        lord_details['base_weights'] = {'sl': 0.5, 'sub': 0.3, 'ssl': 0.2}
        lord_details['original_scores'] = lord_scores  # Keep original for reference
        
    else:
        # Use original fixed weighting
        total_asc_score = 0
        total_desc_score = 0
        
        for lord_type, lord_name in [("moon_sl", moon_sl), ("moon_sub", moon_sub), ("moon_ssl", moon_ssl)]:
            if lord_name:
                asc_score, desc_score, details = calculate_planet_strength(lord_name, analysis_chart, combined_ruling_planets, nakshatra_df, None)
                
                # Accumulate weighted scores for the final verdict
                weight = LORD_WEIGHTS.get(lord_type.replace("moon_", ""), 0)
                total_asc_score += asc_score * weight
                total_desc_score += desc_score * weight

        # Final score is the difference between the weighted totals
        final_score = total_asc_score - total_desc_score

    summary = {
        "final_score": final_score,
        "ruling_planets": ",".join(combined_ruling_planets) if combined_ruling_planets else "",
        "muhurta_ascendant": muhurta_chart["ascendant_degree"],
        "analysis_method": f"Authentic KP: Muhurta houses + Current planets {'(Enhanced)' if use_enhanced_rules else '(Original)'}"
    }
    summary.update(lord_details)

    return summary


def verdict_label(diff):
    if diff > 20:
        return "Favors Ascendant"
    elif diff > 0:
        return "Favors Ascendant"
    elif diff < -20:
        return "Favors Descendant"
    else:
        return "Neutral"


# Enhanced Rules Functions using Proper Contradiction Hierarchy

# Initialize the contradiction system
_contradiction_system = KPContradictionSystem()

def apply_chart_level_contradictions(chart: dict, nakshatra_df: pd.DataFrame = None) -> dict:
    """
    Apply Level 1 planet-level contradictions to all planets in the chart.
    
    This should be called once per chart to get corrected planetary scores
    that account for house co-placements and aspects.
    
    Returns: dict mapping planet names to corrected base strength scores
    """
    if not chart or "planets" not in chart:
        return {}
    
    # Calculate base scores for all planets in the chart
    ruling_planets = get_ruling_planets(chart, nakshatra_df) if nakshatra_df is not None else []
    planet_base_scores = {}
    
    for planet_name in chart["planets"]:
        if planet_name in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']:
            # Calculate base strength for this planet
            asc_score, desc_score, _ = calculate_planet_strength_base(planet_name, chart, ruling_planets, nakshatra_df)
            net_score = asc_score - desc_score
            planet_base_scores[planet_name] = net_score
    
    # Apply Level 1 contradictions to the base scores
    corrected_scores = _contradiction_system.apply_level1_contradictions(planet_base_scores, chart)
    
    return corrected_scores

def calculate_planet_strength_base(planet: str, chart: dict, ruling_planets: List[str], nakshatra_df: pd.DataFrame) -> Tuple[float, float, Dict]:
    """
    Calculate the base planet strength without Level 1 contradictions.
    This is used internally by apply_chart_level_contradictions.
    """
    if not planet or planet not in chart["planets"]:
        return 0, 0, {}

    # Get planet details
    details = get_planet_details(planet, chart, nakshatra_df)
    
    # Initialize scores
    asc_score = 0
    desc_score = 0
    
    # 1. Score based on house lordship and occupation (hierarchical)
    planet_houses = get_significator_houses(planet, chart)
    star_lord = details.get("star_lord")
    sl_houses = get_significator_houses(star_lord, chart) if star_lord else []
    slsl = details.get("star_lord_of_star_lord")
    slsl_houses = get_significator_houses(slsl, chart) if slsl else []

    # Apply scores with hierarchical weighting
    for house in planet_houses:
        w = HIERARCHICAL_WEIGHTS['planet']
        asc_score += HOUSE_WEIGHTS_ASC.get(house, 0) * w
        desc_score += HOUSE_WEIGHTS_DESC.get(house, 0) * w
    
    for house in sl_houses:
        w = HIERARCHICAL_WEIGHTS['star_lord']
        asc_score += HOUSE_WEIGHTS_ASC.get(house, 0) * w
        desc_score += HOUSE_WEIGHTS_DESC.get(house, 0) * w

    for house in slsl_houses:
        w = HIERARCHICAL_WEIGHTS['slsl']
        asc_score += HOUSE_WEIGHTS_ASC.get(house, 0) * w
        desc_score += HOUSE_WEIGHTS_DESC.get(house, 0) * w

    # 2. Apply Modifiers
    # Dignity
    dignity_multiplier = get_planet_dignity_strength(planet, chart)
    asc_score *= dignity_multiplier
    desc_score *= dignity_multiplier
    
    # Combust check
    if is_combust(planet, chart):
        asc_score *= COMBUST_MODIFIER
        desc_score *= COMBUST_MODIFIER

    # Ruling Planet Bonus
    if planet in ruling_planets:
        ruling_planet_multiplier = 1.2
        asc_score *= ruling_planet_multiplier
        desc_score *= ruling_planet_multiplier

    # Aspects
    aspect_modifier = get_planet_aspect_modifier(planet, chart)
    if asc_score > desc_score:
        asc_score *= aspect_modifier
    else:
        desc_score *= aspect_modifier

    # Retrograde
    if chart["planets"][planet].get("retrograde", False):
        retro_multiplier = RETROGRADE_MODIFIERS.get(planet, 1.0)
        asc_score *= retro_multiplier
        desc_score *= retro_multiplier
    
    # High-Risk High-Reward
    if any(h in [8, 12] for h in planet_houses) or (star_lord and any(h in [8, 12] for h in get_significator_houses(star_lord, chart))):
         hrhr_multiplier = 0.6
         if asc_score < desc_score:
              desc_score *= hrhr_multiplier

    return asc_score, desc_score, {}

def apply_opposite_result_corrections(lord_scores: dict, chart: dict, nakshatra_df: pd.DataFrame) -> dict:
    """
    Apply any remaining corrections to lord scores.
    
    Note: Level 1 planet-level contradictions are now applied at the chart level
    in apply_chart_level_contradictions(), so this function is mainly for
    backward compatibility and any lord-specific corrections not handled elsewhere.
    """
    # Level 1 contradictions are now handled at chart level, so just return the scores
    return lord_scores.copy()


def calculate_dynamic_weights(lord_scores: dict, chart: dict) -> dict:
    """
    Calculate dynamic weights based on planetary strength and relationships.
    
    Note: Level 2 lordship contradictions are now handled in the hybrid model
    for proper separation of concerns. This function focuses on basic strength-based weighting.
    """
    # Get absolute strengths
    sl_strength = abs(lord_scores.get('moon_sl', 0))
    sub_strength = abs(lord_scores.get('moon_sub', 0)) 
    ssl_strength = abs(lord_scores.get('moon_ssl', 0))
    
    total_strength = sl_strength + sub_strength + ssl_strength
    
    # Default to fixed weights if no strength
    if total_strength == 0:
        return {'sl': 0.5, 'sub': 0.3, 'ssl': 0.2}
    
    # 1. Strength-proportional weights (70% based on strength, 30% hierarchy)
    strength_weights = {
        'sl': sl_strength / total_strength,
        'sub': sub_strength / total_strength,
        'ssl': ssl_strength / total_strength
    }
    
    base_weights = {'sl': 0.5, 'sub': 0.3, 'ssl': 0.2}
    
    # Blend strength with hierarchy
    dynamic_weights = {}
    for lord in ['sl', 'sub', 'ssl']:
        dynamic_weights[lord] = 0.7 * strength_weights[lord] + 0.3 * base_weights[lord]
    
    # 2. Check for dominant lord (3x stronger than average)
    max_strength = max(sl_strength, sub_strength, ssl_strength)
    avg_other_strength = (total_strength - max_strength) / 2
    
    if avg_other_strength > 0 and max_strength / avg_other_strength >= 3.0:
        # One lord is dominant
        if max_strength == sl_strength:
            dynamic_weights = {'sl': 0.70, 'sub': 0.20, 'ssl': 0.10}
        elif max_strength == sub_strength:
            dynamic_weights = {'sl': 0.25, 'sub': 0.65, 'ssl': 0.10}
        else:  # SSL is dominant
            dynamic_weights = {'sl': 0.20, 'sub': 0.15, 'ssl': 0.65}
    
    # 3. Check for contradictions (opposite signs)
    sl_score = lord_scores.get('moon_sl', 0)
    sub_score = lord_scores.get('moon_sub', 0)
    ssl_score = lord_scores.get('moon_ssl', 0)
    
    scores = [sl_score, sub_score, ssl_score]
    positive_count = sum(1 for s in scores if s > 0)
    negative_count = sum(1 for s in scores if s < 0)
    
    # If there's contradiction, give more weight to the strongest contradicting opinion
    if min(positive_count, negative_count) > 0:
        strengths = [sl_strength, sub_strength, ssl_strength]
        max_strength_idx = strengths.index(max(strengths))
        
        # Give 20% bonus to the strongest lord in contradiction
        lord_names = ['sl', 'sub', 'ssl']
        strongest_lord = lord_names[max_strength_idx]
        dynamic_weights[strongest_lord] += 0.2
        
        # Normalize weights
        total_weight = sum(dynamic_weights.values())
        dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
    
    # Final normalization to ensure weights sum to 1.0
    total_weight = sum(dynamic_weights.values())
    if total_weight > 0:
        dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
    else:
        dynamic_weights = {'sl': 0.5, 'sub': 0.3, 'ssl': 0.2}
    
    return dynamic_weights
