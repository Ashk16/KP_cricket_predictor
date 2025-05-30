# KP Astrology Favorability Rules Engine
# Determines whether a period favors Ascendant or Descendant based on KP principles

from typing import List, Dict

# Define house groups based on favorability with weights
ASCENDANT_HOUSE_WEIGHTS = {
    1: 10, 3: 8, 6: 10, 10: 12, 11: 10
}
DESCENDANT_HOUSE_WEIGHTS = {
    4: 6, 5: 8, 7: 10, 8: 12, 12: 6
}

# Planet strength weights (arbitrary scale, can be tuned)
PLANET_STRENGTHS = {
    "Jupiter": 10,
    "Venus": 9,
    "Mercury": 8,
    "Moon": 7,
    "Sun": 6,
    "Mars": 7,
    "Saturn": 6,
    "Rahu": 5,
    "Ketu": 5
}

BENEFIC_PLANETS = ["Jupiter", "Venus", "Mercury", "Moon"]
MALEFIC_PLANETS = ["Saturn", "Mars", "Rahu", "Ketu", "Sun"]


def favorability_score(significators: List[int], planet_name: str, is_retrograde=False, is_combust=False, conjunct_with=[], star_lord=None, ruling_planets=[]) -> int:
    score = 0
    planet_strength = PLANET_STRENGTHS.get(planet_name, 5)

    # House contributions
    for house in significators:
        if house in ASCENDANT_HOUSE_WEIGHTS:
            score += ASCENDANT_HOUSE_WEIGHTS[house]
        elif house in DESCENDANT_HOUSE_WEIGHTS:
            score -= DESCENDANT_HOUSE_WEIGHTS[house]

    # Apply strength multiplier
    score = int(score * (planet_strength / 10))

    # Retrograde or combust adjustments
    if is_retrograde or is_combust:
        if planet_name in BENEFIC_PLANETS:
            score -= int(planet_strength * 0.5)
        elif planet_name in MALEFIC_PLANETS:
            score -= int(planet_strength * 0.3)

    # Star lord influence
    if star_lord:
        if star_lord in BENEFIC_PLANETS:
            score += 5
        elif star_lord in MALEFIC_PLANETS:
            score -= 5

    # Conjunction influence
    for planet in conjunct_with:
        if planet in BENEFIC_PLANETS:
            score += 3
        elif planet in MALEFIC_PLANETS:
            score -= 3

    # Ruling planets support
    if planet_name in ruling_planets:
        score += 10

    return score


def evaluate_period(sub_lord_data: Dict, ascendant_sign: str) -> str:
    sub_score = favorability_score(
        sub_lord_data['Sub_Lord_Houses'],
        sub_lord_data['Sub_Lord'],
        sub_lord_data['Sub_Lord_Retrograde'],
        sub_lord_data['Sub_Lord_Combust'],
        conjunct_with=sub_lord_data.get('Sub_Lord_Conjunct', []),
        star_lord=sub_lord_data.get('Sub_Lord_Star_Lord'),
        ruling_planets=sub_lord_data.get('Ruling_Planets', [])
    )

    sub_sub_score = favorability_score(
        sub_lord_data['Sub_Sub_Lord_Houses'],
        sub_lord_data['Sub_Sub_Lord'],
        sub_lord_data['Sub_Sub_Lord_Retrograde'],
        sub_lord_data['Sub_Sub_Lord_Combust'],
        conjunct_with=sub_lord_data.get('Sub_Sub_Lord_Conjunct', []),
        star_lord=sub_lord_data.get('Sub_Sub_Lord_Star_Lord'),
        ruling_planets=sub_lord_data.get('Ruling_Planets', [])
    )

    total_score = sub_score + sub_sub_score

    if total_score >= 50:
        return f"Strongly Ascendant (+{total_score})"
    elif total_score >= 20:
        return f"Ascendant (+{total_score})"
    elif total_score <= -50:
        return f"Strongly Descendant ({total_score})"
    elif total_score <= -20:
        return f"Descendant ({total_score})"
    else:
        return f"Neutral ({total_score})"


# Example usage
if __name__ == "__main__":
    period_data = {
        'Sub_Lord': 'Saturn',
        'Sub_Sub_Lord': 'Mercury',
        'Sub_Lord_Houses': [1, 6, 11],
        'Sub_Sub_Lord_Houses': [4, 8],
        'Sub_Lord_Retrograde': False,
        'Sub_Sub_Lord_Retrograde': True,
        'Sub_Lord_Combust': False,
        'Sub_Sub_Lord_Combust': False,
        'Sub_Lord_Conjunct': ['Moon'],
        'Sub_Sub_Lord_Conjunct': ['Ketu'],
        'Sub_Lord_Star_Lord': 'Venus',
        'Sub_Sub_Lord_Star_Lord': 'Saturn',
        'Ruling_Planets': ['Mercury', 'Saturn']
    }

    result = evaluate_period(period_data, ascendant_sign="Libra")
    print(f"This period favors: {result}")
