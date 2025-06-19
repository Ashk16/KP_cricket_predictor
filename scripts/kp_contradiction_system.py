"""
KP Contradiction System - Proper Hierarchy Implementation

This module implements the correct contradiction hierarchy as per authentic KP principles:

Level 1: Planet-Level Contradictions (applied to individual planetary scores)
- House co-placement contradictions (planets in same house)
- Aspect-based contradictions (planets aspecting each other)
- Applied after house signification calculation but before combining lords

Level 2: Lordship Hierarchy Contradictions (applied to final combined score)
- Star Lord + Sub Lord contradictions
- Star Lord + Sub Sub Lord contradictions
- Sub Lord + Sub Sub Lord contradictions
- Applied after weighted combination of all lord scores

This ensures planetary score consistency while properly separating chart-based 
influences from decision-making hierarchy influences.
"""

from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class KPContradictionSystem:
    """
    Comprehensive KP Contradiction System implementing proper hierarchy
    """
    
    def __init__(self):
        self.setup_contradiction_rules()
    
    def setup_contradiction_rules(self):
        """Setup all contradiction rules grouped by astrological basis"""
        
        # Level 1: Planet-Level Contradictions
        self.planet_contradictions = {
            # Natural Enemy Contradictions
            'mars_rahu': {
                'planets': ['Mars', 'Rahu'],
                'effect': 'identical_negative',  # Both get identical negative score
                'multiplier': -1.2,
                'reason': 'Mars (energy) conflicts with Rahu (illusion/materialism)'
            },
            'sun_saturn': {
                'planets': ['Sun', 'Saturn'],
                'effect': 'mutual_reduction', 
                'multiplier': 0.6,
                'reason': 'Sun (authority) vs Saturn (restriction) - mutual weakening'
            },
            'mars_saturn': {
                'planets': ['Mars', 'Saturn'],
                'effect': 'mutual_reduction',
                'multiplier': 0.7,
                'reason': 'Mars (action) blocked by Saturn (delay/restriction)'
            },
            'mercury_moon': {
                'planets': ['Mercury', 'Moon'],
                'effect': 'mutual_reduction',
                'multiplier': 0.75,
                'reason': 'Mercury (rational mind) conflicts with Moon (emotional intuition)'
            },
            'mercury_ketu': {
                'planets': ['Mercury', 'Ketu'],
                'effect': 'mutual_reduction',
                'multiplier': 0.8,
                'reason': 'Mercury (detailed analysis) vs Ketu (detached spirituality)'
            },
            'jupiter_mercury': {
                'planets': ['Jupiter', 'Mercury'],
                'effect': 'mutual_reduction',
                'multiplier': 0.75,
                'reason': 'Jupiter (expansive wisdom) vs Mercury (discriminating analysis)'
            },
            'jupiter_venus': {
                'planets': ['Jupiter', 'Venus'],
                'effect': 'mutual_reduction',
                'multiplier': 0.8,
                'reason': 'Jupiter (spiritual wisdom) vs Venus (material pleasures)'
            },
            'venus_sun': {
                'planets': ['Venus', 'Sun'],
                'effect': 'mutual_reduction',
                'multiplier': 0.75,
                'reason': 'Venus (partnership harmony) vs Sun (individual authority)'
            },
            'venus_moon': {
                'planets': ['Venus', 'Moon'],
                'effect': 'mutual_reduction',
                'multiplier': 0.8,
                'reason': 'Venus (romantic love) vs Moon (maternal care)'
            },
            'saturn_rahu': {
                'planets': ['Saturn', 'Rahu'],
                'effect': 'mutual_reduction',
                'multiplier': 0.85,
                'reason': 'Saturn (discipline/structure) vs Rahu (shortcuts/chaos)'
            },
            
            # Corruption/Influence
            'jupiter_rahu': {
                'planets': ['Jupiter', 'Rahu'],
                'effect': 'jupiter_corruption',
                'multiplier': -0.8,
                'reason': 'Jupiter (wisdom) corrupted by Rahu (materialism)'
            },
            
            # Temperamental Conflicts
            'venus_mars': {
                'planets': ['Venus', 'Mars'],
                'effect': 'mutual_reduction',
                'multiplier': 0.8,
                'reason': 'Venus (harmony) vs Mars (aggression) - disharmony'
            },
            'moon_mars': {
                'planets': ['Moon', 'Mars'],
                'effect': 'mutual_reduction',
                'multiplier': 0.75,
                'reason': 'Moon (emotions) + Mars (aggression) - emotional volatility'
            }
        }
        
        # Level 2: Lordship Hierarchy Contradictions
        # Note: Mercury-Jupiter also exists as Level 1, but Level 2 has precedence rules
        self.hierarchy_contradictions = {
            # 1. Mercury-Jupiter: Quick vs Deep thinking (Weight reduction)
            'mercury_jupiter': {
                'planets': ['Mercury', 'Jupiter'],
                'effect': 'calculation_instability',
                'multiplier': 0.8,
                'reason': 'Mercury (quick thinking) vs Jupiter (deep wisdom) - affects calculation flow'
            },
            
            # 2. Mars-Rahu: Direct action vs Scheming (Score corruption)
            'mars_rahu': {
                'planets': ['Mars', 'Rahu'],
                'effect': 'score_corruption',
                'multiplier': -0.9,
                'reason': 'Mars (immediate action) vs Rahu (long-term scheming) - corrupts protective energy into destruction'
            },
            
            # 3. Sun-Saturn: Authority vs Restriction (Authority undermining)
            'sun_saturn': {
                'planets': ['Sun', 'Saturn'],
                'effect': 'authority_undermining',
                'multiplier': -0.8,
                'reason': 'Sun (divine authority) vs Saturn (worldly restriction) - authority becomes oppression'
            },
            
            # 4. Jupiter-Venus: Spiritual vs Material (Weight reduction)
            'jupiter_venus': {
                'planets': ['Jupiter', 'Venus'],
                'effect': 'calculation_instability',
                'multiplier': 0.75,
                'reason': 'Jupiter (spiritual wisdom) vs Venus (material pleasure) - conflicting value systems'
            },
            
            # 5. Sun-Rahu: Divine vs Deceptive (Complete reversal - Eclipse effect)
            'sun_rahu': {
                'planets': ['Sun', 'Rahu'],
                'effect': 'complete_reversal',
                'multiplier': -1.2,
                'reason': 'Sun (divine truth) vs Rahu (illusion/deception) - eclipse effect reverses divine nature'
            },
            
            # 6. Moon-Saturn: Emotion vs Logic (Weight reduction)
            'moon_saturn': {
                'planets': ['Moon', 'Saturn'],
                'effect': 'calculation_instability',
                'multiplier': 0.7,
                'reason': 'Moon (emotional intuition) vs Saturn (cold logic) - creates internal conflict'
            },
            
            # 7. Mars-Saturn: Action vs Delay (Weight reduction)
            'mars_saturn': {
                'planets': ['Mars', 'Saturn'],
                'effect': 'calculation_instability',
                'multiplier': 0.65,
                'reason': 'Mars (immediate action) vs Saturn (delays/obstacles) - timing becomes erratic'
            },
            
            # 8. Jupiter-Rahu: Wisdom vs Materialism (Guru Chandal - Wisdom corruption)
            'jupiter_rahu': {
                'planets': ['Jupiter', 'Rahu'],
                'effect': 'wisdom_corruption',
                'multiplier': -1.1,
                'reason': 'Jupiter (divine wisdom) vs Rahu (material obsession) - classic Guru Chandal corruption'
            }
        }
    
    def detect_house_coplacements(self, chart: dict) -> List[Dict]:
        """
        Detect planets in the same house for Level 1 contradictions
        """
        house_planets = {}
        contradictions = []
        
        # Handle both nested chart structure and flat structure
        planets_data = chart.get('planets', chart)  # Try nested first, fallback to flat
        houses = chart.get('houses', [])
        
        # Group planets by house
        for planet in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']:
            planet_longitude = None
            
            # Handle nested structure: chart['planets']['Mars']['longitude']
            if isinstance(planets_data, dict) and planet in planets_data:
                if isinstance(planets_data[planet], dict):
                    planet_longitude = planets_data[planet].get('longitude')
                else:
                    planet_longitude = planets_data[planet]  # Flat structure: chart['Mars']
            
            if planet_longitude is not None:
                planet_house = self.get_house_for_planet(planet_longitude, houses)
                if planet_house not in house_planets:
                    house_planets[planet_house] = []
                house_planets[planet_house].append(planet)
        
        # Check for contradictory co-placements
        for house, planets in house_planets.items():
            if len(planets) > 1:
                for rule_name, rule_data in self.planet_contradictions.items():
                    rule_planets = rule_data['planets']
                    if all(planet in planets for planet in rule_planets):
                        contradictions.append({
                            'type': 'house_coplacement',
                            'rule': rule_name,
                            'planets': rule_planets,
                            'house': house,
                            'effect': rule_data['effect'],
                            'multiplier': rule_data['multiplier'],
                            'reason': rule_data['reason']
                        })
        
        return contradictions
    
    def detect_planetary_aspects(self, chart: dict) -> List[Dict]:
        """
        Detect contradictory planetary aspects for Level 1 contradictions
        """
        contradictions = []
        
        # Handle both nested chart structure and flat structure
        planets_data = chart.get('planets', chart)  # Try nested first, fallback to flat
        
        # Check aspects between contradictory planets
        for rule_name, rule_data in self.planet_contradictions.items():
            planets = rule_data['planets']
            if len(planets) == 2:
                # Get longitudes for both planets
                planet1_long = None
                planet2_long = None
                
                for planet in planets:
                    if isinstance(planets_data, dict) and planet in planets_data:
                        if isinstance(planets_data[planet], dict):
                            longitude = planets_data[planet].get('longitude')
                        else:
                            longitude = planets_data[planet]  # Flat structure
                        
                        if planet == planets[0]:
                            planet1_long = longitude
                        else:
                            planet2_long = longitude
                
                if planet1_long is not None and planet2_long is not None:
                    # Check if planets are aspecting each other (within 5 degrees of exact aspects)
                    angular_distance = abs(planet1_long - planet2_long) % 360
                    if angular_distance > 180:
                        angular_distance = 360 - angular_distance
                    
                    # Check for major aspects: Traditional Western + KP Special Aspects
                    # Traditional: 0° (conjunction), 60° (sextile), 90° (square), 120° (trine), 180° (opposition)
                    # KP Special: 150° (5th house aspect), 240° (9th house aspect)
                    major_aspects = [0, 60, 90, 120, 150, 180, 240]
                    orb = 5.0  # 5-degree orb
                    
                    for aspect_angle in major_aspects:
                        if abs(angular_distance - aspect_angle) <= orb:
                            contradictions.append({
                                'type': 'planetary_aspect',
                                'rule': rule_name,
                                'planets': planets,
                                'aspect_angle': aspect_angle,
                                'angular_distance': angular_distance,
                                'effect': rule_data['effect'],
                                'multiplier': rule_data['multiplier'],
                                'reason': rule_data['reason']
                            })
                            break
        
        return contradictions
    
    def detect_lordship_contradictions(self, star_lord: str, sub_lord: str, sub_sub_lord: str) -> List[Dict]:
        """
        Detect Level 2 lordship hierarchy contradictions
        """
        contradictions = []
        lords = [star_lord, sub_lord, sub_sub_lord]
        
        for rule_name, rule_data in self.hierarchy_contradictions.items():
            rule_planets = rule_data['planets']
            if all(planet in lords for planet in rule_planets):
                contradictions.append({
                    'type': 'lordship_hierarchy',
                    'rule': rule_name,
                    'planets': rule_planets,
                    'lords': {'star_lord': star_lord, 'sub_lord': sub_lord, 'sub_sub_lord': sub_sub_lord},
                    'effect': rule_data['effect'],
                    'multiplier': rule_data['multiplier'],
                    'reason': rule_data['reason']
                })
        
        return contradictions
    
    def apply_level1_contradictions(self, planet_scores: Dict[str, float], chart: dict) -> Dict[str, float]:
        """
        Apply Level 1 contradictions to individual planetary scores
        """
        corrected_scores = planet_scores.copy()
        
        # Detect house co-placements and aspects
        house_contradictions = self.detect_house_coplacements(chart)
        aspect_contradictions = self.detect_planetary_aspects(chart)
        
        all_contradictions = house_contradictions + aspect_contradictions
        
        # Apply contradictions to planet scores
        for contradiction in all_contradictions:
            effect = contradiction['effect']
            planets = contradiction['planets']
            multiplier = contradiction['multiplier']
            
            if effect == 'identical_negative':
                # Mars-Rahu: Both get identical negative score
                planet_base_scores = []
                for planet in planets:
                    if planet in corrected_scores:
                        planet_base_scores.append(abs(corrected_scores[planet]))
                
                if planet_base_scores:
                    max_base_score = max(planet_base_scores)
                    standardized_score = max_base_score * multiplier  # multiplier is negative
                    
                    for planet in planets:
                        if planet in corrected_scores:
                            corrected_scores[planet] = standardized_score
                            logger.info(f"Applied {contradiction['rule']} contradiction: {planet} = {standardized_score}")
            
            elif effect == 'jupiter_corruption':
                # Jupiter-Rahu: Only Jupiter gets affected
                if 'Jupiter' in corrected_scores:
                    original_score = corrected_scores['Jupiter']
                    corrected_scores['Jupiter'] = original_score * multiplier
                    logger.info(f"Applied {contradiction['rule']} contradiction: Jupiter {original_score} → {corrected_scores['Jupiter']}")
            
            elif effect == 'mutual_reduction':
                # Both planets get reduced
                for planet in planets:
                    if planet in corrected_scores:
                        original_score = corrected_scores[planet]
                        corrected_scores[planet] = original_score * multiplier
                        logger.info(f"Applied {contradiction['rule']} contradiction: {planet} {original_score} → {corrected_scores[planet]}")
        
        return corrected_scores
    
    def has_level1_contradictions(self, chart: dict, star_lord: str, sub_lord: str, sub_sub_lord: str) -> List[str]:
        """
        Check if any Level 1 contradictions exist between the lords.
        Returns list of planetary pairs that have Level 1 contradictions.
        """
        lords = [star_lord, sub_lord, sub_sub_lord]
        level1_pairs = []
        
        # Check house co-placements
        house_contradictions = self.detect_house_coplacements(chart)
        for contradiction in house_contradictions:
            planets = contradiction['planets']
            if all(planet in lords for planet in planets):
                level1_pairs.append(f"{planets[0]}-{planets[1]}")
        
        # Check aspects
        aspect_contradictions = self.detect_planetary_aspects(chart)
        for contradiction in aspect_contradictions:
            planets = contradiction['planets']
            if all(planet in lords for planet in planets):
                level1_pairs.append(f"{planets[0]}-{planets[1]}")
        
        return level1_pairs

    def apply_level2_contradictions(self, final_score: float, weights: Dict[str, float], 
                                  star_lord: str, sub_lord: str, sub_sub_lord: str, chart: dict = None) -> Tuple[float, Dict[str, float]]:
        """
        Apply Level 2 contradictions to final combined score and weights.
        
        IMPORTANT: Only applies Level 2 if NO Level 1 contradictions exist for the same planetary pairs.
        This prevents double contradiction penalties.
        """
        corrected_score = final_score
        corrected_weights = weights.copy()
        
        # Check for Level 1 contradiction precedence
        level1_pairs = []
        if chart:
            level1_pairs = self.has_level1_contradictions(chart, star_lord, sub_lord, sub_sub_lord)
        
        # Detect lordship contradictions
        lordship_contradictions = self.detect_lordship_contradictions(star_lord, sub_lord, sub_sub_lord)
        
        # Apply lordship contradictions ONLY if no Level 1 contradictions exist for the same pairs
        for contradiction in lordship_contradictions:
            planets = contradiction['planets']
            planetary_pair = f"{planets[0]}-{planets[1]}"
            reverse_pair = f"{planets[1]}-{planets[0]}"
            
            # Check if this planetary pair already has Level 1 contradiction
            if planetary_pair in level1_pairs or reverse_pair in level1_pairs:
                logger.info(f"Skipping Level 2 {contradiction['rule']} - Level 1 contradiction already applied for {planetary_pair}")
                continue
            
            effect = contradiction['effect']
            multiplier = contradiction['multiplier']
            
            if effect == 'calculation_instability':
                # Weight reduction: Mercury-Jupiter, Jupiter-Venus, Moon-Saturn, Mars-Saturn
                lords = [star_lord, sub_lord, sub_sub_lord]
                weight_keys = ['star_lord', 'sub_lord', 'sub_sub_lord']
                
                for planet in planets:
                    for i, lord in enumerate(lords):
                        if lord == planet:
                            weight_key = weight_keys[i]
                            original_weight = corrected_weights[weight_key]
                            corrected_weights[weight_key] = original_weight * multiplier
                            logger.info(f"Applied Level 2 {contradiction['rule']} weight adjustment: {weight_key} {original_weight} → {corrected_weights[weight_key]}")
                
                # Normalize weights
                weight_sum = sum(corrected_weights.values())
                if weight_sum > 0:
                    corrected_weights = {k: v/weight_sum for k, v in corrected_weights.items()}
            
            elif effect == 'complete_reversal':
                # Complete score reversal: Sun-Rahu (Eclipse effect)
                corrected_score = corrected_score * multiplier  # Negative multiplier flips the sign
                logger.info(f"Applied Level 2 {contradiction['rule']} complete reversal: final score {final_score} → {corrected_score}")
            
            elif effect == 'score_corruption':
                # Score corruption: Mars-Rahu 
                if corrected_score > 0:  # Only corrupt positive scores
                    corrected_score = corrected_score * multiplier  # Negative multiplier
                    logger.info(f"Applied Level 2 {contradiction['rule']} score corruption: positive score → {corrected_score}")
            
            elif effect == 'wisdom_corruption':
                # Wisdom corruption: Jupiter-Rahu (Guru Chandal)
                if corrected_score > 0:  # Only corrupt positive wisdom
                    corrected_score = corrected_score * multiplier  # Negative multiplier
                    logger.info(f"Applied Level 2 {contradiction['rule']} wisdom corruption: positive wisdom → {corrected_score}")
            
            elif effect == 'authority_undermining':
                # Authority undermining: Sun-Saturn
                if corrected_score > 0:  # Only undermine positive authority
                    corrected_score = corrected_score * multiplier  # Negative multiplier
                    logger.info(f"Applied Level 2 {contradiction['rule']} authority undermining: positive authority → {corrected_score}")
        
        return corrected_score, corrected_weights
    
    def get_house_for_planet(self, planet_longitude: float, houses: List[float]) -> int:
        """Get house number for a planet given its longitude and house cusps"""
        if not houses or len(houses) < 12:
            return 1  # Default to 1st house if houses not available
        
        # Houses are 30-degree segments starting from ascendant
        for i in range(12):
            house_start = houses[i] % 360
            house_end = houses[(i + 1) % 12] % 360
            
            planet_long = planet_longitude % 360
            
            # Handle cases where house crosses 0 degrees
            if house_start > house_end:
                if planet_long >= house_start or planet_long < house_end:
                    return i + 1
            else:
                if house_start <= planet_long < house_end:
                    return i + 1
        
        return 1  # Default to 1st house


def test_contradiction_system():
    """Test the contradiction system with sample data"""
    system = KPContradictionSystem()
    
    # Test Level 1: House co-placement
    test_chart = {
        'Mars': 45.0,      # House 2
        'Rahu': 50.0,      # House 2 (co-placed with Mars)
        'Jupiter': 120.0,  # House 5
        'houses': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    }
    
    test_scores = {'Mars': 8.5, 'Rahu': 6.2, 'Jupiter': 9.1}
    
    print("=== Testing Level 1 Contradictions ===")
    corrected_scores = system.apply_level1_contradictions(test_scores, test_chart)
    print(f"Original scores: {test_scores}")
    print(f"Corrected scores: {corrected_scores}")
    
    # Test Level 2: Lordship contradictions  
    print("\n=== Testing Level 2 Contradictions ===")
    final_score = 5.5
    weights = {'star_lord': 0.5, 'sub_lord': 0.3, 'sub_sub_lord': 0.2}
    
    corrected_final, corrected_weights = system.apply_level2_contradictions(
        final_score, weights, 'Mercury', 'Jupiter', 'Venus'
    )
    
    print(f"Original final score: {final_score}")
    print(f"Original weights: {weights}")
    print(f"Corrected weights: {corrected_weights}")

if __name__ == "__main__":
    test_contradiction_system() 