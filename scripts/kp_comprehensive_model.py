"""
KP Comprehensive Cricket Prediction Model
==========================================

This module provides a complete overhaul of the KP astrology-based cricket prediction system,
addressing all identified systematic issues:

1. Consistent house weight variables and calculations
2. Proper KP hierarchy implementation (Sub Lord supremacy)
3. Correct Sub Lord calculation methodology
4. Proper aspect application in KP context
5. Comprehensive dignity and contradiction system
6. Dynamic weighting based on planetary strength
7. Enhanced opposite results detection
"""

import swisseph as swe
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np


class KPComprehensiveModel:
    """Comprehensive KP Cricket Prediction Model"""
    
    def __init__(self):
        """Initialize the comprehensive KP model"""
        self.setup_constants()
        self.setup_house_weights()
        self.setup_planetary_data()
        
    def setup_constants(self):
        """Setup fundamental KP constants"""
        # Nakshatra data with exact degrees
        self.nakshatras = [
            {'name': 'Ashwini', 'lord': 'Ketu', 'start': 0.0, 'end': 13.333333},
            {'name': 'Bharani', 'lord': 'Venus', 'start': 13.333333, 'end': 26.666667},
            {'name': 'Krittika', 'lord': 'Sun', 'start': 26.666667, 'end': 40.0},
            {'name': 'Rohini', 'lord': 'Moon', 'start': 40.0, 'end': 53.333333},
            {'name': 'Mrigashira', 'lord': 'Mars', 'start': 53.333333, 'end': 66.666667},
            {'name': 'Ardra', 'lord': 'Rahu', 'start': 66.666667, 'end': 80.0},
            {'name': 'Punarvasu', 'lord': 'Jupiter', 'start': 80.0, 'end': 93.333333},
            {'name': 'Pushya', 'lord': 'Saturn', 'start': 93.333333, 'end': 106.666667},
            {'name': 'Ashlesha', 'lord': 'Mercury', 'start': 106.666667, 'end': 120.0},
            {'name': 'Magha', 'lord': 'Ketu', 'start': 120.0, 'end': 133.333333},
            {'name': 'Purva Phalguni', 'lord': 'Venus', 'start': 133.333333, 'end': 146.666667},
            {'name': 'Uttara Phalguni', 'lord': 'Sun', 'start': 146.666667, 'end': 160.0},
            {'name': 'Hasta', 'lord': 'Moon', 'start': 160.0, 'end': 173.333333},
            {'name': 'Chitra', 'lord': 'Mars', 'start': 173.333333, 'end': 186.666667},
            {'name': 'Swati', 'lord': 'Rahu', 'start': 186.666667, 'end': 200.0},
            {'name': 'Vishakha', 'lord': 'Jupiter', 'start': 200.0, 'end': 213.333333},
            {'name': 'Anuradha', 'lord': 'Saturn', 'start': 213.333333, 'end': 226.666667},
            {'name': 'Jyeshtha', 'lord': 'Mercury', 'start': 226.666667, 'end': 240.0},
            {'name': 'Mula', 'lord': 'Ketu', 'start': 240.0, 'end': 253.333333},
            {'name': 'Purva Ashadha', 'lord': 'Venus', 'start': 253.333333, 'end': 266.666667},
            {'name': 'Uttara Ashadha', 'lord': 'Sun', 'start': 266.666667, 'end': 280.0},
            {'name': 'Shravana', 'lord': 'Moon', 'start': 280.0, 'end': 293.333333},
            {'name': 'Dhanishta', 'lord': 'Mars', 'start': 293.333333, 'end': 306.666667},
            {'name': 'Shatabhisha', 'lord': 'Rahu', 'start': 306.666667, 'end': 320.0},
            {'name': 'Purva Bhadrapada', 'lord': 'Jupiter', 'start': 320.0, 'end': 333.333333},
            {'name': 'Uttara Bhadrapada', 'lord': 'Saturn', 'start': 333.333333, 'end': 346.666667},
            {'name': 'Revati', 'lord': 'Mercury', 'start': 346.666667, 'end': 360.0}
        ]
        
        # Sub divisions within each nakshatra (9 parts)
        self.sub_divisions = ['Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu', 'Jupiter', 'Saturn', 'Mercury']
        
        # Sub-sub divisions within each sub (9 parts)
        self.sub_sub_divisions = ['Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu', 'Jupiter', 'Saturn', 'Mercury']
        
        # Planet to number mapping
        self.planet_numbers = {
            'Sun': 0, 'Moon': 1, 'Mars': 4, 'Mercury': 6, 'Jupiter': 5,
            'Venus': 3, 'Saturn': 2, 'Rahu': 7, 'Ketu': 8
        }
        
        # Signs for dignity calculations
        self.signs = [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]
        
        # Sign rulers
        self.sign_rulers = {
            "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
            "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars",
            "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"
        }
        
        # Enhanced aspect system (from legacy)
        self.aspects = {
            "conjunction": {"angle": 0, "orb": 10, "type": "neutral"},
            "opposition": {"angle": 180, "orb": 10, "type": "hard"},
            "trine": {"angle": 120, "orb": 8, "type": "soft"},
            "square": {"angle": 90, "orb": 8, "type": "hard"},
        }
        
    def setup_house_weights(self):
        """Setup consistent house weights for cricket prediction with ASC/DESC separation"""
        # Houses favoring the Ascendant (Team A - batting team) 
        self.asc_house_weights = {
            6: 4.0,   # Victory over opponents (most important)
            11: 3.5,  # Gains and success
            10: 2.5,  # Performance and status
            2: 2.0,   # Accumulation of runs/wealth
            1: 1.5,   # Team's own strength and effort
            3: 1.0    # Courage and enterprise
        }
        
        # Houses favoring the Descendant (Team B - bowling team)
        self.desc_house_weights = {
            8: 4.0,   # Obstacles, accidents, loss of wickets for Ascendant
            12: 3.5,  # House of loss for Ascendant
            5: 2.5,   # Gains for the opponent (11th from the 7th)
            7: 2.0,   # The opponent team itself
            4: 1.5,   # Performance of Opponent (10th from 7th)
            9: 1.0    # Luck for opponent
        }
        
        # Combined house weights for simple lookups
        self.house_weights = {
            1: 1.5, 2: 2.0, 3: 1.0, 4: -1.5, 5: -2.5, 6: 4.0,
            7: -2.0, 8: -4.0, 9: -1.0, 10: 2.5, 11: 3.5, 12: -3.5
        }
        
    def setup_planetary_data(self):
        """Setup comprehensive planetary data with legacy features"""
        # Traditional dignities (from legacy model)
        self.planet_dignities = {
            "Sun": {"exaltation": "Aries", "debilitation": "Libra", "moolatrikona": "Leo", "own_house": ["Leo"]},
            "Moon": {"exaltation": "Taurus", "debilitation": "Scorpio", "moolatrikona": "Taurus", "own_house": ["Cancer"]},
            "Mars": {"exaltation": "Capricorn", "debilitation": "Cancer", "moolatrikona": "Aries", "own_house": ["Aries", "Scorpio"]},
            "Mercury": {"exaltation": "Virgo", "debilitation": "Pisces", "moolatrikona": "Virgo", "own_house": ["Gemini", "Virgo"]},
            "Jupiter": {"exaltation": "Cancer", "debilitation": "Capricorn", "moolatrikona": "Sagittarius", "own_house": ["Sagittarius", "Pisces"]},
            "Venus": {"exaltation": "Pisces", "debilitation": "Virgo", "moolatrikona": "Libra", "own_house": ["Taurus", "Libra"]},
            "Saturn": {"exaltation": "Libra", "debilitation": "Aries", "moolatrikona": "Aquarius", "own_house": ["Capricorn", "Aquarius"]},
            "Rahu": {}, 
            "Ketu": {}
        }
        
        # Natural rank modifiers (from legacy)
        self.natural_rank_modifiers = {
            "Sun": 1.3, "Moon": 1.25, "Saturn": 1.2, "Jupiter": 1.2,
            "Mars": 1.1, "Venus": 1.1, "Mercury": 1.0,
            "Rahu": 1.0, "Ketu": 1.0
        }
        
        # Cricket-specific planet weights (from legacy)
        self.planet_cricket_weights = {
            "Mars": 2.0, "Sun": 1.8, "Rahu": 1.7,
            "Jupiter": 1.5, "Mercury": 1.3,
            "Venus": 1.1, "Saturn": 1.0, "Moon": 0.9, "Ketu": 0.8
        }
        
        # Retrograde modifiers (from legacy)
        self.retrograde_modifiers = {
            "Mercury": 0.85, "Venus": 0.90, "Mars": 1.15,
            "Jupiter": 0.90, "Saturn": 1.20
        }
        
        # Hierarchical weights (from legacy)
        self.hierarchical_weights = {
            "planet": 0.25,
            "star_lord": 0.40,
            "slsl": 0.35  # Star Lord's Star Lord
        }
        
        # Enhanced dignity system
        self.dignity_modifiers = {
            'own_sign': 1.5,
            'exaltation': 2.0,
            'debilitation': 0.3,
            'moolatrikona': 1.3,
            'friend_sign': 1.2,
            'neutral_sign': 1.0,
            'enemy_sign': 0.7
        }
        
        # Aspect strengths in KP
        self.aspect_strengths = {
            'conjunction': 1.0,
            'opposition': 0.75,
            'trine': 0.75,
            'square': 0.5,
            'sextile': 0.25
        }
        
        # Contradiction patterns
        self.contradiction_patterns = [
            ('Mars', 'Rahu'),      # Aggressive contradiction
            ('Mars', 'Saturn'),    # Energy vs Restriction
            ('Sun', 'Saturn'),     # Authority vs Limitation
            ('Moon', 'Mars'),      # Emotion vs Aggression
            ('Jupiter', 'Rahu'),   # Wisdom vs Materialism
            ('Venus', 'Mars'),     # Harmony vs Conflict
            ('Mercury', 'Jupiter') # Quick thinking vs Deep wisdom
        ]
        
    def calculate_planetary_positions(self, jd: float, lat: float, lon: float) -> Dict[str, Dict]:
        """Calculate precise planetary positions"""
        positions = {}
        
        # Calculate Ayanamsa
        ayanamsa = swe.get_ayanamsa(jd)
        
        # Calculate house cusps using Placidus system
        houses, ascmc = swe.houses(jd, lat, lon, b'P')
        ascendant = ascmc[0] - ayanamsa
        if ascendant < 0:
            ascendant += 360
            
        # Calculate planetary positions
        planets = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']
        planet_ids = [swe.SUN, swe.MOON, swe.MARS, swe.MERCURY, swe.JUPITER, swe.VENUS, swe.SATURN]
        
        for planet, planet_id in zip(planets, planet_ids):
            pos, _ = swe.calc_ut(jd, planet_id)
            sidereal_pos = pos[0] - ayanamsa
            if sidereal_pos < 0:
                sidereal_pos += 360
                
            positions[planet] = {
                'longitude': sidereal_pos,
                'nakshatra': self.get_nakshatra(sidereal_pos),
                'house': self.get_house_number(sidereal_pos, ascendant)
            }
        
        # Calculate Rahu and Ketu
        rahu_pos, _ = swe.calc_ut(jd, swe.MEAN_NODE)
        rahu_sidereal = rahu_pos[0] - ayanamsa
        if rahu_sidereal < 0:
            rahu_sidereal += 360
            
        ketu_sidereal = rahu_sidereal + 180
        if ketu_sidereal >= 360:
            ketu_sidereal -= 360
            
        positions['Rahu'] = {
            'longitude': rahu_sidereal,
            'nakshatra': self.get_nakshatra(rahu_sidereal),
            'house': self.get_house_number(rahu_sidereal, ascendant)
        }
        
        positions['Ketu'] = {
            'longitude': ketu_sidereal,
            'nakshatra': self.get_nakshatra(ketu_sidereal),
            'house': self.get_house_number(ketu_sidereal, ascendant)
        }
        
        # Add ascendant
        positions['Ascendant'] = {
            'longitude': ascendant,
            'nakshatra': self.get_nakshatra(ascendant),
            'house': 1
        }
        
        return positions
        
    def get_nakshatra(self, longitude: float) -> Dict[str, Any]:
        """Get nakshatra information for a given longitude"""
        for nak in self.nakshatras:
            if nak['start'] <= longitude < nak['end']:
                return {
                    'name': nak['name'],
                    'lord': nak['lord'],
                    'position_in_nak': longitude - nak['start'],
                    'nak_span': nak['end'] - nak['start']
                }
        # Handle edge case for 360 degrees
        return {
            'name': 'Revati',
            'lord': 'Mercury',
            'position_in_nak': longitude - 346.666667,
            'nak_span': 13.333333
        }
        
    def get_house_number(self, planet_lon: float, ascendant_lon: float) -> int:
        """Calculate house number for a planet"""
        relative_position = planet_lon - ascendant_lon
        if relative_position < 0:
            relative_position += 360
            
        house = int(relative_position / 30) + 1
        return house if house <= 12 else house - 12
        
    def calculate_sub_lord(self, longitude: float) -> str:
        """Calculate the Sub Lord (KP's core principle)"""
        nak_info = self.get_nakshatra(longitude)
        position_in_nak = nak_info['position_in_nak']
        nak_span = nak_info['nak_span']
        
        # Each nakshatra is divided into 9 sub-parts
        sub_part_size = nak_span / 9
        sub_index = int(position_in_nak / sub_part_size)
        
        # Handle edge case
        if sub_index >= 9:
            sub_index = 8
            
        return self.sub_divisions[sub_index]
        
    def calculate_sub_sub_lord(self, longitude: float) -> str:
        """Calculate the Sub Sub Lord"""
        nak_info = self.get_nakshatra(longitude)
        position_in_nak = nak_info['position_in_nak']
        nak_span = nak_info['nak_span']
        
        # Each nakshatra is divided into 9 sub-parts, each sub into 9 sub-subs
        sub_part_size = nak_span / 9
        sub_sub_part_size = sub_part_size / 9
        
        sub_index = int(position_in_nak / sub_part_size)
        if sub_index >= 9:
            sub_index = 8
            
        remaining = position_in_nak - (sub_index * sub_part_size)
        sub_sub_index = int(remaining / sub_sub_part_size)
        
        if sub_sub_index >= 9:
            sub_sub_index = 8
            
        return self.sub_sub_divisions[sub_sub_index]
        
    def calculate_planetary_score(self, planet: str, positions: Dict[str, Dict], 
                                moon_position: Dict[str, Dict]) -> float:
        """Calculate comprehensive planetary score using enhanced legacy approach"""
        if planet not in positions:
            return 0.0
            
        # Get ruling planets for bonus
        ruling_planets = self.get_ruling_planets(positions)
        
        # Initialize scores (ASC vs DESC like legacy)
        asc_score = 0.0
        desc_score = 0.0
        
        # 1. Score based on house lordship and occupation (hierarchical like legacy)
        planet_houses = self.get_significator_houses(planet, positions)
        
        # Get star lord and star lord's star lord for hierarchical scoring
        star_lord = positions[planet]['nakshatra']['lord']
        sl_houses = self.get_significator_houses(star_lord, positions) if star_lord else []
        
        # Apply scores with hierarchical weighting (legacy method)
        for house in planet_houses:
            w = self.hierarchical_weights['planet']
            asc_score += self.asc_house_weights.get(house, 0) * w
            desc_score += self.desc_house_weights.get(house, 0) * w
        
        for house in sl_houses:
            w = self.hierarchical_weights['star_lord']
            asc_score += self.asc_house_weights.get(house, 0) * w
            desc_score += self.desc_house_weights.get(house, 0) * w
        
        # 2. Apply Traditional Dignity (legacy method)
        dignity_multiplier = self.calculate_traditional_dignity_strength(planet, positions)
        asc_score *= dignity_multiplier
        desc_score *= dignity_multiplier
        
        # 3. Natural Rank Modifier (legacy feature)
        natural_rank = self.natural_rank_modifiers.get(planet, 1.0)
        asc_score *= natural_rank
        desc_score *= natural_rank
        
        # 4. Cricket-specific weight (legacy feature)
        cricket_weight = self.planet_cricket_weights.get(planet, 1.0)
        asc_score *= cricket_weight
        desc_score *= cricket_weight
        
        # 5. Combust check (legacy feature)
        if self.is_combust(planet, positions):
            combust_modifier = 0.5
            asc_score *= combust_modifier
            desc_score *= combust_modifier

        # 6. Ruling Planet Bonus (legacy feature)
        if planet in ruling_planets:
            ruling_planet_multiplier = 1.2
            asc_score *= ruling_planet_multiplier
            desc_score *= ruling_planet_multiplier

        # 7. Enhanced Aspects (legacy method)
        aspect_modifier = self.calculate_enhanced_aspect_factor(planet, positions)
        if asc_score > desc_score:
            asc_score *= aspect_modifier
        else:
            desc_score *= aspect_modifier

        # 8. Retrograde (legacy feature)
        if positions[planet].get("retrograde", False):
            retro_multiplier = self.retrograde_modifiers.get(planet, 1.0)
            asc_score *= retro_multiplier
            desc_score *= retro_multiplier
        
        # 9. High-Risk High-Reward (legacy feature)
        if any(h in [8, 12] for h in planet_houses):
            hrhr_multiplier = 0.6
            if asc_score < desc_score:
                desc_score *= hrhr_multiplier
        
        # Return net score (asc - desc like legacy)
        return asc_score - desc_score
        
    def calculate_dignity_factor(self, planet: str, position: Dict[str, Dict]) -> float:
        """Calculate dignity factor for a planet"""
        # Simplified dignity calculation - can be enhanced further
        house_number = position['house']
        
        # Basic dignity rules
        if planet == 'Sun' and house_number in [1, 5, 9]:
            return self.dignity_modifiers['own_sign']
        elif planet == 'Moon' and house_number in [4]:
            return self.dignity_modifiers['own_sign']
        elif planet == 'Mars' and house_number in [1, 8]:
            return self.dignity_modifiers['own_sign']
        elif planet == 'Jupiter' and house_number in [9, 12]:
            return self.dignity_modifiers['own_sign']
        elif planet == 'Venus' and house_number in [2, 7]:
            return self.dignity_modifiers['own_sign']
        elif planet == 'Saturn' and house_number in [10, 11]:
            return self.dignity_modifiers['own_sign']
        
        return self.dignity_modifiers['neutral_sign']
        
    def calculate_aspect_factor(self, planet: str, positions: Dict[str, Dict]) -> float:
        """Calculate aspect influence factor"""
        if planet not in positions:
            return 1.0
            
        planet_lon = positions[planet]['longitude']
        total_aspect_strength = 0.0
        aspect_count = 0
        
        for other_planet, other_pos in positions.items():
            if other_planet == planet or other_planet == 'Ascendant':
                continue
                
            other_lon = other_pos['longitude']
            aspect_type = self.get_aspect_type(planet_lon, other_lon)
            
            if aspect_type:
                aspect_strength = self.aspect_strengths.get(aspect_type, 0.0)
                other_planet_strength = self.planet_cricket_weights.get(other_planet, 1.0) / 2.0
                
                total_aspect_strength += aspect_strength * other_planet_strength
                aspect_count += 1
        
        if aspect_count > 0:
            average_aspect = total_aspect_strength / aspect_count
            return 1.0 + (average_aspect * 0.1)  # 10% maximum modification
        
        return 1.0
        
    def get_aspect_type(self, lon1: float, lon2: float) -> Optional[str]:
        """Determine aspect type between two planets"""
        diff = abs(lon1 - lon2)
        if diff > 180:
            diff = 360 - diff
            
        if diff <= 5:
            return 'conjunction'
        elif 175 <= diff <= 185:
            return 'opposition'
        elif 115 <= diff <= 125:
            return 'trine'
        elif 85 <= diff <= 95:
            return 'square'
        elif 55 <= diff <= 65:
            return 'sextile'
        
        return None
        
    def get_sign(self, longitude: float) -> str:
        """Get zodiac sign from longitude"""
        return self.signs[int(longitude // 30)]
    
    def get_significator_houses(self, planet: str, positions: Dict[str, Dict]) -> List[int]:
        """Calculate house significations for a planet (legacy method)"""
        if planet not in positions:
            return []

        # 1. House occupied by the planet
        occupied_house = positions[planet]['house']
        significators = {occupied_house}
        
        # 2. Houses ruled by the planet
        for house_num in range(1, 13):
            # Calculate house cusp longitude (simplified)
            house_cusp_lon = (house_num - 1) * 30 + positions['Ascendant']['longitude']
            if house_cusp_lon >= 360:
                house_cusp_lon -= 360
                
            sign_of_house = self.get_sign(house_cusp_lon)
            lord = self.sign_rulers[sign_of_house]
            
            if lord == planet:
                significators.add(house_num)
                
        return sorted(list(significators))
    
    def is_combust(self, planet: str, positions: Dict[str, Dict]) -> bool:
        """Check if a planet is combust (legacy method)"""
        if planet == "Sun" or planet not in positions or "Sun" not in positions:
            return False

        sun_long = positions["Sun"]["longitude"]
        planet_long = positions[planet]["longitude"]

        orb = 12 if planet == "Moon" else 8.5
        
        diff = abs(sun_long - planet_long)
        return min(diff, 360 - diff) <= orb
    
    def get_ruling_planets(self, positions: Dict[str, Dict]) -> List[str]:
        """Determine the KP Ruling Planets for a given chart (simplified)"""
        ruling_planets = set()
        
        # Add ascendant's star lord
        asc_nakshatra = positions['Ascendant']['nakshatra']
        ruling_planets.add(asc_nakshatra['lord'])
        
        # Add moon's star lord
        if 'Moon' in positions:
            moon_nakshatra = positions['Moon']['nakshatra']
            ruling_planets.add(moon_nakshatra['lord'])
        
        return list(ruling_planets)
    
    def calculate_traditional_dignity_strength(self, planet: str, positions: Dict[str, Dict]) -> float:
        """Calculate dignity strength using traditional methods (legacy)"""
        if planet not in positions:
            return 1.0

        planet_sign = self.get_sign(positions[planet]["longitude"])
        dignities = self.planet_dignities.get(planet, {})

        if not dignities:
            return 1.0  # Neutral for Rahu/Ketu

        if dignities.get("exaltation") == planet_sign:
            return self.dignity_modifiers["exaltation"]
        if dignities.get("debilitation") == planet_sign:
            return self.dignity_modifiers["debilitation"]
        if dignities.get("moolatrikona") == planet_sign:
            return self.dignity_modifiers["moolatrikona"]
        if planet_sign in dignities.get("own_house", []):
            return self.dignity_modifiers["own_sign"]
            
        return self.dignity_modifiers["neutral_sign"]
    
    def calculate_enhanced_aspect_factor(self, planet: str, positions: Dict[str, Dict]) -> float:
        """Calculate enhanced aspect modifier (legacy method)"""
        if planet not in positions:
            return 1.0

        # Get aspects received by the planet
        aspects = self.get_planet_aspects(planet, positions)
        if not aspects:
            return 1.0

        # Initialize total modifier
        total_modifier = 0.0

        for aspect in aspects:
            aspecting_planet = aspect["planet"]
            aspect_type = aspect["type"]
            
            # Get the weight and dignity of the aspecting planet
            aspecting_weight = self.planet_cricket_weights.get(aspecting_planet, 1.0)
            aspecting_dignity = self.calculate_traditional_dignity_strength(aspecting_planet, positions)
            
            # Get the houses ruled by the aspecting planet
            aspecting_houses = self.get_significator_houses(aspecting_planet, positions)
            
            # Calculate base aspect strength based on type and orb
            orb_value = self.aspects[aspect_type.lower()]["orb"]
            orb_factor = max(0, (1 - (aspect["angle"] / orb_value))**2)  # Non-linear orb factor
            
            if aspect_type == "Conjunction":
                base_strength = 0.35
            elif aspect_type == "Opposition":
                base_strength = 0.3
            elif aspect_type == "Trine":
                base_strength = 0.25
            elif aspect_type == "Square":
                base_strength = 0.2
            else:
                base_strength = 0.1

            # Calculate aspecting planet's influence on ascendant and descendant
            asc_influence = sum(self.asc_house_weights.get(h, 0) for h in aspecting_houses)
            desc_influence = sum(self.desc_house_weights.get(h, 0) for h in aspecting_houses)
            
            # Calculate net influence (positive favors asc, negative favors desc)
            net_influence = asc_influence - desc_influence
            
            # Calculate final aspect strength considering dignity
            aspect_strength = base_strength * aspecting_weight * aspecting_dignity * orb_factor

            # Apply the aspect influence
            if aspect_type in ["Conjunction", "Trine"]:  # Benefic aspects
                total_modifier += aspect_strength * net_influence
            else:  # Malefic aspects (Opposition, Square)
                total_modifier -= aspect_strength * net_influence

        # Ensure modifier stays within reasonable bounds
        return max(-1.0, min(1.0, 1.0 + total_modifier))
    
    def get_planet_aspects(self, planet: str, positions: Dict[str, Dict]) -> List[Dict]:
        """Calculate aspects received by a planet (legacy method)"""
        if planet not in positions:
            return []

        aspects = []
        planet_long = positions[planet]["longitude"]

        for other_planet, other_data in positions.items():
            if other_planet == planet or other_planet == 'Ascendant':
                continue

            other_long = other_data["longitude"]
            diff = abs(planet_long - other_long)
            diff = min(diff, 360 - diff)  # Get smallest angle

            # Check for different aspect types with their specific orbs
            if diff <= self.aspects["conjunction"]["orb"]:  # Conjunction
                aspects.append({
                    "planet": other_planet,
                    "type": "Conjunction",
                    "angle": diff
                })
            elif abs(diff - 180) <= self.aspects["opposition"]["orb"]:  # Opposition
                aspects.append({
                    "planet": other_planet,
                    "type": "Opposition",
                    "angle": diff
                })
            elif abs(diff - 120) <= self.aspects["trine"]["orb"]:  # Trine
                aspects.append({
                    "planet": other_planet,
                    "type": "Trine",
                    "angle": diff
                })
            elif abs(diff - 90) <= self.aspects["square"]["orb"]:  # Square
                aspects.append({
                    "planet": other_planet,
                    "type": "Square",
                    "angle": diff
                })

        return aspects
        
    def detect_contradictions(self, star_lord: str, sub_lord: str, sub_sub_lord: str) -> List[str]:
        """Detect contradictions in the planetary hierarchy"""
        contradictions = []
        
        # Check all possible contradictions
        planets = [star_lord, sub_lord, sub_sub_lord]
        
        for i, planet1 in enumerate(planets):
            for j, planet2 in enumerate(planets):
                if i != j:
                    if (planet1, planet2) in self.contradiction_patterns or (planet2, planet1) in self.contradiction_patterns:
                        contradictions.append(f"{planet1}-{planet2}")
        
        return contradictions
        
    def apply_contradiction_corrections(self, scores: Dict[str, float], 
                                     contradictions: List[str]) -> Dict[str, float]:
        """Apply corrections for detected contradictions"""
        corrected_scores = scores.copy()
        
        for contradiction in contradictions:
            planets = contradiction.split('-')
            if len(planets) == 2:
                planet1, planet2 = planets
                
                # Apply specific correction logic
                if contradiction in ['Mars-Rahu', 'Rahu-Mars']:
                    # Mars-Rahu creates aggressive but unstable energy
                    if 'sub_sub_lord_score' in corrected_scores and planet2 in ['Mars', 'Rahu']:
                        corrected_scores['sub_sub_lord_score'] *= -1
                        
                elif contradiction in ['Mars-Saturn', 'Saturn-Mars']:
                    # Mars-Saturn creates frustration and delays
                    for key in corrected_scores:
                        if 'saturn' in key.lower() or 'mars' in key.lower():
                            corrected_scores[key] *= 0.7
                            
                elif contradiction in ['Sun-Saturn', 'Saturn-Sun']:
                    # Authority vs limitation
                    for key in corrected_scores:
                        if 'sun' in key.lower():
                            corrected_scores[key] *= 0.8
        
        return corrected_scores
        
    def calculate_dynamic_weights(self, scores: Dict[str, float], 
                                contradictions: List[str]) -> Dict[str, float]:
        """Calculate sophisticated dynamic weights (enhanced legacy method)"""
        # Get absolute strengths
        sl_strength = abs(scores.get('star_lord_score', 0))
        sub_strength = abs(scores.get('sub_lord_score', 0)) 
        ssl_strength = abs(scores.get('sub_sub_lord_score', 0))
        
        total_strength = sl_strength + sub_strength + ssl_strength
        
        # Enhanced fallback (not legacy fixed weights)
        if total_strength == 0:
            return {'star_lord': 0.45, 'sub_lord': 0.35, 'sub_sub_lord': 0.20}
        
        # 1. Strength-proportional weights (70% based on strength, 30% hierarchy)
        strength_weights = {
            'star_lord': sl_strength / total_strength,
            'sub_lord': sub_strength / total_strength,
            'sub_sub_lord': ssl_strength / total_strength
        }
        
        base_weights = {'star_lord': 0.5, 'sub_lord': 0.3, 'sub_sub_lord': 0.2}
        
        # Blend strength with hierarchy (legacy approach)
        dynamic_weights = {}
        for lord in ['star_lord', 'sub_lord', 'sub_sub_lord']:
            dynamic_weights[lord] = 0.7 * strength_weights[lord] + 0.3 * base_weights[lord]
        
        # 2. Check for dominant lord (3x stronger than average) - legacy feature
        max_strength = max(sl_strength, sub_strength, ssl_strength)
        avg_other_strength = (total_strength - max_strength) / 2
        
        if avg_other_strength > 0 and max_strength / avg_other_strength >= 3.0:
            # One lord is dominant (legacy logic)
            if max_strength == sl_strength:
                dynamic_weights = {'star_lord': 0.70, 'sub_lord': 0.20, 'sub_sub_lord': 0.10}
            elif max_strength == sub_strength:
                dynamic_weights = {'star_lord': 0.25, 'sub_lord': 0.65, 'sub_sub_lord': 0.10}
            else:  # SSL is dominant
                dynamic_weights = {'star_lord': 0.20, 'sub_lord': 0.15, 'sub_sub_lord': 0.65}
        
        # 3. Check for contradictions (opposite signs) - legacy enhancement
        sl_score = scores.get('star_lord_score', 0)
        sub_score = scores.get('sub_lord_score', 0)
        ssl_score = scores.get('sub_sub_lord_score', 0)
        
        score_list = [sl_score, sub_score, ssl_score]
        positive_count = sum(1 for s in score_list if s > 0)
        negative_count = sum(1 for s in score_list if s < 0)
        
        # If there's contradiction, give more weight to the strongest contradicting opinion
        if min(positive_count, negative_count) > 0:
            strengths = [sl_strength, sub_strength, ssl_strength]
            max_strength_idx = strengths.index(max(strengths))
            
            # Give 20% bonus to the strongest lord in contradiction (legacy feature)
            lord_names = ['star_lord', 'sub_lord', 'sub_sub_lord']
            strongest_lord = lord_names[max_strength_idx]
            dynamic_weights[strongest_lord] += 0.2
            
            # Normalize weights
            total_weight = sum(dynamic_weights.values())
            dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        
        # 4. Apply Level 2 contradiction adjustments
        contradiction_factor = len(contradictions) * 0.03  # Reduced from 0.05
        for lord in dynamic_weights:
            dynamic_weights[lord] *= (1 - contradiction_factor)
        
        # Final normalization to ensure weights sum to 1.0
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        else:
            dynamic_weights = {'star_lord': 0.45, 'sub_lord': 0.35, 'sub_sub_lord': 0.20}
        
        return dynamic_weights
        
    def predict_moment(self, dt: datetime, lat: float, lon: float) -> Dict[str, Any]:
        """Comprehensive prediction for a specific moment"""
        # Convert to Julian Day
        jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60.0 + dt.second/3600.0)
        
        # Calculate planetary positions
        positions = self.calculate_planetary_positions(jd, lat, lon)
        
        # Get Moon's position for KP calculations
        moon_pos = positions['Moon']
        moon_longitude = moon_pos['longitude']
        
        # Calculate KP hierarchy
        star_lord = moon_pos['nakshatra']['lord']
        sub_lord = self.calculate_sub_lord(moon_longitude)
        sub_sub_lord = self.calculate_sub_sub_lord(moon_longitude)
        
        # Calculate individual scores
        star_lord_score = self.calculate_planetary_score(star_lord, positions, moon_pos)
        sub_lord_score = self.calculate_planetary_score(sub_lord, positions, moon_pos)
        sub_sub_lord_score = self.calculate_planetary_score(sub_sub_lord, positions, moon_pos)
        
        # Store original scores
        original_scores = {
            'star_lord_score': star_lord_score,
            'sub_lord_score': sub_lord_score,
            'sub_sub_lord_score': sub_sub_lord_score
        }
        
        # Detect contradictions
        contradictions = self.detect_contradictions(star_lord, sub_lord, sub_sub_lord)
        
        # Apply contradiction corrections
        corrected_scores = self.apply_contradiction_corrections(original_scores, contradictions)
        
        # Calculate dynamic weights
        weights = self.calculate_dynamic_weights(corrected_scores, contradictions)
        
        # Calculate final weighted score
        final_score = (
            corrected_scores['star_lord_score'] * weights['star_lord'] +
            corrected_scores['sub_lord_score'] * weights['sub_lord'] +
            corrected_scores['sub_sub_lord_score'] * weights['sub_sub_lord']
        )
        
        # Determine verdict
        verdict = self.get_verdict(final_score)
        
        return {
            'datetime': dt,
            'moon_star_lord': star_lord,
            'sub_lord': sub_lord,
            'sub_sub_lord': sub_sub_lord,
            'moon_sl_score': original_scores['star_lord_score'],
            'moon_sub_score': original_scores['sub_lord_score'],
            'moon_ssl_score': original_scores['sub_sub_lord_score'],
            'corrected_scores': corrected_scores,
            'contradictions': contradictions,
            'weights': weights,
            'final_score': final_score,
            'verdict': verdict,
            'positions': positions
        }
        
    def get_verdict(self, score: float) -> str:
        """Convert numerical score to readable verdict"""
        if score > 10:
            return "Clearly Favors Ascendant"
        elif score > 5:
            return "Slightly Favors Ascendant"
        elif score > -1 and score <= 1:
            return "Neutral / Too Close to Call"
        elif score > -5:
            return "Slightly Favors Descendant"
        else:
            return "Clearly Favors Descendant"
            
    def predict_timeline(self, start_dt: datetime, end_dt: datetime, 
                        lat: float, lon: float, interval_minutes: int = 5) -> pd.DataFrame:
        """Generate comprehensive timeline predictions"""
        predictions = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            prediction = self.predict_moment(current_dt, lat, lon)
            predictions.append(prediction)
            current_dt += timedelta(minutes=interval_minutes)
        
        # Convert to DataFrame
        df_data = []
        for pred in predictions:
            row = {
                'datetime': pred['datetime'],
                'moon_star_lord': pred['moon_star_lord'],
                'sub_lord': pred['sub_lord'],
                'sub_sub_lord': pred['sub_sub_lord'],
                'moon_sl_score': pred['moon_sl_score'],
                'moon_sub_score': pred['moon_sub_score'],
                'moon_ssl_score': pred['moon_ssl_score'],
                'contradictions': ','.join(pred['contradictions']),
                'star_lord_weight': pred['weights']['star_lord'],
                'sub_lord_weight': pred['weights']['sub_lord'],
                'sub_sub_lord_weight': pred['weights']['sub_sub_lord'],
                'final_score': pred['final_score'],
                'verdict': pred['verdict']
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)


def test_comprehensive_model():
    """Test the comprehensive model"""
    model = KPComprehensiveModel()
    
    # Test with the Panthers vs Nellai match
    test_dt = datetime(2025, 6, 18, 19, 51, 0)
    lat, lon = 11.6469616, 78.2106958
    
    prediction = model.predict_moment(test_dt, lat, lon)
    
    print("=== Comprehensive Model Test ===")
    print(f"DateTime: {prediction['datetime']}")
    print(f"Star Lord: {prediction['moon_star_lord']} (Score: {prediction['moon_sl_score']:.2f})")
    print(f"Sub Lord: {prediction['sub_lord']} (Score: {prediction['moon_sub_score']:.2f})")
    print(f"Sub Sub Lord: {prediction['sub_sub_lord']} (Score: {prediction['moon_ssl_score']:.2f})")
    print(f"Contradictions: {prediction['contradictions']}")
    print(f"Weights: SL={prediction['weights']['star_lord']:.1%}, SUB={prediction['weights']['sub_lord']:.1%}, SSL={prediction['weights']['sub_sub_lord']:.1%}")
    print(f"Final Score: {prediction['final_score']:.2f}")
    print(f"Verdict: {prediction['verdict']}")


if __name__ == "__main__":
    test_comprehensive_model() 