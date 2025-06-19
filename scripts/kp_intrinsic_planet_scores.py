#!/usr/bin/env python3

"""
KP Intrinsic Planetary Scores System
=====================================

This module provides FIXED intrinsic planetary scores based on KP astrological principles.
Unlike the current system that calculates scores dynamically based on changing house positions,
this provides consistent planetary nature scores that remain constant.

Key KP Principles:
1. Each planet has an intrinsic nature (benefic/malefic)
2. This nature doesn't change every 5 minutes based on house transit
3. Contradictions modify the expression, not the base nature
4. Planets maintain consistent scores regardless of lordship position
"""

from typing import Dict, Tuple
import pandas as pd

class KPIntrinsicPlanetarySystem:
    """KP system with fixed intrinsic planetary scores"""
    
    def __init__(self):
        """Initialize the intrinsic planetary scoring system"""
        self.setup_intrinsic_scores()
        self.setup_natural_significations()
    
    def setup_intrinsic_scores(self):
        """
        Define intrinsic planetary scores based on KP principles.
        These scores are FIXED and don't change based on house positions.
        """
        # Based on natural benefic/malefic nature and sporting context
        self.intrinsic_scores = {
            'Sun': {
                'base_score': 12.0,  # Natural leader, authority, confidence
                'nature': 'benefic',
                'sporting_strength': 'high',  # Leadership, winning attitude
                'kp_description': 'Natural leader, brings authority and confidence'
            },
            
            'Moon': {
                'base_score': 2.0,   # Mind, emotions - neutral but fluctuating
                'nature': 'neutral', 
                'sporting_strength': 'variable',  # Mental state affects performance
                'kp_description': 'Mind and emotions, can be supportive or disruptive'
            },
            
            'Mars': {
                'base_score': 8.0,   # Energy, aggression, competition - good for sports
                'nature': 'malefic_functional',  # Malefic but functional for competition
                'sporting_strength': 'high',  # Physical energy, competitive spirit
                'kp_description': 'Energy and competition, excellent for sporting events'
            },
            
            'Mercury': {
                'base_score': 4.0,   # Communication, quick thinking - moderate benefit
                'nature': 'neutral',
                'sporting_strength': 'moderate',  # Quick decisions, strategy
                'kp_description': 'Quick thinking and strategy, moderately helpful'
            },
            
            'Jupiter': {
                'base_score': -8.0,  # Wisdom, expansion - can cause delays in immediate results
                'nature': 'benefic_delayed',  # Benefic but for long-term, not immediate
                'sporting_strength': 'low',  # Too philosophical for immediate competition
                'kp_description': 'Wisdom and expansion, but can delay immediate results'
            },
            
            'Venus': {
                'base_score': 6.0,   # Harmony, cooperation, luxury - moderate benefit
                'nature': 'benefic',
                'sporting_strength': 'moderate',  # Team harmony, aesthetic performance
                'kp_description': 'Harmony and cooperation, helps team performance'
            },
            
            'Saturn': {
                'base_score': -10.0, # Delays, obstacles, hard work - malefic for immediate results
                'nature': 'malefic',
                'sporting_strength': 'low',  # Restrictions, delays, obstacles
                'kp_description': 'Delays and obstacles, creates difficulties'
            },
            
            'Rahu': {
                'base_score': -6.0,  # Ambition, unconventional - generally malefic
                'nature': 'malefic',
                'sporting_strength': 'low',  # Unconventional, disruptive energy
                'kp_description': 'Unconventional energy, creates confusion and disruption'
            },
            
            'Ketu': {
                'base_score': -4.0,  # Spirituality, detachment - malefic for worldly matters
                'nature': 'malefic',
                'sporting_strength': 'low',  # Detachment from material success
                'kp_description': 'Spiritual detachment, reduces focus on material success'
            }
        }
    
    def setup_natural_significations(self):
        """
        Define what each planet naturally signifies in KP.
        These are FIXED and don't change based on chart positions.
        """
        self.natural_significations = {
            'Sun': ['authority', 'leadership', 'government', 'confidence', 'vitality'],
            'Moon': ['mind', 'emotions', 'public', 'mother', 'fluctuation'],
            'Mars': ['energy', 'competition', 'sports', 'courage', 'aggression'],
            'Mercury': ['communication', 'intelligence', 'business', 'adaptability'],
            'Jupiter': ['wisdom', 'knowledge', 'spirituality', 'expansion', 'delay'],
            'Venus': ['luxury', 'comfort', 'art', 'harmony', 'relationships'],
            'Saturn': ['discipline', 'delay', 'obstacles', 'hard_work', 'limitations'],
            'Rahu': ['ambition', 'foreign', 'technology', 'confusion', 'materialism'],
            'Ketu': ['spirituality', 'detachment', 'past_karma', 'isolation']
        }
    
    def get_planet_intrinsic_score(self, planet: str) -> float:
        """
        Get the intrinsic score for a planet.
        This score is FIXED and doesn't change based on chart positions.
        """
        if planet in self.intrinsic_scores:
            return self.intrinsic_scores[planet]['base_score']
        return 0.0
    
    def get_planet_nature(self, planet: str) -> str:
        """Get the natural KP nature of a planet"""
        if planet in self.intrinsic_scores:
            return self.intrinsic_scores[planet]['nature']
        return 'unknown'
    
    def get_planet_sporting_strength(self, planet: str) -> str:
        """Get how strong this planet is for sporting events"""
        if planet in self.intrinsic_scores:
            return self.intrinsic_scores[planet]['sporting_strength']
        return 'unknown'
    
    def get_planet_description(self, planet: str) -> str:
        """Get KP description of the planet's role"""
        if planet in self.intrinsic_scores:
            return self.intrinsic_scores[planet]['kp_description']
        return 'Unknown planet'
    
    def calculate_lord_scores(self, star_lord: str, sub_lord: str, sub_sub_lord: str) -> Dict[str, float]:
        """
        Calculate scores for the three lords using INTRINSIC scores.
        These scores are consistent regardless of time or chart position.
        """
        return {
            'star_lord_score': self.get_planet_intrinsic_score(star_lord),
            'sub_lord_score': self.get_planet_intrinsic_score(sub_lord),
            'sub_sub_lord_score': self.get_planet_intrinsic_score(sub_sub_lord)
        }
    
    def apply_kp_contradictions(self, star_lord: str, sub_lord: str, sub_sub_lord: str) -> Dict:
        """
        Apply KP contradictions to the intrinsic scores.
        This modifies the EXPRESSION of the planetary energy, not the base nature.
        """
        # Get base intrinsic scores
        base_scores = {
            'star_lord': self.get_planet_intrinsic_score(star_lord),
            'sub_lord': self.get_planet_intrinsic_score(sub_lord),
            'sub_sub_lord': self.get_planet_intrinsic_score(sub_sub_lord)
        }
        
        corrected_scores = base_scores.copy()
        lords = [star_lord, sub_lord, sub_sub_lord]
        contradictions_detected = []
        
        # Mars-Rahu: Both become strongly negative (explosive combination)
        if 'Mars' in lords and 'Rahu' in lords:
            contradictions_detected.append('Mars-Rahu')
            reference_score = max(abs(base_scores['star_lord']), 
                                abs(base_scores['sub_lord']), 
                                abs(base_scores['sub_sub_lord']))
            standardized_negative = -reference_score * 1.2
            
            for position in ['star_lord', 'sub_lord', 'sub_sub_lord']:
                lord_name = [star_lord, sub_lord, sub_sub_lord][['star_lord', 'sub_lord', 'sub_sub_lord'].index(position)]
                if lord_name in ['Mars', 'Rahu']:
                    corrected_scores[position] = standardized_negative
        
        # Jupiter-Rahu: Jupiter's wisdom gets corrupted
        elif 'Jupiter' in lords and 'Rahu' in lords:
            contradictions_detected.append('Jupiter-Rahu')
            for position in ['star_lord', 'sub_lord', 'sub_sub_lord']:
                lord_name = [star_lord, sub_lord, sub_sub_lord][['star_lord', 'sub_lord', 'sub_sub_lord'].index(position)]
                if lord_name == 'Jupiter':
                    corrected_scores[position] *= -0.8  # Partial corruption
        
        # Sun-Saturn: Authority vs limitations (mutual weakening)
        elif 'Sun' in lords and 'Saturn' in lords:
            contradictions_detected.append('Sun-Saturn')
            for position in ['star_lord', 'sub_lord', 'sub_sub_lord']:
                lord_name = [star_lord, sub_lord, sub_sub_lord][['star_lord', 'sub_lord', 'sub_sub_lord'].index(position)]
                if lord_name in ['Sun', 'Saturn']:
                    corrected_scores[position] *= 0.6  # Mutual weakening
        
        # Mars-Saturn: Energy gets blocked
        elif 'Mars' in lords and 'Saturn' in lords:
            contradictions_detected.append('Mars-Saturn')
            for position in ['star_lord', 'sub_lord', 'sub_sub_lord']:
                lord_name = [star_lord, sub_lord, sub_sub_lord][['star_lord', 'sub_lord', 'sub_sub_lord'].index(position)]
                if lord_name in ['Mars', 'Saturn']:
                    corrected_scores[position] *= 0.7  # Energy blockage
        
        # Venus-Mars: Harmony vs aggression
        elif 'Venus' in lords and 'Mars' in lords:
            contradictions_detected.append('Venus-Mars')
            for position in ['star_lord', 'sub_lord', 'sub_sub_lord']:
                lord_name = [star_lord, sub_lord, sub_sub_lord][['star_lord', 'sub_lord', 'sub_sub_lord'].index(position)]
                if lord_name in ['Venus', 'Mars']:
                    corrected_scores[position] *= 0.8  # Disharmony
        
        # Moon-Mars: Emotional volatility
        elif 'Moon' in lords and 'Mars' in lords:
            contradictions_detected.append('Moon-Mars')
            for position in ['star_lord', 'sub_lord', 'sub_sub_lord']:
                lord_name = [star_lord, sub_lord, sub_sub_lord][['star_lord', 'sub_lord', 'sub_sub_lord'].index(position)]
                if lord_name in ['Moon', 'Mars']:
                    corrected_scores[position] *= 0.75  # Volatility
        
        return {
            'base_scores': base_scores,
            'corrected_scores': corrected_scores,
            'contradictions': contradictions_detected,
            'star_lord': star_lord,
            'sub_lord': sub_lord,
            'sub_sub_lord': sub_sub_lord
        }
    
    def calculate_final_score(self, star_lord: str, sub_lord: str, sub_sub_lord: str, 
                            weights: Dict[str, float] = None) -> Dict:
        """
        Calculate the final KP score using intrinsic planetary scores.
        
        Args:
            star_lord: Star lord planet name
            sub_lord: Sub lord planet name  
            sub_sub_lord: Sub sub lord planet name
            weights: Custom weights for lordship levels (default: 0.5, 0.3, 0.2)
        """
        if weights is None:
            weights = {'star_lord': 0.5, 'sub_lord': 0.3, 'sub_sub_lord': 0.2}
        
        # Apply contradictions to get corrected scores
        result = self.apply_kp_contradictions(star_lord, sub_lord, sub_sub_lord)
        corrected_scores = result['corrected_scores']
        
        # Calculate weighted final score
        final_score = (
            corrected_scores['star_lord'] * weights['star_lord'] +
            corrected_scores['sub_lord'] * weights['sub_lord'] +
            corrected_scores['sub_sub_lord'] * weights['sub_sub_lord']
        )
        
        result['weights'] = weights
        result['final_score'] = final_score
        result['verdict'] = self.get_verdict(final_score)
        
        return result
    
    def get_verdict(self, score: float) -> str:
        """Convert numerical score to verdict"""
        if score > 8:
            return "Clearly Favors Ascendant"
        elif score > 3:
            return "Slightly Favors Ascendant"  
        elif score > -3:
            return "Neutral / Too Close to Call"
        elif score > -8:
            return "Slightly Favors Descendant"
        else:
            return "Clearly Favors Descendant"
    
    def analyze_consistency(self, test_cases: list) -> bool:
        """
        Test that the same planet always gets the same base score 
        regardless of its lordship position.
        """
        planet_base_scores = {}
        
        for star_lord, sub_lord, sub_sub_lord in test_cases:
            result = self.apply_kp_contradictions(star_lord, sub_lord, sub_sub_lord)
            base_scores = result['base_scores']
            
            # Check each planet's base score consistency
            lords = {'star_lord': star_lord, 'sub_lord': sub_lord, 'sub_sub_lord': sub_sub_lord}
            
            for position, planet in lords.items():
                if planet not in planet_base_scores:
                    planet_base_scores[planet] = base_scores[position]
                else:
                    if abs(planet_base_scores[planet] - base_scores[position]) > 0.001:
                        print(f"INCONSISTENCY: {planet} has different base scores")
                        return False
        
        print("✅ All planets have consistent base scores across positions")
        return True


def test_intrinsic_system():
    """Test the intrinsic planetary scoring system"""
    print("=== TESTING KP INTRINSIC PLANETARY SYSTEM ===\n")
    
    system = KPIntrinsicPlanetarySystem()
    
    # Test 1: Show intrinsic scores
    print("1. INTRINSIC PLANETARY SCORES:")
    for planet, data in system.intrinsic_scores.items():
        print(f"   {planet}: {data['base_score']:.1f} ({data['nature']}) - {data['kp_description']}")
    print()
    
    # Test 2: Consistency check
    print("2. CONSISTENCY TEST:")
    test_cases = [
        ('Mars', 'Sun', 'Rahu'),
        ('Rahu', 'Mars', 'Sun'),
        ('Sun', 'Rahu', 'Mars'),
        ('Jupiter', 'Venus', 'Mercury'),
        ('Saturn', 'Moon', 'Ketu')
    ]
    
    system.analyze_consistency(test_cases)
    print()
    
    # Test 3: Mars-Rahu contradiction consistency
    print("3. MARS-RAHU CONTRADICTION TEST:")
    mars_rahu_cases = [
        ('Mars', 'Sun', 'Rahu'),   # Mars as SL, Rahu as SSL
        ('Rahu', 'Sun', 'Mars'),   # Rahu as SL, Mars as SSL
        ('Sun', 'Mars', 'Rahu'),   # Mars as Sub, Rahu as SSL
        ('Sun', 'Rahu', 'Mars')    # Rahu as Sub, Mars as SSL
    ]
    
    for case in mars_rahu_cases:
        result = system.calculate_final_score(*case)
        corrected = result['corrected_scores']
        
        mars_score = None
        rahu_score = None
        mars_pos = None
        rahu_pos = None
        
        for pos, planet in zip(['star_lord', 'sub_lord', 'sub_sub_lord'], case):
            if planet == 'Mars':
                mars_score = corrected[pos]
                mars_pos = pos
            elif planet == 'Rahu':
                rahu_score = corrected[pos]
                rahu_pos = pos
        
        print(f"Case {case}:")
        print(f"  Mars ({mars_pos}): {mars_score:.2f}")
        print(f"  Rahu ({rahu_pos}): {rahu_score:.2f}")
        
        if mars_score is not None and rahu_score is not None:
            if abs(mars_score - rahu_score) < 0.01:
                print(f"  ✅ IDENTICAL scores: {mars_score:.2f}")
            else:
                print(f"  ❌ DIFFERENT scores!")
        
        print(f"  Final Score: {result['final_score']:.2f} - {result['verdict']}")
        print()


if __name__ == "__main__":
    test_intrinsic_system() 