"""
KP Hybrid Model - Fixed Version
===============================

This model fixes the issues in the comprehensive model by:
1. Using original KP calculations for accurate Star/Sub/Sub-Sub lords
2. Adding enhanced contradiction detection
3. Implementing dynamic weighting system
4. Preventing duplicates with proper Sub Lord change timing
"""

import pandas as pd
from datetime import datetime, timedelta
import swisseph as swe
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Import original functions
sys.path.append(os.path.dirname(__file__))
from chart_generator import generate_kp_chart
from kp_favorability_rules import evaluate_favorability
from kp_contradiction_system import KPContradictionSystem


class KPHybridModel:
    """Fixed hybrid model with original calculations + enhancements"""
    
    def __init__(self):
        """Initialize the hybrid model"""
        self.contradiction_system = KPContradictionSystem()
        self.setup_enhancement_constants()
        
    def setup_enhancement_constants(self):
        """Setup enhanced contradiction patterns and dynamic weighting rules"""
        # Enhanced contradiction patterns (beyond the original 4)
        self.enhanced_contradictions = [
            ('Mars', 'Rahu'),      # Original: Aggressive contradiction
            ('Mars', 'Saturn'),    # Original: Energy vs Restriction  
            ('Sun', 'Saturn'),     # Original: Authority vs Limitation
            ('Moon', 'Mars'),      # Original: Emotion vs Aggression
            ('Jupiter', 'Rahu'),   # Enhanced: Wisdom vs Materialism
            ('Venus', 'Mars'),     # Enhanced: Harmony vs Conflict
            ('Mercury', 'Jupiter') # Enhanced: Quick thinking vs Deep wisdom
        ]
        
        # Dynamic weighting factors
        self.weight_bounds = {
            'star_lord': (0.1, 0.7),
            'sub_lord': (0.15, 0.6),  # KP principle: Sub Lord should have decent weight
            'sub_sub_lord': (0.1, 0.5)
        }
        
    def apply_level2_lordship_contradictions(self, sl_score: float, sub_score: float, ssl_score: float,
                                           star_lord: str, sub_lord: str, sub_sub_lord: str, chart: dict = None) -> Tuple[float, float, float]:
        """
        Apply Level 2 lordship hierarchy contradictions to individual lord scores.
        
        This is a simplified approach that applies lordship contradictions directly to scores
        rather than through weights, for backward compatibility with existing system.
        """
        corrected_sl, corrected_sub, corrected_ssl = sl_score, sub_score, ssl_score
        lords = [star_lord, sub_lord, sub_sub_lord]
        scores = [corrected_sl, corrected_sub, corrected_ssl]
        
        # Check for Level 1 contradiction precedence first
        level1_pairs = []
        if chart:
            level1_pairs = self.contradiction_system.has_level1_contradictions(chart, star_lord, sub_lord, sub_sub_lord)
        
        # Detect lordship contradictions  
        lordship_contradictions = self.contradiction_system.detect_lordship_contradictions(
            star_lord, sub_lord, sub_sub_lord
        )
        
        # Apply lordship contradictions ONLY if no Level 1 contradictions exist for the same pairs
        for contradiction in lordship_contradictions:
            planets = contradiction['planets']
            planetary_pair = f"{planets[0]}-{planets[1]}"
            reverse_pair = f"{planets[1]}-{planets[0]}"
            
            # Check if this planetary pair already has Level 1 contradiction
            if planetary_pair in level1_pairs or reverse_pair in level1_pairs:
                # Skip Level 2 - Level 1 already applied
                continue
                
            if contradiction['effect'] == 'calculation_instability':
                # Mercury-Jupiter: Both get reduced for calculation instability
                multiplier = contradiction['multiplier']
                
                for i, lord in enumerate(lords):
                    if lord in planets:
                        scores[i] *= multiplier
                        # Log contradiction application if needed
                        pass
        
        return tuple(scores)
        
    def calculate_dynamic_weights(self, sl_score: float, sub_score: float, ssl_score: float,
                                contradictions: List[str], star_lord: str = None, sub_lord: str = None, 
                                sub_sub_lord: str = None) -> Dict[str, float]:
        """
        Calculate dynamic weights based on planetary strength and relationships.
        This handles FINAL SCORE CONTRADICTIONS that affect the calculation flow.
        """
        sl_strength = abs(sl_score)
        sub_strength = abs(sub_score)
        ssl_strength = abs(ssl_score)
        
        total_strength = sl_strength + sub_strength + ssl_strength
        if total_strength == 0:
            return {'star_lord': 0.5, 'sub_lord': 0.3, 'sub_sub_lord': 0.2}
        
        # Base proportional weights
        base_sl = sl_strength / total_strength
        base_sub = sub_strength / total_strength
        base_ssl = ssl_strength / total_strength
        
        # Ensure Sub Lord has minimum 25% weight (KP principle)
        adjusted_sub = max(base_sub, 0.25)
        
        # Redistribute remaining weight
        remaining = 1.0 - adjusted_sub
        sl_portion = base_sl / (base_sl + base_ssl) if (base_sl + base_ssl) > 0 else 0.5
        
        final_sl = remaining * sl_portion
        final_ssl = remaining * (1 - sl_portion)
        
        # Apply bounds
        weights = {
            'star_lord': max(self.weight_bounds['star_lord'][0], 
                           min(self.weight_bounds['star_lord'][1], final_sl)),
            'sub_lord': max(self.weight_bounds['sub_lord'][0], 
                          min(self.weight_bounds['sub_lord'][1], adjusted_sub)),
            'sub_sub_lord': max(self.weight_bounds['sub_sub_lord'][0], 
                              min(self.weight_bounds['sub_sub_lord'][1], final_ssl))
        }
        
        # ========== LEVEL 3: FINAL SCORE CONTRADICTIONS ==========
        # These affect the weighted calculation process
        
        if star_lord and sub_lord and sub_sub_lord:
            lords = [star_lord, sub_lord, sub_sub_lord]
            
            # Mercury-Jupiter: Quick thinking vs Deep wisdom (affects calculation flow)
            if "Mercury" in lords and "Jupiter" in lords:
                # This creates calculation instability - reduce the weaker lord's influence
                mercury_positions = [i for i, lord in enumerate(lords) if lord == "Mercury"]
                jupiter_positions = [i for i, lord in enumerate(lords) if lord == "Jupiter"]
                
                lord_names = ['star_lord', 'sub_lord', 'sub_sub_lord']
                scores = [sl_score, sub_score, ssl_score]
                strengths = [sl_strength, sub_strength, ssl_strength]
                max_strength = max(strengths)
                
                for pos in mercury_positions:
                    if abs(scores[pos]) < max_strength:
                        weights[lord_names[pos]] *= 0.8  # Reduce weaker Mercury
                
                for pos in jupiter_positions:
                    if abs(scores[pos]) < max_strength:
                        weights[lord_names[pos]] *= 0.8  # Reduce weaker Jupiter
        
        # Normalize
        weight_sum = sum(weights.values())
        return {k: v/weight_sum for k, v in weights.items()}
        
    def enhance_favorability_result(self, original_result: Dict, nakshatra_df: pd.DataFrame) -> Dict:
        """
        Enhance the original favorability result with proper contradiction hierarchy
        
        Level 1: Planet-level contradictions (would be applied in favorability calculation)
        Level 2: Lordship hierarchy contradictions (applied here to final calculation)
        """
        # Extract original values
        sl_score = original_result.get('moon_sl_score', 0)
        sub_score = original_result.get('moon_sub_score', 0)
        ssl_score = original_result.get('moon_ssl_score', 0)
        
        # Get planetary hierarchy from original calculation
        star_lord = original_result.get('moon_star_lord', 'Unknown')
        sub_lord = original_result.get('moon_sub_lord', 'Unknown')
        sub_sub_lord = original_result.get('moon_sub_sub_lord', 'Unknown')
        
        # Apply Level 2 lordship contradictions to individual scores
        corrected_sl, corrected_sub, corrected_ssl = self.apply_level2_lordship_contradictions(
            sl_score, sub_score, ssl_score, star_lord, sub_lord, sub_sub_lord, None)
        
        # Calculate dynamic weights
        weights = self.calculate_dynamic_weights(corrected_sl, corrected_sub, corrected_ssl, [], 
                                               star_lord, sub_lord, sub_sub_lord)
        
        # Calculate enhanced final score
        enhanced_final_score = (
            corrected_sl * weights['star_lord'] +
            corrected_sub * weights['sub_lord'] +
            corrected_ssl * weights['sub_sub_lord']
        )
        
        # Create enhanced result
        enhanced_result = original_result.copy()
        enhanced_result.update({
            'corrected_sl_score': corrected_sl,
            'corrected_sub_score': corrected_sub,
            'corrected_ssl_score': corrected_ssl,
            'dynamic_weights': weights,
            'enhanced_final_score': enhanced_final_score,
            'original_final_score': original_result.get('final_score', 0)
        })
        
        return enhanced_result
        
    def predict_timeline_hybrid(self, start_dt: datetime, lat: float, lon: float,
                               duration_hours: int = 4, nakshatra_df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate accurate timeline using original KP + enhancements"""
        if nakshatra_df is None:
            # Load nakshatra data
            nakshatra_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'nakshatra_sub_lords_longitudes.csv')
            nakshatra_df = pd.read_csv(nakshatra_path)
        
        timeline_data = []
        current_dt = start_dt
        end_dt = start_dt + timedelta(hours=duration_hours)
        
        # Generate muhurta chart (match start)
        muhurta_chart = generate_kp_chart(start_dt, lat, lon, nakshatra_df)
        if "error" in muhurta_chart:
            raise Exception(f"Error: {muhurta_chart['error']}")
        
        last_lords = None
        last_time = None
        
        while current_dt < end_dt:
            # Generate current chart
            current_chart = generate_kp_chart(current_dt, lat, lon, nakshatra_df)
            if "error" in current_chart:
                current_dt += timedelta(minutes=1)
                continue
            
            # Get original favorability WITHOUT enhanced rules to avoid double-applying contradictions
            original_result = evaluate_favorability(muhurta_chart, current_chart, nakshatra_df, use_enhanced_rules=False)
            
            # Extract lords from original system
            star_lord = current_chart.get('moon_star_lord')
            sub_lord = current_chart.get('moon_sub_lord') 
            sub_sub_lord = current_chart.get('moon_sub_sub_lord')
            
            current_lords = (star_lord, sub_lord, sub_sub_lord)
            
            # Skip exact duplicates with time check
            if (current_lords == last_lords and last_time and 
                (current_dt - last_time).total_seconds() < 60):
                try:
                    next_dt = self.find_next_ssl_change(current_dt, lat, lon, nakshatra_df)
                    current_dt = max(next_dt, current_dt + timedelta(minutes=1))
                except:
                    current_dt += timedelta(minutes=30)
                continue
            
            # Apply enhancements
            sl_score = original_result.get('moon_sl_score', 0)
            sub_score = original_result.get('moon_sub_score', 0)
            ssl_score = original_result.get('moon_ssl_score', 0)
            
            # Apply Level 2 lordship contradictions
            corrected_sl, corrected_sub, corrected_ssl = self.apply_level2_lordship_contradictions(
                sl_score, sub_score, ssl_score, star_lord, sub_lord, sub_sub_lord, None)
            
            weights = self.calculate_dynamic_weights(corrected_sl, corrected_sub, corrected_ssl, [],
                                                    star_lord, sub_lord, sub_sub_lord)
            
            enhanced_final_score = (
                corrected_sl * weights['star_lord'] +
                corrected_sub * weights['sub_lord'] +
                corrected_ssl * weights['sub_sub_lord']
            )
            
            # Create timeline entry
            timeline_row = {
                'datetime': current_dt,
                'moon_star_lord': star_lord,
                'sub_lord': sub_lord,
                'sub_sub_lord': sub_sub_lord,
                'moon_sl_score': corrected_sl,
                'moon_sub_score': corrected_sub,
                'moon_ssl_score': corrected_ssl,
                'contradictions': 'Level 2 applied',
                'star_lord_weight': weights['star_lord'],
                'sub_lord_weight': weights['sub_lord'],
                'sub_sub_lord_weight': weights['sub_sub_lord'],
                'final_score': enhanced_final_score,
                'original_final_score': original_result.get('final_score', 0)
            }
            
            timeline_data.append(timeline_row)
            
            # Update tracking
            last_lords = current_lords
            last_time = current_dt
            
            # Find next change
            try:
                next_dt = self.find_next_ssl_change(current_dt, lat, lon, nakshatra_df)
                current_dt = max(next_dt, current_dt + timedelta(minutes=1))
            except:
                current_dt += timedelta(minutes=30)
        
        df = pd.DataFrame(timeline_data)
        if not df.empty:
            df['verdict'] = df['final_score'].apply(self.get_verdict)
        
        return df
        
    def find_next_ssl_change(self, current_dt: datetime, lat: float, lon: float, nakshatra_df: pd.DataFrame):
        """Find next Sub-Sub Lord change timing"""
        try:
            jd = swe.julday(current_dt.year, current_dt.month, current_dt.day, 
                           current_dt.hour + current_dt.minute/60 + current_dt.second/3600 - 5.5)
            pos = swe.calc_ut(jd, swe.MOON)
            moon_long = pos[0][0] % 360
            moon_speed = pos[0][3] / (24 * 3600)
            
            current_row = nakshatra_df[(nakshatra_df['Start_Degree'] <= moon_long) & 
                                     (nakshatra_df['End_Degree'] > moon_long)]
            
            if current_row.empty:
                return current_dt + timedelta(minutes=45)
                
            end_degree = current_row.iloc[0]['End_Degree']
            
            if end_degree < moon_long:
                degrees_to_travel = (360 - moon_long) + end_degree
            else:
                degrees_to_travel = end_degree - moon_long
            
            degrees_to_travel += 0.001
            
            if moon_speed <= 0:
                return current_dt + timedelta(minutes=45)
                
            seconds_to_change = degrees_to_travel / moon_speed
            next_change = current_dt + timedelta(seconds=seconds_to_change)
            
            min_time = current_dt + timedelta(minutes=1)
            max_time = current_dt + timedelta(minutes=120)
            
            return max(min_time, min(next_change, max_time))
            
        except Exception:
            return current_dt + timedelta(minutes=45)
            
    def get_verdict(self, score: float) -> str:
        """Convert score to verdict"""
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


def test_hybrid_model():
    """Test the hybrid model"""
    model = KPHybridModel()
    
    # Load nakshatra data
    nakshatra_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'nakshatra_sub_lords_longitudes.csv')
    nakshatra_df = pd.read_csv(nakshatra_path)
    
    # Test with Panthers vs Nellai match
    test_dt = datetime(2025, 6, 18, 19, 51, 0)
    lat, lon = 11.6469616, 78.2106958
    
    # Generate 1-hour timeline
    df = model.predict_timeline_hybrid(test_dt, lat, lon, duration_hours=1, nakshatra_df=nakshatra_df)
    
    print("=== Hybrid Model Test ===")
    print(f"Generated {len(df)} timeline entries")
    print("\nFirst few entries:")
    print(df[['datetime', 'moon_star_lord', 'sub_lord', 'sub_sub_lord', 
              'final_score', 'verdict']].head().to_string(index=False))
    
    print(f"\nContradictions detected: {df['contradictions'].iloc[0] if not df.empty else 'None'}")
    print(f"Dynamic weights: SL={df['star_lord_weight'].iloc[0]:.1%}, SUB={df['sub_lord_weight'].iloc[0]:.1%}, SSL={df['sub_sub_lord_weight'].iloc[0]:.1%}")


if __name__ == "__main__":
    test_hybrid_model() 