#!/usr/bin/env python3
"""
Unified KP Cricket Predictor
==========================

A modular architecture that separates common functionality from model-specific logic.
Supports multiple prediction models through a unified interface.

Author: KP System
Version: 3.0
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Protocol
from abc import ABC, abstractmethod

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability, calculate_planet_strength, get_ruling_planets


class ScoringModel(Protocol):
    """Protocol defining the interface for all prediction models"""
    
    def calculate_scores(self, star_lord: str, sub_lord: str, sub_sub_lord: str, 
                        muhurta_chart: Dict, current_chart: Dict, nakshatra_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate scores for the given lords and charts"""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about this model"""
        ...


class FileManager:
    """Unified file management for all models"""
    
    @staticmethod
    def save_predictions(df: pd.DataFrame, team_a: str, team_b: str, 
                        match_date: str, start_time: str, model_name: str) -> str:
        """Save prediction results with consistent naming across all models"""
        try:
            # Create results directory
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Create consistent filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_str = match_date.replace('-', '')
            time_str = start_time.replace(':', '')
            filename = f"{team_a}_vs_{team_b}_{date_str}_{time_str}_{timestamp}.csv"
            filepath = os.path.join(results_dir, filename)
            
            # Ensure valid data
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            # Create metadata header
            metadata = {
                'Match': f"{team_a} vs {team_b}",
                'Date': match_date,
                'Start_Time': start_time,
                'Model_Type': model_name,
                'Total_Predictions': len(df),
                'Generated_At': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Write file with metadata
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("# KP Cricket Prediction Results\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("#\n")
            
            # Append DataFrame
            df.to_csv(filepath, mode='a', index=False, encoding='utf-8')
            
            print(f"✅ Results saved: {filename}")
            print(f"📊 Predictions: {len(df)}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            # Fallback
            fallback_path = f"results_{team_a}_vs_{team_b}_{timestamp}.csv"
            df.to_csv(fallback_path, index=False)
            print(f"⚠️ Saved to fallback: {fallback_path}")
            return fallback_path

    @staticmethod
    def load_predictions(filepath: str) -> pd.DataFrame:
        """Load predictions from file"""
        return pd.read_csv(filepath, comment='#')


class TimingEngine:
    """Unified timing calculations for authentic KP SSL changes"""
    
    @staticmethod
    def find_next_ssl_change(current_dt: datetime, lat: float, lon: float, 
                           nakshatra_df: pd.DataFrame) -> datetime:
        """Calculate next Sub-Sub Lord change time"""
        try:
            import swisseph as swe
            
            # Get current moon position
            jd = swe.julday(current_dt.year, current_dt.month, current_dt.day, 
                           current_dt.hour + current_dt.minute/60 + current_dt.second/3600 - 5.5)
            pos = swe.calc_ut(jd, swe.MOON)
            moon_long = pos[0][0] % 360
            moon_speed = pos[0][3] / (24 * 3600)  # degrees per second
            
            # Find current SSL's end degree
            current_row = nakshatra_df[
                (nakshatra_df['Start_Degree'] <= moon_long) & 
                (nakshatra_df['End_Degree'] > moon_long)
            ]
            
            if current_row.empty:
                return current_dt + timedelta(minutes=45)  # Fallback
                
            end_degree = current_row.iloc[0]['End_Degree']
            
            # Calculate time to reach end degree
            if end_degree < moon_long:
                degrees_to_travel = (360 - moon_long) + end_degree
            else:
                degrees_to_travel = end_degree - moon_long
                
            degrees_to_travel += 0.001  # Small buffer
            
            if moon_speed <= 0:
                return current_dt + timedelta(minutes=45)
                
            seconds_to_change = degrees_to_travel / moon_speed
            next_change = current_dt + timedelta(seconds=seconds_to_change)
            
            # Apply bounds
            min_time = current_dt + timedelta(minutes=1)
            max_time = current_dt + timedelta(minutes=120)
            
            return max(min_time, min(next_change, max_time))
            
        except Exception:
            return current_dt + timedelta(minutes=45)


class LegacyModel:
    """Legacy KP Model implementation"""
    
    def calculate_scores(self, star_lord: str, sub_lord: str, sub_sub_lord: str,
                        muhurta_chart: Dict, current_chart: Dict, nakshatra_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate scores using legacy methodology"""
        
        # Use legacy favorability evaluation
        favorability_data = evaluate_favorability(
            muhurta_chart, current_chart, nakshatra_df, use_enhanced_rules=False
        )
        
        return {
            'moon_sl_score': favorability_data.get("moon_sl_score", 0),
            'moon_sub_score': favorability_data.get("moon_sub_score", 0),
            'moon_ssl_score': favorability_data.get("moon_ssl_score", 0),
            'final_score': favorability_data.get("final_score", 0),
            'contradictions': [],
            'weights': {'star_lord': 0.5, 'sub_lord': 0.3, 'sub_sub_lord': 0.2},
            'verdict': self._get_verdict(favorability_data.get("final_score", 0))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': 'Legacy KP Model',
            'version': '1.0',
            'features': ['Fixed weights', 'Basic contradictions', 'KP calculations'],
            'description': 'Original KP system with fixed 50-30-20 weighting'
        }
    
    def _get_verdict(self, score: float) -> str:
        if score > 25: return "Strongly Favors Ascendant"
        if score > 10: return "Clearly Favors Ascendant"
        if score > 2: return "Slightly Favors Ascendant"
        if score < -25: return "Strongly Favors Descendant"
        if score < -10: return "Clearly Favors Descendant"
        if score < -2: return "Slightly Favors Descendant"
        return "Neutral / Too Close to Call"


class ComprehensiveModel:
    """Comprehensive KP Model with enhanced features"""
    
    def __init__(self):
        from scripts.kp_comprehensive_model import KPComprehensiveModel
        self.model = KPComprehensiveModel()
    
    def calculate_scores(self, star_lord: str, sub_lord: str, sub_sub_lord: str,
                        muhurta_chart: Dict, current_chart: Dict, nakshatra_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate scores using comprehensive methodology with fixed planetary scores"""
        
        # Use SAME legacy calculation for individual scores (ensures consistency)
        legacy_data = evaluate_favorability(
            muhurta_chart, current_chart, nakshatra_df, use_enhanced_rules=False
        )
        
        # Extract individual scores (same as legacy)
        sl_score = legacy_data.get("moon_sl_score", 0)
        sub_score = legacy_data.get("moon_sub_score", 0)
        ssl_score = legacy_data.get("moon_ssl_score", 0)
        
        # Store original scores for consistent display
        original_scores = {
            'star_lord_score': sl_score,
            'sub_lord_score': sub_score,
            'sub_sub_lord_score': ssl_score
        }
        
        try:
            # Apply comprehensive enhancements
            contradictions = self.model.detect_contradictions(star_lord, sub_lord, sub_sub_lord)
            
            # Level 2 Contradictions: Apply negative weights to BOTH planets for these malefic combinations
            level2_contradictions = ['Mars-Rahu', 'Rahu-Mars', 'Mars-Saturn', 'Saturn-Mars', 
                                   'Sun-Saturn', 'Saturn-Sun', 'Jupiter-Rahu', 'Rahu-Jupiter']
            
            # Check if any Level 2 contradiction exists
            level2_active = any(any(l2 in c for l2 in level2_contradictions) for c in contradictions)
            
            # Calculate dynamic weights (not scores)
            weights = self.model.calculate_dynamic_weights(original_scores, contradictions)
            
            # Apply Level 2 contradictions: BOTH conflicting planets get negative weights
            if level2_active:
                # Deduplicate contradictions to avoid double application
                processed_pairs = set()
                for contradiction in contradictions:
                    if any(l2 in contradiction for l2 in level2_contradictions):
                        # Extract the two planets from contradiction (e.g., "Mars-Rahu" -> ["Mars", "Rahu"])
                        planets = contradiction.split('-')
                        if len(planets) == 2:
                            planet1, planet2 = planets
                            
                            # Create a normalized pair (alphabetically sorted to avoid duplicates)
                            pair = tuple(sorted([planet1, planet2]))
                            if pair in processed_pairs:
                                continue  # Skip duplicate
                            processed_pairs.add(pair)
                            
                            # Apply negative weight to planet1 wherever it appears
                            if star_lord == planet1:
                                weights['star_lord'] *= -1
                            if sub_lord == planet1:
                                weights['sub_lord'] *= -1
                            if sub_sub_lord == planet1:
                                weights['sub_sub_lord'] *= -1
                                
                            # Apply negative weight to planet2 wherever it appears
                            if star_lord == planet2:
                                weights['star_lord'] *= -1
                            if sub_lord == planet2:
                                weights['sub_lord'] *= -1
                            if sub_sub_lord == planet2:
                                weights['sub_sub_lord'] *= -1
            
            # Calculate final score using ORIGINAL scores but modified weights
            final_score = (
                original_scores['star_lord_score'] * weights['star_lord'] +
                original_scores['sub_lord_score'] * weights['sub_lord'] + 
                original_scores['sub_sub_lord_score'] * weights['sub_sub_lord']
            )
            
            return {
                'moon_sl_score': original_scores['star_lord_score'],  # ALWAYS original
                'moon_sub_score': original_scores['sub_lord_score'],  # ALWAYS original
                'moon_ssl_score': original_scores['sub_sub_lord_score'],  # ALWAYS original
                'final_score': final_score,
                'contradictions': contradictions,
                'weights': weights,
                'verdict': self._get_verdict(final_score)
            }
            
        except Exception as e:
            # Fallback to basic scores with legacy weights
            final_score = (sl_score * 0.5) + (sub_score * 0.3) + (ssl_score * 0.2)
            
            return {
                'moon_sl_score': sl_score,
                'moon_sub_score': sub_score,
                'moon_ssl_score': ssl_score,
                'final_score': final_score,
                'contradictions': [],
                'weights': {'star_lord': 0.5, 'sub_lord': 0.3, 'sub_sub_lord': 0.2},
                'verdict': self._get_verdict(final_score)
            }
    
    def _calculate_fixed_scores(self, muhurta_chart: Dict, nakshatra_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate fixed planetary scores once from muhurta chart (DEPRECATED - now using legacy method)"""
        ruling_planets = get_ruling_planets(muhurta_chart, nakshatra_df)
        fixed_scores = {}
        
        for planet in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']:
            if planet in muhurta_chart.get("planets", {}):
                asc_score, desc_score, _ = calculate_planet_strength(
                    planet, muhurta_chart, ruling_planets, nakshatra_df, None
                )
                fixed_scores[planet] = asc_score - desc_score
        
        return fixed_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': 'Comprehensive KP Model',
            'version': '2.0',
            'features': ['Dynamic weights', 'Advanced contradictions', 'Fixed planetary scores'],
            'description': 'Enhanced KP system with contradiction corrections and dynamic weighting'
        }
    
    def _get_verdict(self, score: float) -> str:
        if score > 25: return "Strongly Favors Ascendant"
        if score > 10: return "Clearly Favors Ascendant"
        if score > 2: return "Slightly Favors Ascendant"
        if score < -25: return "Strongly Favors Descendant"
        if score < -10: return "Clearly Favors Descendant"
        if score < -2: return "Slightly Favors Descendant"
        return "Neutral / Too Close to Call"


class UnifiedKPPredictor:
    """Unified predictor that works with any scoring model"""
    
    def __init__(self, model_type: str = "comprehensive"):
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.file_manager = FileManager()
        self.timing_engine = TimingEngine()
        
        # Team information storage
        self.team_a = None
        self.team_b = None
        self.match_date = None
        self.start_time = None
    
    def _get_model(self, model_type: str) -> ScoringModel:
        """Factory method to get the appropriate model"""
        if model_type.lower() == "legacy":
            return LegacyModel()
        elif model_type.lower() == "comprehensive":
            return ComprehensiveModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_match_timeline(self, team_a: str, team_b: str, match_date: str,
                             start_time: str, lat: float, lon: float,
                             duration_hours: int = 5) -> pd.DataFrame:
        """Generate match timeline predictions using the selected model"""
        
        # Store team information
        self.team_a = team_a
        self.team_b = team_b
        self.match_date = match_date
        self.start_time = start_time
        
        # Parse datetime
        start_dt = datetime.strptime(f"{match_date} {start_time}", "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + timedelta(hours=duration_hours)
        
        # Load nakshatra data
        nakshatra_data_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'nakshatra_sub_lords_longitudes.csv')
        nakshatra_df = pd.read_csv(nakshatra_data_path)
        
        # Generate muhurta chart (foundation)
        muhurta_chart = generate_kp_chart(start_dt, lat, lon, nakshatra_df)
        if "error" in muhurta_chart:
            raise ValueError(f"Error generating muhurta chart: {muhurta_chart['error']}")
        
        predictions = []
        current_dt = start_dt
        
        # Track duplicates
        last_entry = {"lords": None, "score": None, "datetime": None}
        
        while current_dt <= end_dt:
            # Generate current chart
            current_chart = generate_kp_chart(current_dt, lat, lon, nakshatra_df)
            if "error" in current_chart:
                current_dt += timedelta(minutes=1)
                continue
            
            # Extract current lords
            star_lord = current_chart.get("moon_star_lord")
            sub_lord = current_chart.get("moon_sub_lord")
            sub_sub_lord = current_chart.get("moon_sub_sub_lord")
            
            # Calculate scores using the selected model
            try:
                score_data = self.model.calculate_scores(
                    star_lord, sub_lord, sub_sub_lord,
                    muhurta_chart, current_chart, nakshatra_df
                )
                
                # Check for duplicates
                current_lords = (star_lord, sub_lord, sub_sub_lord)
                current_score = score_data['final_score']
                
                is_duplicate = (
                    last_entry["lords"] == current_lords and
                    abs((current_score or 0) - (last_entry["score"] or 0)) < 0.001
                )
                
                # Check minimum time gap
                time_gap_ok = True
                if last_entry["datetime"]:
                    time_diff = (current_dt - last_entry["datetime"]).total_seconds()
                    time_gap_ok = time_diff >= 30
                
                if is_duplicate or not time_gap_ok:
                    current_dt = self.timing_engine.find_next_ssl_change(current_dt, lat, lon, nakshatra_df)
                    continue
                
                # Create prediction entry
                prediction = {
                    'datetime': current_dt,
                    'moon_star_lord': star_lord,
                    'sub_lord': sub_lord,
                    'sub_sub_lord': sub_sub_lord,
                    'moon_sl_score': score_data['moon_sl_score'],
                    'moon_sub_score': score_data['moon_sub_score'],
                    'moon_ssl_score': score_data['moon_ssl_score'],
                    'contradictions': ','.join(score_data.get('contradictions', [])),
                    'star_lord_weight': score_data['weights']['star_lord'],
                    'sub_lord_weight': score_data['weights']['sub_lord'],
                    'sub_sub_lord_weight': score_data['weights']['sub_sub_lord'],
                    'final_score': score_data['final_score'],
                    'verdict': score_data['verdict']
                }
                
                predictions.append(prediction)
                
                # Update last entry
                last_entry = {
                    "lords": current_lords,
                    "score": current_score,
                    "datetime": current_dt
                }
                
                # Move to next SSL change
                current_dt = self.timing_engine.find_next_ssl_change(current_dt, lat, lon, nakshatra_df)
                
            except Exception as e:
                print(f"Error processing {current_dt}: {e}")
                current_dt += timedelta(minutes=1)
                continue
        
        return pd.DataFrame(predictions)
    
    def save_results(self, df: pd.DataFrame) -> str:
        """Save results using stored team information"""
        if not all([self.team_a, self.team_b, self.match_date, self.start_time]):
            raise ValueError("Team information not set. Call predict_match_timeline first.")
        
        model_info = self.model.get_model_info()
        return self.file_manager.save_predictions(
            df, self.team_a, self.team_b, self.match_date, self.start_time, model_info['name']
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.model.get_model_info()


def test_unified_predictor():
    """Test the unified predictor with different models"""
    print("=== Testing Unified KP Predictor ===")
    
    # Test parameters
    team_a = "Panthers"
    team_b = "Nellai"
    match_date = "2025-06-18"
    start_time = "19:51:00"
    lat = 11.6469616
    lon = 78.2106958
    
    # Test Legacy Model
    print("\n🔧 Testing Legacy Model:")
    legacy_predictor = UnifiedKPPredictor("legacy")
    legacy_df = legacy_predictor.predict_match_timeline(
        team_a, team_b, match_date, start_time, lat, lon, duration_hours=1
    )
    legacy_file = legacy_predictor.save_results(legacy_df)
    print(f"Legacy: {len(legacy_df)} predictions → {os.path.basename(legacy_file)}")
    
    # Test Comprehensive Model
    print("\n🚀 Testing Comprehensive Model:")
    comp_predictor = UnifiedKPPredictor("comprehensive")
    comp_df = comp_predictor.predict_match_timeline(
        team_a, team_b, match_date, start_time, lat, lon, duration_hours=1
    )
    comp_file = comp_predictor.save_results(comp_df)
    print(f"Comprehensive: {len(comp_df)} predictions → {os.path.basename(comp_file)}")
    
    # Compare results
    print(f"\n📊 Comparison:")
    print(f"Legacy Model Info: {legacy_predictor.get_model_info()}")
    print(f"Comprehensive Model Info: {comp_predictor.get_model_info()}")


if __name__ == "__main__":
    test_unified_predictor() 