#!/usr/bin/env python3
"""
KP Timeline Predictor - FIXED VERSION
Addresses all variable naming and database schema issues identified in codebase analysis
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Fix import paths
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
ROOT_DIR = os.path.dirname(scripts_dir)
sys.path.insert(0, ROOT_DIR)

from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability

class KPTimelinePredictorFixed:
    def __init__(self, db_path="training_analysis/cricket_predictions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Load nakshatra data with error checking
        try:
            self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
            
            # Verify required columns exist
            required_columns = ['Start_Degree', 'End_Degree', 'Sub_Lord', 'Sub_Sub_Lord', 'Nakshatra', 'Pada']
            missing_columns = [col for col in required_columns if col not in self.nakshatra_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in nakshatra data: {missing_columns}")
                
            print(f"✓ Nakshatra data loaded successfully: {len(self.nakshatra_df)} rows")
            
        except Exception as e:
            print(f"✗ Error loading nakshatra data: {str(e)}")
            self.nakshatra_df = None
        
        # KP Constants
        self.benefic_planets = ["Jupiter", "Venus"]
        self.malefic_planets = ["Saturn", "Mars", "Sun", "Rahu", "Ketu"]
        self.neutral_planets = ["Moon", "Mercury"]
        
        print("KP Timeline Predictor (Fixed) initialized")
    
    def validate_database_schema(self):
        """Validate that all required database tables and columns exist"""
        
        validation_results = {
            'tables_exist': True,
            'columns_exist': True,
            'missing_tables': [],
            'missing_columns': {},
            'recommendations': []
        }
        
        # Check required tables
        required_tables = ['matches', 'deliveries', 'astrological_predictions', 'chart_data']
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        for table in required_tables:
            if table not in existing_tables:
                validation_results['tables_exist'] = False
                validation_results['missing_tables'].append(table)
        
        # Check required columns for each table
        table_column_requirements = {
            'matches': ['match_id', 'start_datetime', 'team1', 'team2', 'venue'],
            'deliveries': ['id', 'match_id', 'inning', 'over', 'ball', 'timestamp'],
            'astrological_predictions': ['id', 'delivery_id', 'match_id', 'asc_score', 'desc_score'],
            'chart_data': ['id', 'delivery_id', 'match_id', 'moon_longitude', 'moon_nakshatra']
        }
        
        for table, required_cols in table_column_requirements.items():
            if table in existing_tables:
                cursor.execute(f"PRAGMA table_info({table})")
                existing_cols = [row[1] for row in cursor.fetchall()]
                
                missing_cols = [col for col in required_cols if col not in existing_cols]
                if missing_cols:
                    validation_results['columns_exist'] = False
                    validation_results['missing_columns'][table] = missing_cols
        
        # Generate recommendations
        if not validation_results['tables_exist']:
            validation_results['recommendations'].append("Run database initialization script")
        
        if not validation_results['columns_exist']:
            validation_results['recommendations'].append("Update database schema or modify code to use existing columns")
        
        return validation_results
    
    def generate_safe_chart(self, dt, lat=19.0760, lon=72.8777):
        """Generate KP chart with comprehensive error handling"""
        
        if self.nakshatra_df is None:
            return {"error": "Nakshatra data not loaded"}
        
        try:
            chart = generate_kp_chart(dt, lat, lon, self.nakshatra_df)
            
            # Verify chart structure
            if isinstance(chart, dict) and 'error' not in chart:
                required_keys = ['planets', 'moon_longitude', 'moon_nakshatra', 'moon_sub_lord']
                missing_keys = [key for key in required_keys if key not in chart]
                
                if missing_keys:
                    return {"error": f"Chart missing required keys: {missing_keys}"}
                
                return chart
            else:
                return chart  # Return error as-is
                
        except Exception as e:
            return {"error": f"Chart generation failed: {str(e)}"}
    
    def generate_timeline_with_probabilities(self, match_id, hours=5):
        """
        Generate timeline with probability scores - FIXED VERSION
        Addresses all database schema and variable naming issues
        """
        
        print(f"Generating timeline for match {match_id}...")
        
        # Validate database schema first
        validation = self.validate_database_schema()
        if not validation['tables_exist'] or not validation['columns_exist']:
            return {
                "error": "Database schema validation failed",
                "validation_results": validation
            }
        
        # Get match information using correct column names
        try:
            match_info = pd.read_sql_query(
                "SELECT * FROM matches WHERE match_id = ?", 
                self.conn, params=[match_id]
            )
            
            if match_info.empty:
                return {"error": f"Match {match_id} not found"}
            
            match_row = match_info.iloc[0]
            
        except Exception as e:
            return {"error": f"Failed to fetch match data: {str(e)}"}
        
        # Get delivery data using correct column names
        try:
            deliveries_df = pd.read_sql_query(
                "SELECT timestamp FROM deliveries WHERE match_id = ? AND timestamp IS NOT NULL ORDER BY timestamp",
                self.conn, params=[match_id]
            )
            
            if deliveries_df.empty:
                return {"error": f"No timestamped deliveries found for match {match_id}"}
            
            # Convert timestamps
            deliveries_df['timestamp'] = pd.to_datetime(deliveries_df['timestamp'])
            
        except Exception as e:
            return {"error": f"Failed to fetch delivery data: {str(e)}"}
        
        # Determine timeline boundaries
        match_start = deliveries_df['timestamp'].min() - timedelta(minutes=30)
        timeline_end = match_start + timedelta(hours=hours)
        
        # Generate astrological periods
        timeline_periods = self.generate_astrological_periods_safe(match_start, timeline_end)
        
        if not timeline_periods:
            return {"error": "Failed to generate astrological periods"}
        
        # Analyze each period with probability scores
        period_predictions = []
        
        for i, period in enumerate(timeline_periods):
            period_start = period['start_time']
            period_end = period['end_time']
            chart = period['chart']
            favorability = period['favorability']
            
            # Skip periods with chart errors
            if 'error' in chart:
                continue
            
            # Determine team favorability
            kp_score = favorability.get('final_score', 0)
            
            if kp_score > 0.1:
                favored_team = match_row['team1']  # Ascendant
                favorability_strength = "Strong Ascendant"
            elif kp_score < -0.1:
                favored_team = match_row['team2']  # Descendant
                favorability_strength = "Strong Descendant"
            else:
                favored_team = "Neutral"
                favorability_strength = "Balanced"
            
            # Calculate match dynamics with probabilities
            dynamics_result = self.analyze_period_dynamics_safe(chart, favorability)
            
            period_prediction = {
                'period_num': i + 1,
                'time_range': f"{period_start.strftime('%H:%M')} - {period_end.strftime('%H:%M')}",
                'duration_minutes': period['duration_minutes'],
                'current_sub_lord': chart.get('moon_sub_lord', 'Unknown'),
                'current_sub_sub_lord': chart.get('moon_sub_sub_lord', 'Unknown'),
                'moon_nakshatra': chart.get('moon_nakshatra', 'Unknown'),
                'moon_pada': chart.get('moon_pada', 0),
                'ascendant_degree': round(chart.get('ascendant_degree', 0), 2),
                'favored_team': favored_team,
                'favorability_strength': favorability_strength,
                'kp_score': round(kp_score, 3),
                'match_dynamics': dynamics_result['categories'],
                'dynamics_probabilities': dynamics_result['probabilities'],
                'astrological_notes': self.get_astrological_notes_safe(chart, favorability)
            }
            
            period_predictions.append(period_prediction)
        
        return {
            'match_id': match_id,
            'match_info': {
                'team1': match_row['team1'],
                'team2': match_row['team2'],
                'date': match_row['start_datetime'],  # Use correct column name
                'venue': match_row.get('venue', 'Unknown')
            },
            'timeline_start': match_start.strftime('%Y-%m-%d %H:%M:%S'),
            'timeline_duration_hours': hours,
            'total_periods': len(period_predictions),
            'period_predictions': period_predictions,
            'database_validation': validation
        }
    
    def generate_astrological_periods_safe(self, start_time, end_time, max_periods=20):
        """Generate astrological periods with comprehensive error handling"""
        
        periods = []
        current_time = start_time
        period_count = 0
        
        while current_time < end_time and period_count < max_periods:
            # Generate chart with error handling
            current_chart = self.generate_safe_chart(current_time.to_pydatetime())
            
            if 'error' in current_chart:
                # Fall back to 45-minute periods on chart errors
                next_time = current_time + timedelta(minutes=45)
                print(f"Chart generation failed for {current_time}, using fallback period")
            else:
                # Find next sub lord change
                next_time = self.find_next_sub_lord_change_safe(current_time, current_chart)
                
                # Ensure reasonable period bounds
                min_next_time = current_time + timedelta(minutes=15)
                max_next_time = current_time + timedelta(minutes=90)
                
                if next_time < min_next_time:
                    next_time = min_next_time
                elif next_time > max_next_time:
                    next_time = max_next_time
            
            # Don't exceed end time
            if next_time > end_time:
                next_time = end_time
            
            # Evaluate favorability with error handling
            if 'error' not in current_chart:
                try:
                    dummy_deliveries = pd.DataFrame({
                        'timestamp': [current_time],
                        'runs_off_bat': [0],
                        'wicket_type': [None]
                    })
                    favorability = evaluate_favorability(current_chart, dummy_deliveries)
                except Exception as e:
                    print(f"Favorability evaluation failed: {str(e)}")
                    favorability = {'final_score': 0}
            else:
                favorability = {'final_score': 0}
            
            periods.append({
                'start_time': current_time,
                'end_time': next_time,
                'chart': current_chart,
                'favorability': favorability,
                'duration_minutes': int((next_time - current_time).total_seconds() / 60)
            })
            
            current_time = next_time
            period_count += 1
        
        return periods
    
    def find_next_sub_lord_change_safe(self, current_time, current_chart):
        """Find next sub lord change with comprehensive error handling"""
        
        if self.nakshatra_df is None:
            return current_time + timedelta(minutes=45)
        
        try:
            import swisseph as swe
            
            # Get current moon position and speed
            jd = swe.julday(
                current_time.year, current_time.month, current_time.day, 
                current_time.hour + current_time.minute/60 + current_time.second/3600 - 5.5
            )
            
            pos = swe.calc_ut(jd, swe.MOON)
            moon_long = pos[0][0] % 360
            moon_speed_deg_per_day = pos[0][3]
            moon_speed_deg_per_minute = moon_speed_deg_per_day / (24 * 60)
            
            # Find current sub lord boundaries
            current_row = self.nakshatra_df[
                (self.nakshatra_df['Start_Degree'] <= moon_long) & 
                (self.nakshatra_df['End_Degree'] > moon_long)
            ]
            
            if current_row.empty:
                # Handle edge case for 360 degrees
                if moon_long >= 359.99:
                    current_row = self.nakshatra_df.iloc[-1:]
                else:
                    return current_time + timedelta(minutes=45)
            
            if current_row.empty:
                return current_time + timedelta(minutes=45)
            
            # Get the end degree of current sub lord
            end_degree = current_row.iloc[0]['End_Degree']
            
            # Calculate degrees to travel
            if end_degree < moon_long:  # Handle 360 -> 0 crossover
                degrees_to_travel = (360 - moon_long) + end_degree
            else:
                degrees_to_travel = end_degree - moon_long
            
            degrees_to_travel += 0.001  # Small buffer
            
            # Calculate time to reach next sub lord
            if moon_speed_deg_per_minute > 0:
                minutes_to_change = degrees_to_travel / moon_speed_deg_per_minute
                next_change_time = current_time + timedelta(minutes=minutes_to_change)
            else:
                next_change_time = current_time + timedelta(minutes=45)
            
            return next_change_time
            
        except Exception as e:
            print(f"Error calculating sub lord change: {str(e)}")
            return current_time + timedelta(minutes=45)
    
    def analyze_period_dynamics_safe(self, chart, favorability):
        """Analyze match dynamics with comprehensive error handling"""
        
        dynamics = []
        probabilities = {}
        
        try:
            # Get planetary positions safely
            planets = chart.get('planets', {})
            kp_score = favorability.get('final_score', 0)
            
            # Calculate probabilities with error handling
            probabilities['high_scoring_probability'] = round(
                self.calculate_high_scoring_probability_safe(chart, favorability), 3
            )
            probabilities['collapse_probability'] = round(
                self.calculate_collapse_probability_safe(chart, favorability), 3
            )
            probabilities['wicket_pressure_probability'] = round(
                self.calculate_wicket_pressure_probability_safe(chart, favorability), 3
            )
            probabilities['momentum_shift_probability'] = round(
                self.calculate_momentum_shift_probability_safe(chart, favorability), 3
            )
            
            # Convert probabilities to categories
            if probabilities['high_scoring_probability'] > 0.6:
                dynamics.append("High Scoring")
            elif probabilities['high_scoring_probability'] > 0.4:
                dynamics.append("Moderate Scoring")
                
            if probabilities['collapse_probability'] > 0.6:
                dynamics.append("Collapse Risk")
            elif probabilities['collapse_probability'] > 0.4:
                dynamics.append("Moderate Risk")
                
            if probabilities['wicket_pressure_probability'] > 0.5:
                dynamics.append("Wicket Pressure")
                
            if probabilities['momentum_shift_probability'] > 0.5:
                dynamics.append("Momentum Shift")
            
            # Default to normal if no special dynamics
            if not dynamics:
                dynamics.append("Normal")
            
        except Exception as e:
            print(f"Error analyzing period dynamics: {str(e)}")
            dynamics = ["Normal"]
            probabilities = {
                'high_scoring_probability': 0.5,
                'collapse_probability': 0.3,
                'wicket_pressure_probability': 0.4,
                'momentum_shift_probability': 0.3
            }
        
        return {
            'categories': dynamics,
            'probabilities': probabilities
        }
    
    def calculate_high_scoring_probability_safe(self, chart, favorability):
        """Calculate high scoring probability with error handling"""
        
        try:
            probability = 0.0
            planets = chart.get('planets', {})
            kp_score = favorability.get('final_score', 0)
            
            # Base probability from KP favorability
            if abs(kp_score) > 0.2:
                probability += 0.3
            elif abs(kp_score) > 0.1:
                probability += 0.15
            
            # Benefic planet influence
            jupiter_long = planets.get('Jupiter', {}).get('longitude', 0)
            venus_long = planets.get('Venus', {}).get('longitude', 0)
            
            benefic_count = 0
            if jupiter_long > 0:
                benefic_count += 1
            if venus_long > 0:
                benefic_count += 1
                
            probability += benefic_count * 0.2
            
            # Moon nakshatra influence
            nakshatra = chart.get('moon_nakshatra', '')
            benefic_nakshatras = ['Rohini', 'Pushya', 'Magha', 'Uttara Phalguni', 'Hasta']
            if nakshatra in benefic_nakshatras:
                probability += 0.1
            
            return min(probability, 1.0)
            
        except Exception as e:
            print(f"Error calculating high scoring probability: {str(e)}")
            return 0.5
    
    def calculate_collapse_probability_safe(self, chart, favorability):
        """Calculate collapse probability with error handling"""
        
        try:
            probability = 0.0
            planets = chart.get('planets', {})
            kp_score = favorability.get('final_score', 0)
            
            # Negative favorability increases collapse risk
            if kp_score < -0.2:
                probability += 0.3
            elif kp_score < -0.1:
                probability += 0.15
            
            # Malefic planet influence
            malefic_count = 0
            for planet in ['Saturn', 'Mars', 'Rahu']:
                if planets.get(planet, {}).get('longitude', 0) > 0:
                    malefic_count += 1
                    
            probability += malefic_count * 0.15
            
            # Retrograde malefics increase risk
            if planets.get('Saturn', {}).get('retrograde', False):
                probability += 0.2
            if planets.get('Mars', {}).get('retrograde', False):
                probability += 0.15
            
            return min(probability, 1.0)
            
        except Exception as e:
            print(f"Error calculating collapse probability: {str(e)}")
            return 0.3
    
    def calculate_wicket_pressure_probability_safe(self, chart, favorability):
        """Calculate wicket pressure probability with error handling"""
        
        try:
            probability = 0.0
            planets = chart.get('planets', {})
            
            # Saturn influence
            if planets.get('Saturn', {}).get('retrograde', False):
                probability += 0.3
            elif planets.get('Saturn', {}).get('longitude', 0) > 0:
                probability += 0.15
            
            # Mars influence
            if planets.get('Mars', {}).get('retrograde', False):
                probability += 0.25
            elif planets.get('Mars', {}).get('longitude', 0) > 0:
                probability += 0.1
            
            # Moderate KP scores indicate tight contests
            kp_score = favorability.get('final_score', 0)
            if 0.05 <= abs(kp_score) <= 0.15:
                probability += 0.15
            
            return min(probability, 1.0)
            
        except Exception as e:
            print(f"Error calculating wicket pressure probability: {str(e)}")
            return 0.4
    
    def calculate_momentum_shift_probability_safe(self, chart, favorability):
        """Calculate momentum shift probability with error handling"""
        
        try:
            probability = 0.0
            kp_score = favorability.get('final_score', 0)
            planets = chart.get('planets', {})
            
            # Neutral KP zones indicate transitions
            if -0.1 <= kp_score <= 0.1:
                probability += 0.4
            
            # Sub lord changes indicate natural transition points
            probability += 0.3
            
            # Mercury influence (changeability)
            if planets.get('Mercury', {}).get('longitude', 0) > 0:
                probability += 0.15
            
            # Fast-moving moon
            moon_speed = planets.get('Moon', {}).get('speed', 0)
            if abs(moon_speed) > 13:
                probability += 0.1
            
            return min(probability, 1.0)
            
        except Exception as e:
            print(f"Error calculating momentum shift probability: {str(e)}")
            return 0.3
    
    def get_astrological_notes_safe(self, chart, favorability):
        """Generate astrological notes with error handling"""
        
        try:
            notes = []
            
            # Moon nakshatra significance
            nakshatra = chart.get('moon_nakshatra', 'Unknown')
            sub_lord = chart.get('moon_sub_lord', 'Unknown')
            notes.append(f"Moon in {nakshatra} nakshatra, sub lord: {sub_lord}")
            
            # Planetary influences
            planets = chart.get('planets', {})
            retrograde_planets = []
            for planet, data in planets.items():
                if data.get('retrograde', False):
                    retrograde_planets.append(planet)
            
            if retrograde_planets:
                notes.append(f"Retrograde planets: {', '.join(retrograde_planets)}")
            
            # KP score interpretation
            kp_score = favorability.get('final_score', 0)
            if kp_score > 0.2:
                notes.append("Strong ascendant favorability - positive momentum")
            elif kp_score < -0.2:
                notes.append("Strong descendant favorability - challenging period")
            else:
                notes.append("Balanced planetary influences")
            
            return notes
            
        except Exception as e:
            print(f"Error generating astrological notes: {str(e)}")
            return ["Astrological analysis unavailable"]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    """Test the fixed timeline predictor"""
    
    predictor = KPTimelinePredictorFixed()
    
    # Test database validation
    print("\n=== Database Schema Validation ===")
    validation = predictor.validate_database_schema()
    
    if validation['tables_exist'] and validation['columns_exist']:
        print("✓ Database schema validation passed")
        
        # Test with a sample match
        test_match_id = "1011497"
        print(f"\n=== Testing Timeline Generation for Match {test_match_id} ===")
        
        result = predictor.generate_timeline_with_probabilities(test_match_id, hours=3)
        
        if 'error' not in result:
            print(f"✓ Timeline generated successfully")
            print(f"  - Total periods: {result['total_periods']}")
            print(f"  - Timeline start: {result['timeline_start']}")
            
            # Show first period details
            if result['period_predictions']:
                first_period = result['period_predictions'][0]
                print(f"\n=== First Period Details ===")
                print(f"Time Range: {first_period['time_range']}")
                print(f"Favored Team: {first_period['favored_team']}")
                print(f"KP Score: {first_period['kp_score']}")
                print(f"Match Dynamics: {first_period['match_dynamics']}")
                print(f"Probability Scores:")
                for prob_type, score in first_period['dynamics_probabilities'].items():
                    print(f"  - {prob_type}: {score:.1%}")
        else:
            print(f"✗ Timeline generation failed: {result['error']}")
    else:
        print("✗ Database schema validation failed")
        print(f"Missing tables: {validation['missing_tables']}")
        print(f"Missing columns: {validation['missing_columns']}")
        print(f"Recommendations: {validation['recommendations']}")
    
    predictor.close()

if __name__ == "__main__":
    main() 