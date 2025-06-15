#!/usr/bin/env python3
"""
KP Ascendant Validator & Timeline Predictor
Validates correct ascendant/descendant assignment and provides timeline predictions
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability

class KPAscendantValidator:
    def __init__(self, db_path="training_analysis/cricket_predictions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Load nakshatra data
        self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
        
        # KP Constants
        self.benefic_planets = ["Jupiter", "Venus"]
        self.malefic_planets = ["Saturn", "Mars", "Sun", "Rahu", "Ketu"]
        self.neutral_planets = ["Moon", "Mercury"]
        
        print("KP Ascendant Validator & Timeline Predictor initialized")
    
    def validate_ascendant_assignment(self, match_id):
        """Validate if Team1 is truly ascendant or if assignment should be flipped"""
        
        print(f"Validating ascendant assignment for match {match_id}...")
        
        # Get match data
        match_query = """
        SELECT 
            d.match_id,
            d.team1,
            d.team2,
            d.winner,
            mc.muhurta_strength_score,
            mc.ascendant_favorability,
            mc.descendant_favorability,
            mc.match_start_time,
            
            -- Team performance
            SUM(CASE WHEN d.batting_team = d.team1 THEN d.runs_off_bat ELSE 0 END) as team1_runs,
            SUM(CASE WHEN d.batting_team = d.team2 THEN d.runs_off_bat ELSE 0 END) as team2_runs,
            SUM(CASE WHEN d.batting_team = d.team1 AND d.wicket_type IS NOT NULL THEN 1 ELSE 0 END) as team1_wickets,
            SUM(CASE WHEN d.batting_team = d.team2 AND d.wicket_type IS NOT NULL THEN 1 ELSE 0 END) as team2_wickets
            
        FROM deliveries d
        LEFT JOIN muhurta_charts mc ON d.match_id = mc.match_id
        WHERE d.match_id = ?
        GROUP BY d.match_id
        """
        
        match_data = pd.read_sql_query(match_query, self.conn, params=[match_id])
        
        if match_data.empty:
            return None
        
        match = match_data.iloc[0]
        
        # Analyze period-wise performance
        periods_analysis = self.analyze_match_periods(match_id)
        
        # Calculate validation scores
        validation_result = self.calculate_validation_scores(match, periods_analysis)
        
        return validation_result
    
    def analyze_match_periods(self, match_id):
        """Analyze match performance in different astrological periods"""
        
        # Get delivery-level data with timestamps
        deliveries_query = """
        SELECT 
            d.*,
            mc.match_start_time,
            mc.moon_star_lord,
            mc.moon_sub_lord,
            mc.moon_sub_sub_lord
        FROM deliveries d
        LEFT JOIN muhurta_charts mc ON d.match_id = mc.match_id
        WHERE d.match_id = ? AND d.timestamp IS NOT NULL
        ORDER BY d.timestamp
        """
        
        deliveries_df = pd.read_sql_query(deliveries_query, self.conn, params=[match_id])
        
        if deliveries_df.empty:
            return []
        
        # Convert timestamps
        deliveries_df['timestamp'] = pd.to_datetime(deliveries_df['timestamp'])
        
        # Use first delivery time if no match start time
        if pd.isna(deliveries_df['match_start_time'].iloc[0]):
            match_start = deliveries_df['timestamp'].min() - timedelta(minutes=30)
        else:
            match_start = pd.to_datetime(deliveries_df['match_start_time'].iloc[0])
        
        # Create 30-minute periods for analysis
        periods = []
        
        for period_num in range(12):  # 6 hours = 12 periods of 30 minutes
            period_start = match_start + timedelta(minutes=period_num * 30)
            period_end = period_start + timedelta(minutes=30)
            
            # Get deliveries in this period
            period_deliveries = deliveries_df[
                (deliveries_df['timestamp'] >= period_start) & 
                (deliveries_df['timestamp'] < period_end)
            ]
            
            if len(period_deliveries) == 0:
                continue
            
            # Generate chart for period middle time
            period_middle = period_start + timedelta(minutes=15)
            period_chart = generate_kp_chart(
                period_middle.to_pydatetime(), 
                19.0760, 72.8777, 
                self.nakshatra_df
            )
            
            if 'error' in period_chart:
                continue
            
            # Evaluate favorability for this period
            favorability = evaluate_favorability(period_chart, period_deliveries)
            
            # Calculate period performance
            period_analysis = self.calculate_period_performance(period_deliveries, favorability)
            period_analysis.update({
                'period_num': period_num + 1,
                'period_start': period_start.strftime('%H:%M'),
                'period_end': period_end.strftime('%H:%M'),
                'deliveries_count': len(period_deliveries),
                'current_sub_lord': period_chart.get('moon_sub_lord', 'Unknown'),
                'current_sub_sub_lord': period_chart.get('moon_sub_sub_lord', 'Unknown'),
                'kp_favorability': favorability.get('final_score', 0)
            })
            
            periods.append(period_analysis)
        
        return periods
    
    def calculate_period_performance(self, period_deliveries, favorability):
        """Calculate team performance metrics for a period"""
        
        if period_deliveries.empty:
            return {}
        
        team1 = period_deliveries['team1'].iloc[0]
        team2 = period_deliveries['team2'].iloc[0]
        
        # Team1 performance
        team1_deliveries = period_deliveries[period_deliveries['batting_team'] == team1]
        team1_runs = team1_deliveries['runs_off_bat'].sum()
        team1_wickets = len(team1_deliveries[team1_deliveries['wicket_type'].notna()])
        team1_run_rate = team1_runs / len(team1_deliveries) if len(team1_deliveries) > 0 else 0
        
        # Team2 performance
        team2_deliveries = period_deliveries[period_deliveries['batting_team'] == team2]
        team2_runs = team2_deliveries['runs_off_bat'].sum()
        team2_wickets = len(team2_deliveries[team2_deliveries['wicket_type'].notna()])
        team2_run_rate = team2_runs / len(team2_deliveries) if len(team2_deliveries) > 0 else 0
        
        # Performance indicators
        performance_diff = team1_run_rate - team2_run_rate
        wicket_pressure = (team1_wickets + team2_wickets) / len(period_deliveries) if len(period_deliveries) > 0 else 0
        
        # Scoring intensity
        total_runs = team1_runs + team2_runs
        total_deliveries = len(period_deliveries)
        scoring_intensity = total_runs / total_deliveries if total_deliveries > 0 else 0
        
        return {
            'team1_runs': team1_runs,
            'team2_runs': team2_runs,
            'team1_wickets': team1_wickets,
            'team2_wickets': team2_wickets,
            'team1_run_rate': team1_run_rate,
            'team2_run_rate': team2_run_rate,
            'performance_diff': performance_diff,
            'wicket_pressure': wicket_pressure,
            'scoring_intensity': scoring_intensity,
            'high_scoring_period': scoring_intensity > 1.5,
            'collapse_risk_period': wicket_pressure > 0.15,
            'team1_dominant': performance_diff > 0.5,
            'team2_dominant': performance_diff < -0.5
        }
    
    def calculate_validation_scores(self, match_data, periods_analysis):
        """Calculate validation scores to determine correct ascendant assignment"""
        
        if not periods_analysis:
            return {'assignment_correct': True, 'confidence': 0.5, 'reason': 'Insufficient data'}
        
        # Count periods where KP favorability matches actual performance
        correct_predictions = 0
        total_predictions = 0
        
        ascendant_favored_periods = 0
        descendant_favored_periods = 0
        
        for period in periods_analysis:
            kp_score = period.get('kp_favorability', 0)
            performance_diff = period.get('performance_diff', 0)
            
            if abs(kp_score) > 0.1 and abs(performance_diff) > 0.1:  # Significant differences
                total_predictions += 1
                
                # Check if KP prediction matches actual performance
                kp_favors_ascendant = kp_score > 0
                actual_favors_team1 = performance_diff > 0
                
                if kp_favors_ascendant == actual_favors_team1:
                    correct_predictions += 1
                
                if kp_favors_ascendant:
                    ascendant_favored_periods += 1
                else:
                    descendant_favored_periods += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.5
        
        # Determine if assignment should be flipped
        assignment_correct = accuracy >= 0.6  # 60% threshold
        
        # Additional validation using match outcome
        winner = match_data.get('winner', 'Unknown')
        team1 = match_data.get('team1', 'Unknown')
        muhurta_strength = match_data.get('muhurta_strength_score', 0) or 0
        
        outcome_validation = True
        if winner == team1 and muhurta_strength < -1:  # Team1 won but muhurta strongly favors descendant
            outcome_validation = False
        elif winner != team1 and muhurta_strength > 1:  # Team2 won but muhurta strongly favors ascendant
            outcome_validation = False
        
        final_assignment_correct = assignment_correct and outcome_validation
        confidence = min(accuracy + 0.1, 1.0) if outcome_validation else max(accuracy - 0.2, 0.0)
        
        return {
            'match_id': match_data.get('match_id', 'Unknown'),
            'team1': team1,
            'team2': match_data.get('team2', 'Unknown'),
            'assignment_correct': final_assignment_correct,
            'confidence': confidence,
            'period_accuracy': accuracy,
            'periods_analyzed': total_predictions,
            'ascendant_favored_periods': ascendant_favored_periods,
            'descendant_favored_periods': descendant_favored_periods,
            'muhurta_strength': muhurta_strength,
            'winner': winner,
            'reason': f"Period accuracy: {accuracy:.2f}, Outcome validation: {outcome_validation}"
        }
    
    def generate_match_timeline(self, match_id, team1_name, team2_name, match_start_time, location_lat=19.0760, location_lon=72.8777):
        """Generate detailed match timeline with period-wise predictions"""
        
        print(f"Generating match timeline for {team1_name} vs {team2_name}...")
        
        if isinstance(match_start_time, str):
            match_start = pd.to_datetime(match_start_time).to_pydatetime()
        else:
            match_start = match_start_time
        
        timeline = []
        
        # Generate 15-minute intervals for 6 hours (24 periods)
        for period_num in range(24):
            period_start = match_start + timedelta(minutes=period_num * 15)
            period_end = period_start + timedelta(minutes=15)
            
            # Generate KP chart for this period
            period_chart = generate_kp_chart(period_start, location_lat, location_lon, self.nakshatra_df)
            
            if 'error' in period_chart:
                continue
            
            # Evaluate favorability
            favorability = evaluate_favorability(period_chart, pd.DataFrame())  # Empty df for chart-only analysis
            
            # Determine period characteristics
            period_analysis = self.analyze_period_characteristics(period_chart, favorability)
            
            timeline_entry = {
                'period_num': period_num + 1,
                'time_start': period_start.strftime('%H:%M'),
                'time_end': period_end.strftime('%H:%M'),
                'duration_minutes': 15,
                
                # Current planetary rulers
                'current_sub_lord': period_chart.get('moon_sub_lord', 'Unknown'),
                'current_sub_sub_lord': period_chart.get('moon_sub_sub_lord', 'Unknown'),
                'current_star_lord': period_chart.get('moon_star_lord', 'Unknown'),
                
                # Favorability
                'kp_favorability_score': favorability.get('final_score', 0),
                'favors_team': 'Ascendant' if favorability.get('final_score', 0) > 0 else 'Descendant',
                'favorability_strength': abs(favorability.get('final_score', 0)),
                
                # Period characteristics
                'high_scoring_probability': period_analysis['high_scoring_prob'],
                'collapse_probability': period_analysis['collapse_prob'],
                'wicket_probability': period_analysis['wicket_prob'],
                'momentum_indicator': period_analysis['momentum'],
                'period_nature': period_analysis['nature'],
                
                # Detailed analysis
                'key_influences': period_analysis['key_influences'],
                'recommendations': period_analysis['recommendations']
            }
            
            timeline.append(timeline_entry)
        
        return timeline
    
    def analyze_period_characteristics(self, chart, favorability):
        """Analyze characteristics of a specific period"""
        
        sub_lord = chart.get('moon_sub_lord', 'Unknown')
        sub_sub_lord = chart.get('moon_sub_sub_lord', 'Unknown')
        star_lord = chart.get('moon_star_lord', 'Unknown')
        
        # Base probabilities
        high_scoring_prob = 0.3
        collapse_prob = 0.2
        wicket_prob = 0.4
        momentum = 0.0
        
        key_influences = []
        recommendations = []
        
        # Analyze sub lord influence
        if sub_lord in self.benefic_planets:
            high_scoring_prob += 0.3
            momentum += 0.2
            key_influences.append(f"{sub_lord} sub lord brings positive energy")
            recommendations.append("Good time for aggressive batting")
        elif sub_lord in self.malefic_planets:
            collapse_prob += 0.3
            wicket_prob += 0.2
            momentum -= 0.2
            key_influences.append(f"{sub_lord} sub lord creates challenges")
            recommendations.append("Cautious approach recommended")
        
        # Analyze sub-sub lord influence
        if sub_sub_lord in self.benefic_planets:
            high_scoring_prob += 0.2
            key_influences.append(f"{sub_sub_lord} sub-sub lord supports scoring")
        elif sub_sub_lord in self.malefic_planets:
            wicket_prob += 0.15
            key_influences.append(f"{sub_sub_lord} sub-sub lord increases wicket risk")
        
        # Analyze star lord influence
        if star_lord in self.benefic_planets:
            momentum += 0.1
            key_influences.append(f"{star_lord} star lord provides stability")
        elif star_lord in self.malefic_planets:
            momentum -= 0.1
            key_influences.append(f"{star_lord} star lord creates pressure")
        
        # Determine period nature
        if high_scoring_prob > 0.6:
            nature = "High Scoring"
        elif collapse_prob > 0.5:
            nature = "Collapse Risk"
        elif wicket_prob > 0.6:
            nature = "Wicket-taking"
        elif momentum > 0.2:
            nature = "Momentum Building"
        elif momentum < -0.2:
            nature = "Pressure Period"
        else:
            nature = "Balanced"
        
        # Cap probabilities
        high_scoring_prob = min(high_scoring_prob, 0.9)
        collapse_prob = min(collapse_prob, 0.8)
        wicket_prob = min(wicket_prob, 0.8)
        momentum = max(-1.0, min(1.0, momentum))
        
        return {
            'high_scoring_prob': round(high_scoring_prob, 2),
            'collapse_prob': round(collapse_prob, 2),
            'wicket_prob': round(wicket_prob, 2),
            'momentum': round(momentum, 2),
            'nature': nature,
            'key_influences': key_influences,
            'recommendations': recommendations
        }
    
    def validate_multiple_matches(self, limit=50):
        """Validate ascendant assignments for multiple matches"""
        
        print(f"Validating ascendant assignments for up to {limit} matches...")
        
        # Get matches with data
        matches_query = """
        SELECT DISTINCT d.match_id
        FROM deliveries d
        WHERE d.timestamp IS NOT NULL
        LIMIT ?
        """
        
        matches_df = pd.read_sql_query(matches_query, self.conn, params=[limit])
        
        validation_results = []
        
        for _, match in matches_df.iterrows():
            match_id = match['match_id']
            
            try:
                validation = self.validate_ascendant_assignment(match_id)
                if validation:
                    validation_results.append(validation)
                    
                    if len(validation_results) % 10 == 0:
                        print(f"Validated {len(validation_results)} matches...")
                        
            except Exception as e:
                print(f"Error validating match {match_id}: {str(e)}")
                continue
        
        # Summary statistics
        if validation_results:
            correct_assignments = sum(1 for v in validation_results if v['assignment_correct'])
            avg_confidence = np.mean([v['confidence'] for v in validation_results])
            avg_accuracy = np.mean([v['period_accuracy'] for v in validation_results])
            
            print(f"\nValidation Summary:")
            print(f"Total matches validated: {len(validation_results)}")
            print(f"Correct assignments: {correct_assignments} ({correct_assignments/len(validation_results)*100:.1f}%)")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Average period accuracy: {avg_accuracy:.3f}")
            
            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)
            
            # Save results
            results_df = pd.DataFrame(validation_results)
            results_df.to_csv("reports/ascendant_validation_results.csv", index=False)
            print(f"Results saved to reports/ascendant_validation_results.csv")
        
        return validation_results
    
    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    """Main function to run ascendant validation"""
    
    validator = KPAscendantValidator()
    
    try:
        # Validate multiple matches
        results = validator.validate_multiple_matches(limit=20)
        
        # Example timeline generation
        if results:
            sample_match = results[0]
            print(f"\nGenerating sample timeline for match {sample_match['match_id']}...")
            
            timeline = validator.generate_match_timeline(
                sample_match['match_id'],
                sample_match['team1'],
                sample_match['team2'],
                datetime.now().replace(hour=19, minute=30, second=0, microsecond=0)
            )
            
            print(f"Generated timeline with {len(timeline)} periods")
            
            # Ensure reports directory exists
            os.makedirs("reports", exist_ok=True)
            
            # Save sample timeline
            with open("reports/sample_match_timeline.json", 'w') as f:
                json.dump(timeline, f, indent=2)
            
            print("Sample timeline saved to reports/sample_match_timeline.json")
    
    except Exception as e:
        print(f"Error in ascendant validation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        validator.close()

if __name__ == "__main__":
    main() 