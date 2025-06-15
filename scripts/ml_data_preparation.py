#!/usr/bin/env python3
"""
ML Data Preparation for KP Cricket Predictor
Extracts astrological periods and creates target variables based on actual performance
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Tuple, Optional

class KPMLDataPreparator:
    def __init__(self, db_path="training_analysis/cricket_predictions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Load nakshatra data for period calculations
        try:
            self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
            print(f"✓ Loaded nakshatra data: {len(self.nakshatra_df)} records")
        except Exception as e:
            print(f"❌ Error loading nakshatra data: {str(e)}")
            self.nakshatra_df = None
        
        print("KP ML Data Preparator initialized")
    
    def extract_astrological_periods(self, match_id: str) -> List[Dict]:
        """
        Extract astrological periods based on Sub Lord changes for a specific match
        Returns periods with variable durations based on actual planetary movements
        """
        
        print(f"Extracting astrological periods for match {match_id}...")
        
        # Get all deliveries for this match with astrological data
        query = """
        SELECT 
            d.id as delivery_id,
            d.timestamp,
            d.inning,
            d.over,
            d.ball,
            d.runs_off_bat,
            d.extras,
            d.wicket_kind,
            d.batting_team,
            d.bowling_team,
            ap.moon_sl,
            ap.moon_sub,
            ap.moon_ssl,
            ap.success_score
        FROM deliveries d
        JOIN astrological_predictions ap ON d.id = ap.delivery_id
        WHERE d.match_id = ? 
        AND d.timestamp IS NOT NULL
        AND ap.success_score IS NOT NULL
        ORDER BY d.timestamp
        """
        
        deliveries_df = pd.read_sql_query(query, self.conn, params=[match_id])
        
        if deliveries_df.empty:
            print(f"No deliveries found for match {match_id}")
            return []
        
        # Convert timestamp to datetime
        deliveries_df['timestamp'] = pd.to_datetime(deliveries_df['timestamp'])
        
        # Extract astrological periods based on Sub Lord changes
        periods = []
        current_period = None
        period_deliveries = []
        
        for idx, delivery in deliveries_df.iterrows():
            # Check if we need to start a new period
            if current_period is None:
                # First delivery - start first period
                current_period = {
                    'period_num': 1,
                    'start_time': delivery['timestamp'],
                    'moon_sl': delivery['moon_sl'],
                    'moon_sub': delivery['moon_sub'],
                    'moon_ssl': delivery['moon_ssl'],
                    'kp_prediction_score': delivery['success_score'],
                    'deliveries': []
                }
                period_deliveries = []
            
            # Check if Sub Lord or Sub-Sub Lord changed (new period)
            elif (delivery['moon_sub'] != current_period['moon_sub'] or 
                  delivery['moon_ssl'] != current_period['moon_ssl']):
                
                # Finalize current period
                if period_deliveries:
                    current_period['end_time'] = period_deliveries[-1]['timestamp']
                    current_period['duration_minutes'] = (
                        current_period['end_time'] - current_period['start_time']
                    ).total_seconds() / 60
                    current_period['deliveries'] = period_deliveries.copy()
                    current_period['num_deliveries'] = len(period_deliveries)
                    
                    periods.append(current_period)
                
                # Start new period
                current_period = {
                    'period_num': len(periods) + 1,
                    'start_time': delivery['timestamp'],
                    'moon_sl': delivery['moon_sl'],
                    'moon_sub': delivery['moon_sub'],
                    'moon_ssl': delivery['moon_ssl'],
                    'kp_prediction_score': delivery['success_score'],
                    'deliveries': []
                }
                period_deliveries = []
            
            # Add delivery to current period
            period_deliveries.append({
                'delivery_id': delivery['delivery_id'],
                'timestamp': delivery['timestamp'],
                'inning': delivery['inning'],
                'over': delivery['over'],
                'ball': delivery['ball'],
                'runs_off_bat': delivery['runs_off_bat'],
                'extras': delivery['extras'],
                'wicket_kind': delivery['wicket_kind'],
                'batting_team': delivery['batting_team'],
                'bowling_team': delivery['bowling_team']
            })
        
        # Finalize last period
        if current_period and period_deliveries:
            current_period['end_time'] = period_deliveries[-1]['timestamp']
            current_period['duration_minutes'] = (
                current_period['end_time'] - current_period['start_time']
            ).total_seconds() / 60
            current_period['deliveries'] = period_deliveries.copy()
            current_period['num_deliveries'] = len(period_deliveries)
            periods.append(current_period)
        
        print(f"Extracted {len(periods)} astrological periods")
        return periods
    
    def calculate_period_performance(self, period: Dict, team1: str, team2: str) -> Dict:
        """
        Calculate actual performance metrics for a period
        Returns performance scores for both teams
        """
        
        deliveries = period['deliveries']
        if not deliveries:
            return {'team1_performance': 0, 'team2_performance': 0, 'performance_difference': 0}
        
        # Initialize performance metrics
        team1_runs = 0
        team2_runs = 0
        team1_wickets_lost = 0
        team2_wickets_lost = 0
        team1_balls_faced = 0
        team2_balls_faced = 0
        
        for delivery in deliveries:
            batting_team = delivery['batting_team']
            runs = delivery['runs_off_bat'] or 0
            extras = delivery['extras'] or 0
            total_runs = runs + extras
            wicket = 1 if delivery['wicket_kind'] else 0
            
            if batting_team == team1:
                team1_runs += total_runs
                team1_wickets_lost += wicket
                team1_balls_faced += 1
            elif batting_team == team2:
                team2_runs += total_runs
                team2_wickets_lost += wicket
                team2_balls_faced += 1
        
        # Calculate performance scores (higher = better)
        team1_strike_rate = (team1_runs / team1_balls_faced * 100) if team1_balls_faced > 0 else 0
        team1_wicket_penalty = team1_wickets_lost * 10
        team1_performance = team1_strike_rate - team1_wicket_penalty
        
        team2_strike_rate = (team2_runs / team2_balls_faced * 100) if team2_balls_faced > 0 else 0
        team2_wicket_penalty = team2_wickets_lost * 10
        team2_performance = team2_strike_rate - team2_wicket_penalty
        
        performance_difference = team1_performance - team2_performance
        
        return {
            'team1_performance': team1_performance,
            'team2_performance': team2_performance,
            'performance_difference': performance_difference,
            'team1_runs': team1_runs,
            'team2_runs': team2_runs,
            'team1_wickets_lost': team1_wickets_lost,
            'team2_wickets_lost': team2_wickets_lost
        }
    
    def create_match_dynamics_targets(self, period_data: Dict, performance: Dict) -> Dict:
        """
        Create match dynamics target variables for ML training
        Returns probabilities for high scoring, collapse, wicket pressure, and momentum shift
        """
        
        deliveries = period_data['deliveries']
        total_runs = performance['team1_runs'] + performance['team2_runs']
        total_wickets = performance['team1_wickets_lost'] + performance['team2_wickets_lost']
        num_deliveries = len(deliveries)
        
        # Calculate rates
        run_rate = (total_runs / num_deliveries * 6) if num_deliveries > 0 else 0  # Runs per over
        wicket_rate = (total_wickets / num_deliveries * 6) if num_deliveries > 0 else 0  # Wickets per over
        
        # HIGH SCORING TARGET (0-1): Based on run rate
        # High scoring if > 8 runs per over in this period
        high_scoring_target = 1 if run_rate > 8.0 else 0
        high_scoring_probability = min(run_rate / 12.0, 1.0)  # Normalize to 0-1
        
        # COLLAPSE TARGET (0-1): Based on wickets falling with low runs
        # Collapse if 2+ wickets fell with run rate < 6
        collapse_condition = (total_wickets >= 2 and run_rate < 6.0) or (total_wickets >= 3)
        collapse_target = 1 if collapse_condition else 0
        collapse_probability = min((total_wickets * 2 + (6.0 - run_rate)) / 10.0, 1.0) if total_wickets > 0 else 0
        
        # WICKET PRESSURE TARGET (0-1): Based on wicket rate
        # High wicket pressure if > 1 wicket per over
        wicket_pressure_target = 1 if wicket_rate > 1.0 else 0
        wicket_pressure_probability = min(wicket_rate / 2.0, 1.0)
        
        # MOMENTUM SHIFT TARGET (-1 to +1): Based on performance swing
        # Calculate momentum based on runs vs wickets balance
        momentum_score = (run_rate - 6.0) / 6.0 - (wicket_rate * 2)  # Normalize around 6 RPO
        momentum_shift_target = max(-1.0, min(1.0, momentum_score))
        
        # Convert momentum to binary for classification (positive momentum = 1)
        momentum_binary_target = 1 if momentum_shift_target > 0.1 else 0
        
        return {
            'high_scoring_target': high_scoring_target,
            'high_scoring_probability': round(high_scoring_probability, 3),
            'collapse_target': collapse_target, 
            'collapse_probability': round(max(0, collapse_probability), 3),
            'wicket_pressure_target': wicket_pressure_target,
            'wicket_pressure_probability': round(wicket_pressure_probability, 3),
            'momentum_shift_target': round(momentum_shift_target, 3),
            'momentum_binary_target': momentum_binary_target,
            'period_run_rate': round(run_rate, 2),
            'period_wicket_rate': round(wicket_rate, 2)
        }

    def create_target_variable(self, kp_prediction_score: float, performance_difference: float) -> Dict:
        """
        Create target variable based on KP prediction vs actual performance
        Returns both binary and strength-based targets
        """
        
        # KP prediction direction
        kp_favors_ascendant = kp_prediction_score > 0
        kp_prediction_strength = abs(kp_prediction_score)
        
        # Actual performance direction  
        team1_performed_better = performance_difference > 0
        actual_performance_strength = abs(performance_difference)
        
        # Binary target: Was KP prediction direction correct?
        prediction_correct = (kp_favors_ascendant == team1_performed_better)
        
        # Strength-based target (5 levels: 0=Incorrect, 1=Weak, 2=Moderate, 3=Strong, 4=Very Strong)
        if prediction_correct:
            combined_strength = (kp_prediction_strength + actual_performance_strength / 10) / 2
            
            if combined_strength >= 15:
                target_strength = 4  # Very Strong Correct
            elif combined_strength >= 10:
                target_strength = 3  # Strong Correct  
            elif combined_strength >= 5:
                target_strength = 2  # Moderate Correct
            else:
                target_strength = 1  # Weak Correct
        else:
            target_strength = 0  # Incorrect
        
        return {
            'binary_target': 1 if prediction_correct else 0,
            'strength_target': target_strength,
            'kp_favors_ascendant': kp_favors_ascendant,
            'team1_performed_better': team1_performed_better,
            'kp_prediction_strength': kp_prediction_strength,
            'actual_performance_strength': actual_performance_strength
        }
    
    def process_match(self, match_id: str) -> List[Dict]:
        """
        Process a single match and return ML training records
        """
        
        # Get match info
        match_query = "SELECT team1, team2, winner FROM matches WHERE match_id = ?"
        match_info = pd.read_sql_query(match_query, self.conn, params=[match_id])
        
        if match_info.empty:
            return []
        
        team1 = match_info.iloc[0]['team1']
        team2 = match_info.iloc[0]['team2']
        
        # Extract astrological periods
        periods = self.extract_astrological_periods(match_id)
        
        if not periods:
            return []
        
        # Process each period
        training_records = []
        
        for period in periods:
            # Calculate actual performance
            performance = self.calculate_period_performance(period, team1, team2)
            
            # Create target variable (KP prediction accuracy)
            target = self.create_target_variable(
                period['kp_prediction_score'], 
                performance['performance_difference']
            )
            
            # Create match dynamics targets (new ML targets)
            dynamics_targets = self.create_match_dynamics_targets(period, performance)
            
            # Create training record
            record = {
                'match_id': match_id,
                'period_num': period['period_num'],
                'team1': team1,
                'team2': team2,
                'start_time': period['start_time'].isoformat(),
                'duration_minutes': period['duration_minutes'],
                'num_deliveries': period['num_deliveries'],
                'moon_sl': period['moon_sl'],
                'moon_sub': period['moon_sub'],
                'moon_ssl': period['moon_ssl'],
                'kp_prediction_score': period['kp_prediction_score'],
                **performance,
                **target,
                **dynamics_targets
            }
            
            training_records.append(record)
        
        return training_records
    
    def process_all_matches(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Process all matches and create complete ML training dataset
        """
        
        print("🚀 Processing ALL matches for complete ML training dataset...")
        
        # Get all match IDs
        if limit:
            query = f"SELECT DISTINCT match_id FROM matches ORDER BY match_id LIMIT {limit}"
            print(f"Processing {limit} matches for testing...")
        else:
            query = "SELECT DISTINCT match_id FROM matches ORDER BY match_id"
            print("Processing ALL matches...")
            
        match_ids = pd.read_sql_query(query, self.conn)['match_id'].tolist()
        
        print(f"Found {len(match_ids)} matches to process")
        
        all_records = []
        processed_count = 0
        error_count = 0
        total_periods = 0
        
        # Progress tracking
        checkpoint_interval = 100
        
        for i, match_id in enumerate(match_ids, 1):
            try:
                records = self.process_match(match_id)
                all_records.extend(records)
                processed_count += 1
                total_periods += len(records)
                
                # Progress updates
                if processed_count % checkpoint_interval == 0:
                    print(f"✓ Processed {processed_count:,}/{len(match_ids):,} matches")
                    print(f"  Generated {total_periods:,} periods so far")
                    print(f"  Average {total_periods/processed_count:.1f} periods per match")
                    
                    # Save checkpoint
                    if all_records:
                        checkpoint_df = pd.DataFrame(all_records)
                        checkpoint_file = f"ml_data/kp_training_checkpoint_{processed_count}.csv"
                        checkpoint_df.to_csv(checkpoint_file, index=False)
                        print(f"  Checkpoint saved: {checkpoint_file}")
                    
            except Exception as e:
                print(f"❌ Error processing match {match_id}: {str(e)}")
                error_count += 1
                continue
        
        print(f"\n🎉 Processing Complete!")
        print(f"  Matches processed: {processed_count:,}")
        print(f"  Matches with errors: {error_count:,}")
        print(f"  Total training periods: {total_periods:,}")
        print(f"  Average periods per match: {total_periods/processed_count:.1f}")
        
        if all_records:
            df = pd.DataFrame(all_records)
            
            # Save final dataset
            os.makedirs("ml_data", exist_ok=True)
            output_file = f"ml_data/kp_complete_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"  Final dataset saved: {output_file}")
            
            # Print comprehensive summary
            self.print_dataset_summary(df)
            
            return df
        else:
            print("❌ No training records generated")
            return pd.DataFrame()
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """Print comprehensive summary statistics of the training dataset"""
        
        print(f"\n📊 COMPLETE ML TRAINING DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\n🎯 Dataset Scale:")
        print(f"  Total astrological periods: {len(df):,}")
        print(f"  Unique matches: {df['match_id'].nunique():,}")
        print(f"  Average periods per match: {len(df) / df['match_id'].nunique():.1f}")
        
        print(f"\n🎯 Target Variable Distribution:")
        print(f"  Correct KP predictions: {df['binary_target'].mean():.1%}")
        
        strength_dist = df['strength_target'].value_counts().sort_index()
        print(f"\n  Strength Target Distribution:")
        labels = ['Incorrect', 'Weak Correct', 'Moderate Correct', 'Strong Correct', 'Very Strong Correct']
        for strength, count in strength_dist.items():
            pct = count / len(df) * 100
            print(f"    {strength} ({labels[strength]}): {count:,} ({pct:.1f}%)")
        
        print(f"\n🎯 Match Dynamics Target Distribution:")
        print(f"  High Scoring periods: {df['high_scoring_target'].mean():.1%}")
        print(f"  Collapse periods: {df['collapse_target'].mean():.1%}")
        print(f"  Wicket Pressure periods: {df['wicket_pressure_target'].mean():.1%}")
        print(f"  Positive Momentum periods: {df['momentum_binary_target'].mean():.1%}")
        
        print(f"\n📊 Period Performance Metrics:")
        print(f"  Average run rate: {df['period_run_rate'].mean():.2f} RPO")
        print(f"  Average wicket rate: {df['period_wicket_rate'].mean():.2f} WPO")
        print(f"  High scoring probability: {df['high_scoring_probability'].mean():.3f}")
        print(f"  Collapse probability: {df['collapse_probability'].mean():.3f}")
        print(f"  Wicket pressure probability: {df['wicket_pressure_probability'].mean():.3f}")
        print(f"  Momentum shift range: {df['momentum_shift_target'].min():.2f} to {df['momentum_shift_target'].max():.2f}")
        
        print(f"\n🌟 KP Features Summary:")
        print(f"  KP Score Range: {df['kp_prediction_score'].min():.2f} to {df['kp_prediction_score'].max():.2f}")
        print(f"  Average KP prediction strength: {df['kp_prediction_strength'].mean():.2f}")
        print(f"  KP favors ascendant: {df['kp_favors_ascendant'].mean():.1%}")
        
        print(f"\n⏱️ Period Characteristics:")
        print(f"  Duration range: {df['duration_minutes'].min():.1f} to {df['duration_minutes'].max():.1f} minutes")
        print(f"  Average period duration: {df['duration_minutes'].mean():.1f} minutes")
        print(f"  Deliveries per period: {df['num_deliveries'].min()} to {df['num_deliveries'].max()}")
        print(f"  Average deliveries per period: {df['num_deliveries'].mean():.1f}")
        
        print(f"\n🏏 Performance Analysis:")
        print(f"  Team1 performed better: {df['team1_performed_better'].mean():.1%}")
        print(f"  Average performance difference: {df['performance_difference'].mean():.2f}")
        print(f"  Performance difference range: {df['performance_difference'].min():.1f} to {df['performance_difference'].max():.1f}")
        
        print(f"\n🌙 Planetary Lord Distribution:")
        print(f"  Most common Star Lords: {df['moon_sl'].value_counts().head(3).to_dict()}")
        print(f"  Most common Sub Lords: {df['moon_sub'].value_counts().head(3).to_dict()}")
        print(f"  Most common Sub-Sub Lords: {df['moon_ssl'].value_counts().head(3).to_dict()}")
        
        print(f"\n✅ Data Quality:")
        print(f"  No missing values: {df.isnull().sum().sum() == 0}")
        print(f"  All periods have deliveries: {(df['num_deliveries'] > 0).all()}")
        print(f"  Valid target range: {df['strength_target'].min()} to {df['strength_target'].max()}")
        print(f"  Valid KP score range: Reasonable spread from {df['kp_prediction_score'].min():.1f} to {df['kp_prediction_score'].max():.1f}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main function to run complete ML data preparation"""
    
    preparator = KPMLDataPreparator()
    
    try:
        print("🎯 KP Cricket Predictor - Complete ML Data Preparation")
        print("=" * 60)
        print("This will process ALL matches (~4000) to create the complete training dataset")
        print("Expected: ~200,000 astrological periods")
        print("Estimated time: 30-60 minutes")
        print()
        
        # Process all matches
        complete_df = preparator.process_all_matches()
        
        if not complete_df.empty:
            print("\n🎉 SUCCESS! Complete ML training dataset created!")
            print(f"Ready for model training with {len(complete_df):,} astrological periods")
            
        else:
            print("❌ Failed to create training dataset")
            
    except Exception as e:
        print(f"❌ Error in ML data preparation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        preparator.close()

if __name__ == "__main__":
    main() 