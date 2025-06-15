#!/usr/bin/env python3
"""
Enhanced KP Trainer with Ascendant Validation and Astrological Periods
Trains KP models using astrological period durations for consistency with predictions
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import sys

# Fix import paths - same approach as app.py
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
ROOT_DIR = os.path.dirname(scripts_dir)
sys.path.insert(0, ROOT_DIR)

from scripts.kp_ascendant_validator import KPAscendantValidator
from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability

class KPEnhancedTrainer:
    def __init__(self, db_path="training_analysis/cricket_predictions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.validator = KPAscendantValidator(db_path)
        
        # Load nakshatra data for chart generation
        self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
        
        # Model storage
        self.models = {}
        self.feature_importance = {}
        self.validation_results = {}
        
        print("Enhanced KP Trainer with Astrological Periods initialized")
    
    def validate_and_correct_training_data(self, limit=200):
        """Validate ascendant assignments and correct training data"""
        
        print(f"Validating and correcting training data for {limit} matches...")
        
        # Get validation results
        validation_results = self.validator.validate_multiple_matches(limit=limit)
        
        if not validation_results:
            print("No validation results obtained")
            return pd.DataFrame()
        
        # Create corrected dataset
        corrected_data = []
        
        for result in validation_results:
            match_id = result['match_id']
            assignment_correct = result['assignment_correct']
            confidence = result['confidence']
            
            if confidence < 0.4:  # Skip low confidence matches
                continue
            
            # Get match data
            match_query = """
            SELECT 
                d.*,
                mc.muhurta_strength_score,
                mc.ascendant_favorability,
                mc.descendant_favorability,
                mc.moon_sub_lord,
                mc.moon_sub_sub_lord,
                mc.moon_star_lord,
                mc.match_start_time
            FROM deliveries d
            LEFT JOIN muhurta_charts mc ON d.match_id = mc.match_id
            WHERE d.match_id = ?
            """
            
            match_df = pd.read_sql_query(match_query, self.conn, params=[match_id])
            
            if match_df.empty:
                continue
            
            # Apply correction if needed
            if not assignment_correct:
                # Flip ascendant/descendant assignments
                match_df['team1_original'] = match_df['team1'].copy()
                match_df['team2_original'] = match_df['team2'].copy()
                match_df['team1'] = match_df['team2_original']
                match_df['team2'] = match_df['team1_original']
                
                # Flip favorability scores
                if 'ascendant_favorability' in match_df.columns:
                    match_df['ascendant_favorability'] = -match_df['ascendant_favorability']
                if 'descendant_favorability' in match_df.columns:
                    match_df['descendant_favorability'] = -match_df['descendant_favorability']
                if 'muhurta_strength_score' in match_df.columns:
                    match_df['muhurta_strength_score'] = -match_df['muhurta_strength_score']
                
                match_df['assignment_corrected'] = True
            else:
                match_df['assignment_corrected'] = False
            
            match_df['validation_confidence'] = confidence
            corrected_data.append(match_df)
        
        if corrected_data:
            corrected_df = pd.concat(corrected_data, ignore_index=True)
            print(f"Created corrected dataset with {len(corrected_df)} deliveries from {len(corrected_data)} matches")
            
            # Save corrected data
            os.makedirs("models", exist_ok=True)
            corrected_df.to_csv("models/corrected_training_data.csv", index=False)
            
            return corrected_df
        else:
            print("No corrected data created")
            return pd.DataFrame()
    
    def engineer_enhanced_features(self, df):
        """Engineer enhanced KP features from corrected data using astrological periods"""
        
        if df.empty:
            return pd.DataFrame()
        
        print("Engineering enhanced KP features with astrological periods...")
        
        # Group by match for match-level features
        match_features = []
        
        for match_id in df['match_id'].unique():
            match_df = df[df['match_id'] == match_id]
            
            if match_df.empty:
                continue
            
            # Generate astrological periods for this match
            astrological_periods = self.generate_astrological_periods_for_match(match_id)
            
            if not astrological_periods:
                continue
            
            # Basic match info
            features = {
                'match_id': match_id,
                'team1': match_df['team1'].iloc[0],
                'team2': match_df['team2'].iloc[0],
                'winner': match_df['winner'].iloc[0] if 'winner' in match_df.columns else None,
                'assignment_corrected': match_df['assignment_corrected'].iloc[0],
                'validation_confidence': match_df['validation_confidence'].iloc[0]
            }
            
            # Astrological period features
            features['total_periods'] = len(astrological_periods)
            features['avg_period_duration'] = np.mean([p['duration_minutes'] for p in astrological_periods])
            features['total_match_duration'] = sum([p['duration_minutes'] for p in astrological_periods])
            
            # KP favorability across periods
            kp_scores = [p['kp_score'] for p in astrological_periods]
            features['avg_kp_score'] = np.mean(kp_scores)
            features['max_kp_score'] = np.max(kp_scores)
            features['min_kp_score'] = np.min(kp_scores)
            features['kp_score_variance'] = np.var(kp_scores)
            
            # Count favorable periods
            features['ascendant_favorable_periods'] = sum(1 for score in kp_scores if score > 0.1)
            features['descendant_favorable_periods'] = sum(1 for score in kp_scores if score < -0.1)
            features['neutral_periods'] = sum(1 for score in kp_scores if -0.1 <= score <= 0.1)
            
            # Sub lord analysis across periods
            sub_lords = [p['sub_lord'] for p in astrological_periods]
            sub_sub_lords = [p['sub_sub_lord'] for p in astrological_periods]
            
            benefic_planets = ["Jupiter", "Venus"]
            malefic_planets = ["Saturn", "Mars", "Sun", "Rahu", "Ketu"]
            
            features['benefic_sub_lord_periods'] = sum(1 for sl in sub_lords if sl in benefic_planets)
            features['malefic_sub_lord_periods'] = sum(1 for sl in sub_lords if sl in malefic_planets)
            features['benefic_sub_sub_lord_periods'] = sum(1 for ssl in sub_sub_lords if ssl in benefic_planets)
            features['malefic_sub_sub_lord_periods'] = sum(1 for ssl in sub_sub_lords if ssl in malefic_planets)
            
            # Planetary influence ratios
            total_periods = len(astrological_periods)
            features['benefic_period_ratio'] = features['benefic_sub_lord_periods'] / total_periods if total_periods > 0 else 0
            features['malefic_period_ratio'] = features['malefic_sub_lord_periods'] / total_periods if total_periods > 0 else 0
            
            # Period-weighted KP features
            period_durations = [p['duration_minutes'] for p in astrological_periods]
            total_duration = sum(period_durations)
            
            if total_duration > 0:
                # Weight KP scores by period duration
                weighted_kp_score = sum(score * duration for score, duration in zip(kp_scores, period_durations)) / total_duration
                features['duration_weighted_kp_score'] = weighted_kp_score
                
                # Weight favorable periods by duration
                ascendant_duration = sum(duration for score, duration in zip(kp_scores, period_durations) if score > 0.1)
                descendant_duration = sum(duration for score, duration in zip(kp_scores, period_durations) if score < -0.1)
                
                features['ascendant_favorable_duration_ratio'] = ascendant_duration / total_duration
                features['descendant_favorable_duration_ratio'] = descendant_duration / total_duration
            else:
                features['duration_weighted_kp_score'] = 0
                features['ascendant_favorable_duration_ratio'] = 0
                features['descendant_favorable_duration_ratio'] = 0

            # Muhurta-based features (from original data if available)
            if 'muhurta_strength_score' in match_df.columns:
                features['muhurta_strength'] = match_df['muhurta_strength_score'].iloc[0] or 0
                features['ascendant_favorability'] = match_df['ascendant_favorability'].iloc[0] or 0
                features['descendant_favorability'] = match_df['descendant_favorability'].iloc[0] or 0
            else:
                features['muhurta_strength'] = 0
                features['ascendant_favorability'] = 0
                features['descendant_favorability'] = 0
            
            # Sub lord analysis
            sub_lord = match_df['moon_sub_lord'].iloc[0] if 'moon_sub_lord' in match_df.columns else 'Unknown'
            sub_sub_lord = match_df['moon_sub_sub_lord'].iloc[0] if 'moon_sub_sub_lord' in match_df.columns else 'Unknown'
            star_lord = match_df['moon_star_lord'].iloc[0] if 'moon_star_lord' in match_df.columns else 'Unknown'
            
            # Planetary influence features
            benefic_planets = ["Jupiter", "Venus"]
            malefic_planets = ["Saturn", "Mars", "Sun", "Rahu", "Ketu"]
            
            features['sub_lord_benefic'] = 1 if sub_lord in benefic_planets else 0
            features['sub_lord_malefic'] = 1 if sub_lord in malefic_planets else 0
            features['sub_sub_lord_benefic'] = 1 if sub_sub_lord in benefic_planets else 0
            features['sub_sub_lord_malefic'] = 1 if sub_sub_lord in malefic_planets else 0
            features['star_lord_benefic'] = 1 if star_lord in benefic_planets else 0
            features['star_lord_malefic'] = 1 if star_lord in malefic_planets else 0
            
            # Combined planetary strength
            features['total_benefic_influence'] = (
                features['sub_lord_benefic'] + 
                features['sub_sub_lord_benefic'] + 
                features['star_lord_benefic']
            )
            features['total_malefic_influence'] = (
                features['sub_lord_malefic'] + 
                features['sub_sub_lord_malefic'] + 
                features['star_lord_malefic']
            )
            features['net_planetary_influence'] = features['total_benefic_influence'] - features['total_malefic_influence']
            
            # Match performance features
            team1_deliveries = match_df[match_df['batting_team'] == features['team1']]
            team2_deliveries = match_df[match_df['batting_team'] == features['team2']]
            
            features['team1_total_runs'] = team1_deliveries['runs_off_bat'].sum()
            features['team2_total_runs'] = team2_deliveries['runs_off_bat'].sum()
            features['team1_wickets'] = len(team1_deliveries[team1_deliveries['wicket_type'].notna()])
            features['team2_wickets'] = len(team2_deliveries[team2_deliveries['wicket_type'].notna()])
            
            # Performance ratios
            total_runs = features['team1_total_runs'] + features['team2_total_runs']
            features['team1_run_share'] = features['team1_total_runs'] / total_runs if total_runs > 0 else 0.5
            features['run_difference'] = features['team1_total_runs'] - features['team2_total_runs']
            features['wicket_difference'] = features['team2_wickets'] - features['team1_wickets']  # Lower wickets is better
            
            # Target variables
            features['team1_wins'] = 1 if features['winner'] == features['team1'] else 0
            features['high_scoring_match'] = 1 if total_runs > 300 else 0
            features['total_runs'] = total_runs
            
            # KP prediction alignment
            kp_favors_ascendant = features['muhurta_strength'] > 0
            team1_performed_better = features['run_difference'] > 0
            features['kp_prediction_correct'] = 1 if kp_favors_ascendant == team1_performed_better else 0
            
            match_features.append(features)
        
        features_df = pd.DataFrame(match_features)
        print(f"Engineered features for {len(features_df)} matches")
        
        return features_df
    
    def train_enhanced_models(self, features_df):
        """Train enhanced KP models with validated data"""
        
        if features_df.empty:
            print("No features available for training")
            return
        
        print("Training enhanced KP models...")
        
        # Prepare feature columns
        feature_cols = [
            'muhurta_strength', 'ascendant_favorability', 'descendant_favorability',
            'sub_lord_benefic', 'sub_lord_malefic', 'sub_sub_lord_benefic', 'sub_sub_lord_malefic',
            'star_lord_benefic', 'star_lord_malefic', 'total_benefic_influence', 'total_malefic_influence',
            'net_planetary_influence', 'validation_confidence'
        ]
        
        X = features_df[feature_cols].fillna(0)
        
        # Model 1: Winner Prediction
        if 'team1_wins' in features_df.columns:
            y_winner = features_df['team1_wins']
            valid_indices = y_winner.notna()
            
            if valid_indices.sum() > 10:
                X_winner = X[valid_indices]
                y_winner = y_winner[valid_indices]
                
                X_train, X_test, y_train, y_test = train_test_split(X_winner, y_winner, test_size=0.2, random_state=42)
                
                winner_model = RandomForestClassifier(n_estimators=100, random_state=42)
                winner_model.fit(X_train, y_train)
                
                y_pred = winner_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models['winner_prediction'] = winner_model
                self.feature_importance['winner_prediction'] = dict(zip(feature_cols, winner_model.feature_importances_))
                self.validation_results['winner_prediction'] = {
                    'accuracy': accuracy,
                    'samples': len(X_winner),
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                print(f"Winner Prediction Model - Accuracy: {accuracy:.3f} (n={len(X_winner)})")
        
        # Model 2: High Scoring Match Prediction
        if 'high_scoring_match' in features_df.columns:
            y_scoring = features_df['high_scoring_match']
            valid_indices = y_scoring.notna()
            
            if valid_indices.sum() > 10:
                X_scoring = X[valid_indices]
                y_scoring = y_scoring[valid_indices]
                
                X_train, X_test, y_train, y_test = train_test_split(X_scoring, y_scoring, test_size=0.2, random_state=42)
                
                scoring_model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring_model.fit(X_train, y_train)
                
                y_pred = scoring_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models['high_scoring_prediction'] = scoring_model
                self.feature_importance['high_scoring_prediction'] = dict(zip(feature_cols, scoring_model.feature_importances_))
                self.validation_results['high_scoring_prediction'] = {
                    'accuracy': accuracy,
                    'samples': len(X_scoring)
                }
                
                print(f"High Scoring Match Model - Accuracy: {accuracy:.3f} (n={len(X_scoring)})")
        
        # Model 3: Total Runs Prediction
        if 'total_runs' in features_df.columns:
            y_runs = features_df['total_runs']
            valid_indices = (y_runs.notna()) & (y_runs > 0)
            
            if valid_indices.sum() > 10:
                X_runs = X[valid_indices]
                y_runs = y_runs[valid_indices]
                
                X_train, X_test, y_train, y_test = train_test_split(X_runs, y_runs, test_size=0.2, random_state=42)
                
                runs_model = RandomForestRegressor(n_estimators=100, random_state=42)
                runs_model.fit(X_train, y_train)
                
                y_pred = runs_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                self.models['total_runs_prediction'] = runs_model
                self.feature_importance['total_runs_prediction'] = dict(zip(feature_cols, runs_model.feature_importances_))
                self.validation_results['total_runs_prediction'] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'samples': len(X_runs)
                }
                
                print(f"Total Runs Model - R²: {r2:.3f}, RMSE: {rmse:.1f} (n={len(X_runs)})")
        
        # Save models
        self.save_models()
    
    def save_models(self):
        """Save trained models and metadata"""
        
        os.makedirs("models", exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f"models/{model_name}_enhanced.pkl")
        
        # Save feature importance
        import json
        with open("models/feature_importance_enhanced.json", 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        # Save validation results
        with open("models/validation_results_enhanced.json", 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"Saved {len(self.models)} enhanced models to models/ directory")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        
        print("\n" + "="*60)
        print("ENHANCED KP TRAINING REPORT")
        print("="*60)
        
        print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Models trained: {len(self.models)}")
        
        for model_name, results in self.validation_results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            
            if 'accuracy' in results:
                print(f"Accuracy: {results['accuracy']:.3f}")
            if 'r2_score' in results:
                print(f"R² Score: {results['r2_score']:.3f}")
            if 'rmse' in results:
                print(f"RMSE: {results['rmse']:.1f}")
            
            print(f"Training samples: {results['samples']}")
            
            # Top feature importance
            if model_name in self.feature_importance:
                importance = self.feature_importance[model_name]
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 features:")
                for feature, imp in top_features:
                    print(f"  {feature}: {imp:.3f}")
        
        print("\n" + "="*60)
    
    def predict_match(self, team1, team2, match_start_time, location_lat=19.0760, location_lon=72.8777):
        """Predict match outcome using enhanced models"""
        
        if not self.models:
            print("No models available. Please train models first.")
            return None
        
        print(f"Predicting match: {team1} vs {team2}")
        
        # Generate timeline for match
        timeline = self.validator.generate_match_timeline(
            "prediction", team1, team2, match_start_time, location_lat, location_lon
        )
        
        if not timeline:
            print("Could not generate match timeline")
            return None
        
        # Extract features from timeline
        # Use the first period (match start) for primary prediction
        first_period = timeline[0]
        
        features = {
            'muhurta_strength': first_period['kp_favorability_score'],
            'ascendant_favorability': max(0, first_period['kp_favorability_score']),
            'descendant_favorability': max(0, -first_period['kp_favorability_score']),
            'sub_lord_benefic': 1 if first_period['current_sub_lord'] in ["Jupiter", "Venus"] else 0,
            'sub_lord_malefic': 1 if first_period['current_sub_lord'] in ["Saturn", "Mars", "Sun", "Rahu", "Ketu"] else 0,
            'sub_sub_lord_benefic': 1 if first_period['current_sub_sub_lord'] in ["Jupiter", "Venus"] else 0,
            'sub_sub_lord_malefic': 1 if first_period['current_sub_sub_lord'] in ["Saturn", "Mars", "Sun", "Rahu", "Ketu"] else 0,
            'star_lord_benefic': 1 if first_period['current_star_lord'] in ["Jupiter", "Venus"] else 0,
            'star_lord_malefic': 1 if first_period['current_star_lord'] in ["Saturn", "Mars", "Sun", "Rahu", "Ketu"] else 0,
            'validation_confidence': 1.0  # Full confidence for new predictions
        }
        
        # Calculate derived features
        features['total_benefic_influence'] = (
            features['sub_lord_benefic'] + features['sub_sub_lord_benefic'] + features['star_lord_benefic']
        )
        features['total_malefic_influence'] = (
            features['sub_lord_malefic'] + features['sub_sub_lord_malefic'] + features['star_lord_malefic']
        )
        features['net_planetary_influence'] = features['total_benefic_influence'] - features['total_malefic_influence']
        
        # Prepare feature vector
        feature_cols = [
            'muhurta_strength', 'ascendant_favorability', 'descendant_favorability',
            'sub_lord_benefic', 'sub_lord_malefic', 'sub_sub_lord_benefic', 'sub_sub_lord_malefic',
            'star_lord_benefic', 'star_lord_malefic', 'total_benefic_influence', 'total_malefic_influence',
            'net_planetary_influence', 'validation_confidence'
        ]
        
        X = np.array([[features[col] for col in feature_cols]])
        
        predictions = {}
        
        # Make predictions with each model
        if 'winner_prediction' in self.models:
            winner_prob = self.models['winner_prediction'].predict_proba(X)[0]
            predictions['winner'] = {
                'team1_win_probability': winner_prob[1],
                'team2_win_probability': winner_prob[0],
                'predicted_winner': team1 if winner_prob[1] > 0.5 else team2,
                'confidence': max(winner_prob)
            }
        
        if 'high_scoring_prediction' in self.models:
            scoring_prob = self.models['high_scoring_prediction'].predict_proba(X)[0]
            predictions['scoring'] = {
                'high_scoring_probability': scoring_prob[1],
                'low_scoring_probability': scoring_prob[0],
                'predicted_nature': 'High Scoring' if scoring_prob[1] > 0.5 else 'Low Scoring'
            }
        
        if 'total_runs_prediction' in self.models:
            predicted_runs = self.models['total_runs_prediction'].predict(X)[0]
            predictions['runs'] = {
                'predicted_total_runs': int(predicted_runs),
                'runs_range': f"{int(predicted_runs - 30)} - {int(predicted_runs + 30)}"
            }
        
        # Add timeline summary
        predictions['timeline_summary'] = {
            'total_periods': len(timeline),
            'ascendant_favored_periods': sum(1 for p in timeline if p['kp_favorability_score'] > 0),
            'descendant_favored_periods': sum(1 for p in timeline if p['kp_favorability_score'] < 0),
            'high_scoring_periods': sum(1 for p in timeline if p['high_scoring_probability'] > 0.6),
            'collapse_risk_periods': sum(1 for p in timeline if p['collapse_probability'] > 0.5),
            'key_planetary_rulers': {
                'primary_sub_lord': first_period['current_sub_lord'],
                'primary_sub_sub_lord': first_period['current_sub_sub_lord'],
                'primary_star_lord': first_period['current_star_lord']
            }
        }
        
        return {
            'match_info': {
                'team1': team1,
                'team2': team2,
                'match_time': match_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'predictions': predictions,
            'timeline': timeline[:6]  # First 6 periods (1.5 hours)
        }
    
    def close(self):
        """Close connections"""
        self.validator.close()
        self.conn.close()

    def generate_astrological_periods_for_match(self, match_id, hours=5):
        """
        Generate astrological periods for a match based on sub lord changes
        Returns periods with their KP characteristics for training
        """
        
        # Get delivery data to determine match timing
        deliveries_df = pd.read_sql_query(
            "SELECT timestamp FROM deliveries WHERE match_id = ? AND timestamp IS NOT NULL ORDER BY timestamp",
            self.conn, params=[match_id]
        )
        
        if deliveries_df.empty:
            return []
        
        # Convert timestamps
        deliveries_df['timestamp'] = pd.to_datetime(deliveries_df['timestamp'])
        
        # Determine timeline boundaries
        match_start = deliveries_df['timestamp'].min() - timedelta(minutes=30)
        timeline_end = match_start + timedelta(hours=hours)
        
        # Generate astrological periods
        periods = []
        current_time = match_start
        period_count = 0
        max_periods = 15  # Reasonable limit for training
        
        while current_time < timeline_end and period_count < max_periods:
            # Generate chart for current time
            current_chart = generate_kp_chart(
                current_time.to_pydatetime(), 
                19.0760, 72.8777,  # Default Mumbai coordinates
                self.nakshatra_df
            )
            
            if 'error' in current_chart:
                # If chart generation fails, fall back to 45-minute periods
                next_time = current_time + timedelta(minutes=45)
            else:
                # Find next sub lord change
                next_time = self.find_next_sub_lord_change(current_time, current_chart)
                
                # Ensure minimum period of 15 minutes and maximum of 90 minutes
                min_next_time = current_time + timedelta(minutes=15)
                max_next_time = current_time + timedelta(minutes=90)
                
                if next_time < min_next_time:
                    next_time = min_next_time
                elif next_time > max_next_time:
                    next_time = max_next_time
            
            # Don't exceed end time
            if next_time > timeline_end:
                next_time = timeline_end
            
            # Evaluate favorability for this period
            if 'error' not in current_chart:
                # Create dummy deliveries dataframe for favorability calculation
                dummy_deliveries = pd.DataFrame({
                    'timestamp': [current_time],
                    'runs_off_bat': [0],
                    'wicket_type': [None]
                })
                favorability = evaluate_favorability(current_chart, dummy_deliveries)
            else:
                favorability = {'final_score': 0}
            
            periods.append({
                'match_id': match_id,
                'period_num': period_count + 1,
                'start_time': current_time,
                'end_time': next_time,
                'duration_minutes': int((next_time - current_time).total_seconds() / 60),
                'chart': current_chart,
                'favorability': favorability,
                'kp_score': favorability.get('final_score', 0),
                'sub_lord': current_chart.get('moon_sub_lord', 'Unknown'),
                'sub_sub_lord': current_chart.get('moon_sub_sub_lord', 'Unknown'),
                'nakshatra': current_chart.get('moon_nakshatra', 'Unknown'),
                'pada': current_chart.get('moon_pada', 0),
                'ascendant_degree': current_chart.get('ascendant_degree', 0)
            })
            
            current_time = next_time
            period_count += 1
        
        return periods
    
    def find_next_sub_lord_change(self, current_time, current_chart):
        """
        Find the next time when sub lord or sub-sub lord changes
        Uses moon's movement to calculate transition times
        """
        
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
            
            # Find current sub lord boundaries in nakshatra data
            current_row = self.nakshatra_df[
                (self.nakshatra_df['Start_Degree'] <= moon_long) & 
                (self.nakshatra_df['End_Degree'] > moon_long)
            ]
            
            if current_row.empty:
                # Handle edge case for 360 degrees
                if moon_long >= 359.99:
                    current_row = self.nakshatra_df.iloc[-1:]
                else:
                    # Fallback to 45-minute period
                    return current_time + timedelta(minutes=45)
            
            if current_row.empty:
                return current_time + timedelta(minutes=45)
            
            # Get the end degree of current sub lord
            end_degree = current_row.iloc[0]['End_Degree']
            
            # Calculate degrees to travel to reach next sub lord
            if end_degree < moon_long:  # Handle 360 -> 0 degree crossover
                degrees_to_travel = (360 - moon_long) + end_degree
            else:
                degrees_to_travel = end_degree - moon_long
            
            # Add small buffer to ensure we cross the boundary
            degrees_to_travel += 0.001
            
            # Calculate time to reach next sub lord
            if moon_speed_deg_per_minute > 0:
                minutes_to_change = degrees_to_travel / moon_speed_deg_per_minute
                next_change_time = current_time + timedelta(minutes=minutes_to_change)
            else:
                # Fallback if moon speed calculation fails
                next_change_time = current_time + timedelta(minutes=45)
            
            return next_change_time
            
        except Exception as e:
            print(f"Error calculating sub lord change: {str(e)}")
            # Fallback to 45-minute period
            return current_time + timedelta(minutes=45)

def main():
    """Main function to run enhanced training"""
    
    trainer = KPEnhancedTrainer()
    
    try:
        # Step 1: Validate and correct training data
        corrected_data = trainer.validate_and_correct_training_data(limit=100)
        
        if corrected_data.empty:
            print("No corrected data available for training")
            return
        
        # Step 2: Engineer enhanced features
        features_df = trainer.engineer_enhanced_features(corrected_data)
        
        if features_df.empty:
            print("No features engineered")
            return
        
        # Step 3: Train enhanced models
        trainer.train_enhanced_models(features_df)
        
        # Step 4: Generate training report
        trainer.generate_training_report()
        
        # Step 5: Example prediction
        print("\nExample prediction:")
        sample_prediction = trainer.predict_match(
            "Mumbai Indians", 
            "Chennai Super Kings",
            datetime.now().replace(hour=19, minute=30, second=0, microsecond=0)
        )
        
        if sample_prediction:
            print(f"Winner: {sample_prediction['predictions'].get('winner', {}).get('predicted_winner', 'Unknown')}")
            print(f"Confidence: {sample_prediction['predictions'].get('winner', {}).get('confidence', 0):.3f}")
            print(f"Match Nature: {sample_prediction['predictions'].get('scoring', {}).get('predicted_nature', 'Unknown')}")
    
    except Exception as e:
        print(f"Error in enhanced training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        trainer.close()

if __name__ == "__main__":
    main() 