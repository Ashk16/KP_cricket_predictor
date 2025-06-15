#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Cricket Match Timeline Predictor using KP Astrology
Generates complete astrological timeline predictions with match dynamics
Uses ONLY pre-match data: Time + Location + KP calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import os
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder
import pytz

# Suppress warnings
warnings.filterwarnings('ignore')

# Import KP calculation functions
try:
    from .chart_generator import generate_kp_chart
except ImportError:
    try:
        from scripts.chart_generator import generate_kp_chart
    except ImportError:
        print("⚠️ Warning: chart_generator module not found. KP calculations may not work properly.")

class LiveCricketTimelinePredictor:
    """Live cricket match timeline predictor using KP astrology with multi-target predictions"""
    
    def __init__(self, model_path=None):
        """Initialize the predictor with trained multi-target models"""
        self.model_data = None
        self.feature_names = None
        self.models = {}  # Dictionary to store all target models
        
        # Load the latest model if no path specified
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️ No trained model found. Please train a model first.")
    
    def _find_latest_model(self):
        """Find the latest trained multi-target model"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            return None
        
        # Look for multi-target models first, then fall back to predictive models
        model_files = [f for f in os.listdir(models_dir) if f.startswith('multi_target_predictive_results_') and f.endswith('.pkl')]
        
        if not model_files:
            model_files = [f for f in os.listdir(models_dir) if f.startswith('predictive_ml_results_') and f.endswith('.pkl')]
        
        if not model_files:
            return None
        
        # Sort by timestamp in filename
        model_files.sort(reverse=True)
        latest_model = os.path.join(models_dir, model_files[0])
        
        print(f"📦 Loading latest model: {model_files[0]}")
        return latest_model
    
    def _load_model(self, model_path):
        """Load the trained multi-target models and metadata"""
        try:
            self.model_data = joblib.load(model_path)
            self.feature_names = self.model_data['feature_names']
            
            # Load all target models from the new structure
            all_models = self.model_data['multi_target_models']
            
            # Map target names from new format to expected format
            target_mapping = {
                'kp_prediction': 'binary_target',
                'high_scoring': 'high_scoring_target', 
                'wicket_pressure': 'wicket_pressure_target',
                'momentum': 'momentum_binary_target'
            }
            
            for new_target, old_target in target_mapping.items():
                if new_target in all_models:
                    target_data = all_models[new_target]
                    target_models = target_data['models']
                    
                    # Get best model for this target (highest test accuracy)
                    best_model_name = max(target_models.keys(), key=lambda k: target_models[k]['test_accuracy'])
                    best_model_info = target_models[best_model_name]
                    
                    self.models[old_target] = {
                        'model': best_model_info['model'],
                        'scaler': best_model_info['scaler'],
                        'accuracy': best_model_info['test_accuracy'],
                        'algorithm': best_model_name
                    }
            
            print(f"✅ Multi-target models loaded successfully!")
            print(f"🔮 Features: {len(self.feature_names)}")
            print(f"🎯 Loaded {len(self.models)} target models:")
            
            for target, info in self.models.items():
                target_name = target.replace('_target', '').replace('_', ' ').title()
                print(f"  • {target_name}: {info['algorithm']} ({info['accuracy']:.3f})")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model_data = None
    
    def calculate_kp_features(self, match_datetime, latitude, longitude):
        """Calculate KP astrological features for a match"""
        try:
            # Generate KP chart for the match time and location
            kp_data = generate_kp_chart(match_datetime, latitude, longitude)
            
            if not kp_data or 'error' in kp_data:
                print(f"❌ Failed to generate KP chart: {kp_data.get('error', 'Unknown error')}")
                return None
            
            # Calculate KP prediction score and strength based on chart data
            # This is a simplified calculation - in practice, you'd use more complex KP rules
            moon_long = kp_data.get('moon_longitude', 0)
            asc_deg = kp_data.get('ascendant_degree', 0)
            
            # Simple KP prediction logic (can be enhanced)
            kp_score = (moon_long - asc_deg) % 360
            if kp_score > 180:
                kp_score = kp_score - 360
            
            kp_strength = min(10, abs(kp_score) / 10)  # Scale to 0-10
            kp_favors_ascendant = kp_score > 0
            
            # Extract KP prediction features
            features = {
                'kp_prediction_score': kp_score,
                'kp_prediction_strength': kp_strength,
                'kp_favors_ascendant': int(kp_favors_ascendant),
                'moon_sl': kp_data.get('moon_star_lord', 'Unknown'),
                'moon_sub': kp_data.get('moon_sub_lord', 'Unknown'),
                'moon_ssl': kp_data.get('moon_sub_sub_lord', 'Unknown'),
                'start_time': match_datetime
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Error calculating KP features: {e}")
            return None
    
    def engineer_prediction_features(self, kp_features):
        """Engineer features for prediction (same as training)"""
        try:
            features = pd.DataFrame([kp_features])
            
            # KP Derived Features
            features['kp_strength_squared'] = features['kp_prediction_strength'] ** 2
            features['kp_strength_cubed'] = features['kp_prediction_strength'] ** 3
            features['kp_score_abs'] = np.abs(features['kp_prediction_score'])
            features['kp_score_squared'] = features['kp_prediction_score'] ** 2
            
            # Time-based Features
            dt = pd.to_datetime(features['start_time']).iloc[0]
            features['start_hour'] = dt.hour
            features['start_minute'] = dt.minute
            features['day_of_week'] = dt.dayofweek
            features['month'] = dt.month
            
            # Time categories
            features['is_morning'] = (features['start_hour'] < 12).astype(int)
            features['is_afternoon'] = ((features['start_hour'] >= 12) & (features['start_hour'] < 18)).astype(int)
            features['is_evening'] = (features['start_hour'] >= 18).astype(int)
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            
            # Planetary Lord Features (encode consistently)
            planetary_map = {
                'Sun': 0, 'Moon': 1, 'Mars': 2, 'Mercury': 3, 'Jupiter': 4, 
                'Venus': 5, 'Saturn': 6, 'Rahu': 7, 'Ketu': 8, 'Unknown': 9
            }
            
            features['moon_sl_encoded'] = features['moon_sl'].map(planetary_map).fillna(9)
            features['moon_sub_encoded'] = features['moon_sub'].map(planetary_map).fillna(9)
            features['moon_ssl_encoded'] = features['moon_ssl'].map(planetary_map).fillna(9)
            
            # KP-Time Interaction Features
            features['kp_hour_interaction'] = features['kp_prediction_strength'] * features['start_hour']
            features['kp_minute_interaction'] = features['kp_prediction_strength'] * features['start_minute']
            features['kp_dayofweek_interaction'] = features['kp_prediction_strength'] * features['day_of_week']
            
            # Advanced KP Features
            features['kp_confidence'] = np.abs(features['kp_prediction_score']) * features['kp_prediction_strength']
            features['kp_directional'] = features['kp_prediction_score'] * features['kp_favors_ascendant']
            
            # Select only the features used in training
            feature_matrix = features[self.feature_names].fillna(0)
            
            return feature_matrix
            
        except Exception as e:
            print(f"❌ Error engineering features: {e}")
            return None
    
    def generate_astrological_periods(self, match_datetime, duration_hours=6):
        """Generate astrological periods for the match timeline"""
        periods = []
        current_time = match_datetime
        period_num = 1
        
        # Generate periods every 15-90 minutes based on sub lord changes
        while (current_time - match_datetime).total_seconds() < duration_hours * 3600:
            # Calculate period duration (15-90 minutes, varying based on planetary movements)
            base_duration = 30  # Base 30 minutes
            variation = np.random.randint(-15, 60)  # Random variation
            duration_minutes = max(15, min(90, base_duration + variation))
            
            period_end = current_time + timedelta(minutes=duration_minutes)
            
            period = {
                'period_num': period_num,
                'start_time': current_time,
                'end_time': period_end,
                'duration_minutes': duration_minutes,
                'period_type': self._get_period_type(current_time)
            }
            
            periods.append(period)
            current_time = period_end
            period_num += 1
        
        return periods
    
    def _get_period_type(self, period_time):
        """Determine period type based on time"""
        hour = period_time.hour
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Night"
    
    def predict_period(self, period_time, latitude, longitude):
        """Make multi-target predictions for a specific period"""
        
        if not self.models:
            return {"error": "No models loaded"}
        
        # Calculate KP features for this period
        kp_features = self.calculate_kp_features(period_time, latitude, longitude)
        if kp_features is None:
            return {"error": "Failed to calculate KP features"}
        
        # Engineer prediction features
        feature_matrix = self.engineer_prediction_features(kp_features)
        if feature_matrix is None:
            return {"error": "Failed to engineer features"}
        
        predictions = {}
        
        # Make predictions for each target
        for target, model_info in self.models.items():
            try:
                # Scale features if scaler exists
                if model_info['scaler'] is not None:
                    scaled_features = model_info['scaler'].transform(feature_matrix)
                else:
                    scaled_features = feature_matrix
                
                # Get prediction and probability
                prediction = model_info['model'].predict(scaled_features)[0]
                
                # Get probability if available
                if hasattr(model_info['model'], 'predict_proba'):
                    probabilities = model_info['model'].predict_proba(scaled_features)[0]
                    confidence = max(probabilities)
                    positive_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                else:
                    confidence = 0.5
                    positive_prob = 0.5
                
                predictions[target] = {
                    'prediction': int(prediction),
                    'probability': float(positive_prob),
                    'confidence': float(confidence)
                }
                
            except Exception as e:
                print(f"⚠️ Error predicting {target}: {e}")
                predictions[target] = {
                    'prediction': 0,
                    'probability': 0.5,
                    'confidence': 0.5
                }
        
        return predictions
    
    def predict_timeline(self, match_datetime, latitude, longitude, team1_name="Team 1", team2_name="Team 2", duration_hours=6):
        """Generate complete astrological timeline predictions for a cricket match"""
        
        if not self.models:
            return {"error": "No models loaded"}
        
        print(f"🔮 Generating astrological timeline for {team1_name} vs {team2_name}")
        print(f"📅 Match time: {match_datetime}")
        print(f"📍 Location: {latitude:.4f}, {longitude:.4f}")
        print(f"⏱️ Duration: {duration_hours} hours")
        print()
        
        # Generate astrological periods
        print("🌟 Generating astrological periods...")
        periods = self.generate_astrological_periods(match_datetime, duration_hours)
        
        timeline_predictions = []
        
        for i, period in enumerate(periods):
            print(f"🔮 Predicting period {period['period_num']}/{len(periods)}...")
            
            # Make predictions for this period
            predictions = self.predict_period(period['start_time'], latitude, longitude)
            
            if 'error' in predictions:
                continue
            
            # Calculate KP features for additional context
            kp_features = self.calculate_kp_features(period['start_time'], latitude, longitude)
            
            # Create period prediction
            period_prediction = {
                'period_info': {
                    'period_num': period['period_num'],
                    'start_time': period['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': period['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_minutes': period['duration_minutes'],
                    'period_type': period['period_type'],
                    'moon_nakshatra': kp_features.get('moon_sl', 'Unknown'),
                    'moon_sub_lord': kp_features.get('moon_sub', 'Unknown'),
                    'moon_sub_sub_lord': kp_features.get('moon_ssl', 'Unknown')
                },
                'team_favorability': {
                    'ascendant_favored': predictions['binary_target']['prediction'] == 1,
                    'favorability_strength': predictions['binary_target']['confidence'],
                    'kp_score': kp_features.get('kp_prediction_score', 0),
                    'confidence_level': self._get_confidence_level(predictions['binary_target']['confidence'])
                },
                'match_dynamics': {
                    'high_scoring_probability': predictions['high_scoring_target']['probability'] * 100,
                    'wicket_pressure_probability': predictions['wicket_pressure_target']['probability'] * 100,
                    'momentum_shift_probability': predictions['momentum_binary_target']['probability'] * 100,
                    'collapse_probability': (1 - predictions['high_scoring_target']['probability']) * predictions['wicket_pressure_target']['probability'] * 100
                },
                'astrological_notes': self._generate_astrological_notes(kp_features, predictions)
            }
            
            timeline_predictions.append(period_prediction)
        
        return {
            'match_info': {
                'team1': team1_name,
                'team2': team2_name,
                'match_datetime': match_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'location': f"{latitude:.4f}, {longitude:.4f}",
                'total_periods': len(timeline_predictions)
            },
            'timeline': timeline_predictions
        }
    
    def _get_confidence_level(self, confidence):
        """Convert confidence score to descriptive level"""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        else:
            return "Low"
    
    def _generate_astrological_notes(self, kp_features, predictions):
        """Generate astrological insights and notes"""
        notes = []
        
        # KP Score insights
        kp_score = kp_features.get('kp_prediction_score', 0)
        if abs(kp_score) > 10:
            notes.append(f"Strong KP influence detected (score: {kp_score:.1f})")
        
        # Planetary lord insights
        moon_sl = kp_features.get('moon_sl', 'Unknown')
        if moon_sl in ['Mars', 'Ketu']:
            notes.append("Aggressive planetary influence - expect dynamic play")
        elif moon_sl in ['Venus', 'Moon']:
            notes.append("Harmonious planetary influence - favorable for batting")
        elif moon_sl in ['Saturn', 'Rahu']:
            notes.append("Restrictive planetary influence - challenging conditions")
        
        # Match dynamics insights
        if predictions['wicket_pressure_target']['probability'] > 0.7:
            notes.append("High wicket pressure period - bowlers favored")
        
        if predictions['high_scoring_target']['probability'] > 0.6:
            notes.append("Favorable for high scoring - batsmen advantage")
        
        return notes
    
    def print_timeline_summary(self, timeline_result):
        """Print a formatted timeline prediction summary"""
        if 'error' in timeline_result:
            print(f"❌ Timeline Error: {timeline_result['error']}")
            return
        
        match_info = timeline_result['match_info']
        timeline = timeline_result['timeline']
        
        print("🔮 ASTROLOGICAL TIMELINE PREDICTION")
        print("=" * 80)
        print(f"🏏 Match: {match_info['team1']} vs {match_info['team2']}")
        print(f"📅 Date & Time: {match_info['match_datetime']}")
        print(f"📍 Location: {match_info['location']}")
        print(f"⏱️ Total Periods: {match_info['total_periods']}")
        print("=" * 80)
        
        for i, period in enumerate(timeline):
            period_info = period['period_info']
            favorability = period['team_favorability']
            dynamics = period['match_dynamics']
            notes = period['astrological_notes']
            
            print(f"\n🌟 PERIOD {period_info['period_num']}")
            print(f"⏰ Time: {period_info['start_time']} - {period_info['end_time']}")
            print(f"⏱️ Duration: {period_info['duration_minutes']} minutes ({period_info['period_type']})")
            print(f"🌙 Moon: {period_info['moon_nakshatra']} → {period_info['moon_sub_lord']} → {period_info['moon_sub_sub_lord']}")
            
            print(f"\n🎯 TEAM FAVORABILITY:")
            favored_team = match_info['team1'] if favorability['ascendant_favored'] else match_info['team2']
            print(f"  Favored: {favored_team}")
            print(f"  Strength: {favorability['confidence_level']} ({favorability['favorability_strength']:.1%})")
            print(f"  KP Score: {favorability['kp_score']:.2f}")
            
            print(f"\n📊 MATCH DYNAMICS:")
            print(f"  High Scoring: {dynamics['high_scoring_probability']:.1f}%")
            print(f"  Wicket Pressure: {dynamics['wicket_pressure_probability']:.1f}%")
            print(f"  Momentum Shift: {dynamics['momentum_shift_probability']:.1f}%")
            print(f"  Collapse Risk: {dynamics['collapse_probability']:.1f}%")
            
            if notes:
                print(f"\n🔮 ASTROLOGICAL NOTES:")
                for note in notes:
                    print(f"  • {note}")
            
            if i < len(timeline) - 1:
                print("-" * 60)
        
        print("\n" + "=" * 80)

def get_venue_coordinates(venue_name):
    """Get coordinates for common cricket venues"""
    venues = {
        # India
        'wankhede stadium': (19.0448, 72.8258),
        'eden gardens': (22.5645, 88.3433),
        'chinnaswamy stadium': (12.9784, 77.5996),
        'feroz shah kotla': (28.6358, 77.2424),
        'chepauk stadium': (13.0627, 80.2792),
        'rajiv gandhi stadium': (17.4065, 78.4691),
        'sawai mansingh stadium': (26.8983, 75.8081),
        'brabourne stadium': (18.9388, 72.8258),
        
        # England
        'lords': (51.5294, -0.1716),
        'oval': (51.4816, -0.1150),
        'old trafford': (53.4568, -2.2901),
        'edgbaston': (52.4558, -1.9025),
        'headingley': (53.8175, -1.5822),
        'trent bridge': (52.9373, -1.1324),
        
        # Australia
        'mcg': (-37.8200, 144.9834),
        'scg': (-33.8915, 151.2244),
        'gabba': (-27.4848, 153.0389),
        'adelaide oval': (-34.9156, 138.5959),
        'waca': (-31.9609, 115.7759),
        'marvel stadium': (-37.8164, 144.9475),
        
        # Other
        'dubai international stadium': (25.2225, 55.3094),
        'sharjah cricket stadium': (25.3375, 55.3847),
        'gaddafi stadium': (31.5204, 74.3587),
        'national stadium karachi': (24.8607, 67.0011)
    }
    
    venue_lower = venue_name.lower()
    for venue, coords in venues.items():
        if venue in venue_lower:
            return coords
    
    return None

def quick_timeline_example():
    """Quick timeline prediction example with sample data"""
    print("🔮 QUICK TIMELINE PREDICTION EXAMPLE")
    print("=" * 50)
    
    predictor = LiveCricketTimelinePredictor()
    
    if not predictor.models:
        print("❌ No trained models available.")
        return
    
    # Sample match: India vs Australia at Wankhede Stadium
    team1 = "India"
    team2 = "Australia"
    
    # Tomorrow at 2:30 PM IST
    from datetime import timedelta
    tomorrow = datetime.now() + timedelta(days=1)
    match_time = tomorrow.replace(hour=14, minute=30, second=0, microsecond=0)
    
    # Add IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    match_dt = ist.localize(match_time)
    
    # Wankhede Stadium coordinates
    latitude, longitude = 19.0448, 72.8258
    
    print(f"🏟️ Sample Match: {team1} vs {team2}")
    print(f"📍 Venue: Wankhede Stadium, Mumbai")
    print(f"📅 Time: {match_dt}")
    print()
    
    # Generate timeline prediction
    timeline_result = predictor.predict_timeline(match_dt, latitude, longitude, team1, team2, duration_hours=4)
    
    # Display result
    predictor.print_timeline_summary(timeline_result)

def main():
    """Main function"""
    print("🔮 KP CRICKET PREDICTOR - LIVE TIMELINE INTERFACE")
    print("=" * 70)
    print("Choose an option:")
    print("1. Quick timeline example")
    print("2. Exit")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        quick_timeline_example()
    elif choice == "2":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 