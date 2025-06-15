#!/usr/bin/env python3
"""
Verification script to check the current state of final_score values
and demonstrate the difference between old and new KP methodology.
"""

import sqlite3
import pandas as pd
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability

def check_database_status(db_path="training_analysis/cricket_predictions.db"):
    """Check the current status of the astrological_predictions table."""
    conn = sqlite3.connect(db_path)
    
    try:
        # Get basic stats
        stats_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(success_score) as records_with_score,
            MIN(success_score) as min_score,
            MAX(success_score) as max_score,
            AVG(success_score) as avg_score,
            COUNT(DISTINCT match_id) as unique_matches
        FROM astrological_predictions
        """
        
        stats = pd.read_sql_query(stats_query, conn)
        
        print("=== DATABASE STATUS ===")
        print(f"Total records: {stats.iloc[0]['total_records']:,}")
        print(f"Records with success_score: {stats.iloc[0]['records_with_score']:,}")
        print(f"Unique matches: {stats.iloc[0]['unique_matches']:,}")
        
        if stats.iloc[0]['records_with_score'] > 0:
            print(f"Score range: {stats.iloc[0]['min_score']:.2f} to {stats.iloc[0]['max_score']:.2f}")
            print(f"Average score: {stats.iloc[0]['avg_score']:.2f}")
        else:
            print("Score range: No data")
            print("Average score: No data")
        
        # Check for potential old static scores
        static_score_query = """
        SELECT success_score, COUNT(*) as count
        FROM astrological_predictions 
        WHERE success_score IS NOT NULL
        GROUP BY ROUND(success_score, 2)
        HAVING COUNT(*) > 100
        ORDER BY count DESC
        LIMIT 10
        """
        
        static_scores = pd.read_sql_query(static_score_query, conn)
        
        if not static_scores.empty:
            print("\n=== POTENTIAL STATIC SCORES (>100 occurrences) ===")
            for _, row in static_scores.iterrows():
                print(f"Score {row['success_score']:.2f}: {row['count']:,} records")
        
        return stats.iloc[0].to_dict()
        
    finally:
        conn.close()

def compare_methodologies_sample(db_path="training_analysis/cricket_predictions.db", sample_size=5):
    """Compare old vs new methodology on a sample of records."""
    conn = sqlite3.connect(db_path)
    nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
    
    try:
        # Get a sample of records
        sample_query = """
        SELECT ap.*, m.latitude, m.longitude, m.date, m.start_time
        FROM astrological_predictions ap
        JOIN matches m ON ap.match_id = m.match_id
        WHERE ap.success_score IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        sample_df = pd.read_sql_query(sample_query, conn, params=(sample_size,))
        
        print(f"\n=== METHODOLOGY COMPARISON (Sample of {len(sample_df)} records) ===")
        
        for _, row in sample_df.iterrows():
            try:
                # Parse match start time
                match_date = pd.to_datetime(row['date']).date()
                start_time = pd.to_datetime(row['start_time']).time()
                match_start_dt = datetime.combine(match_date, start_time)
                
                # Parse delivery time
                delivery_dt = pd.to_datetime(row['delivery_datetime'])
                
                # OLD METHOD: Only delivery time chart
                old_chart = generate_kp_chart(delivery_dt, row['latitude'], row['longitude'], nakshatra_df)
                if "error" not in old_chart:
                    old_score = evaluate_favorability(old_chart, None, nakshatra_df)['final_score']
                else:
                    old_score = None
                
                # NEW METHOD: Muhurta + delivery time
                muhurta_chart = generate_kp_chart(match_start_dt, row['latitude'], row['longitude'], nakshatra_df)
                current_chart = generate_kp_chart(delivery_dt, row['latitude'], row['longitude'], nakshatra_df)
                
                if "error" not in muhurta_chart and "error" not in current_chart:
                    new_score = evaluate_favorability(muhurta_chart, current_chart, nakshatra_df)['final_score']
                else:
                    new_score = None
                
                print(f"\nRecord ID {row['id']} (Match {row['match_id']}):")
                print(f"  Delivery time: {delivery_dt.strftime('%H:%M:%S')}")
                print(f"  Match start: {match_start_dt.strftime('%H:%M:%S')}")
                print(f"  Current DB score: {row['success_score']:.2f}")
                print(f"  Old method score: {old_score:.2f if old_score else 'ERROR'}")
                print(f"  New method score: {new_score:.2f if new_score else 'ERROR'}")
                
                if old_score and new_score:
                    diff = abs(new_score - old_score)
                    print(f"  Difference: {diff:.2f} ({'SIGNIFICANT' if diff > 1.0 else 'minor'})")
                
            except Exception as e:
                print(f"  Error processing record {row['id']}: {e}")
                
    finally:
        conn.close()

def main():
    print("KP Cricket Predictor - Score Verification Tool")
    print("=" * 50)
    
    # Check if database exists
    db_path = "training_analysis/cricket_predictions.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found!")
        return
    
    # Check database status
    stats = check_database_status()
    
    # Compare methodologies on sample
    if stats['records_with_score'] > 0:
        compare_methodologies_sample()
    else:
        print("\n⚠️  No records with final_score found. Run astro_processor first.")
    
    print(f"\n=== RECOMMENDATIONS ===")
    if stats['records_with_score'] > 0:
        print("✅ Database has astrological predictions")
        print("🔄 Run 'python -m scripts.update_final_scores' to update all scores with authentic KP methodology")
    else:
        print("⏳ Wait for astro_processor to complete, then run the score updater")

if __name__ == "__main__":
    main() 