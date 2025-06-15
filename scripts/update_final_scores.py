#!/usr/bin/env python3
"""
Script to recalculate and update final_score values in astrological_predictions table
using the corrected authentic KP methodology.

This fixes the issue where old records used broken logic (static scores ignoring muhurta)
and ensures all records use the same authentic KP analysis for ML training.
"""

import sqlite3
import pandas as pd
import sys
import os
from datetime import datetime
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability

class FinalScoreUpdater:
    def __init__(self, db_path="training_analysis/cricket_predictions.db"):
        self.db_path = db_path
        self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
        
    def get_match_coordinates(self, match_id):
        """Get coordinates for a match from the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT latitude, longitude 
            FROM matches 
            WHERE match_id = ?
            """
            result = pd.read_sql_query(query, conn, params=(match_id,))
            if not result.empty:
                return result.iloc[0]['latitude'], result.iloc[0]['longitude']
            return None, None
        finally:
            conn.close()
    
    def get_match_start_time(self, match_id):
        """Get match start datetime from the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT start_datetime 
            FROM matches 
            WHERE match_id = ?
            """
            result = pd.read_sql_query(query, conn, params=(match_id,))
            if not result.empty:
                row = result.iloc[0]
                # Parse the start_datetime directly
                return pd.to_datetime(row['start_datetime'])
            return None
        finally:
            conn.close()
    
    def get_records_to_update(self, batch_size=1000):
        """Get all records that need final_score updates."""
        query = """
        SELECT ap.id, ap.match_id, d.timestamp, ap.moon_sl, ap.moon_sub, ap.moon_ssl
        FROM astrological_predictions ap
        JOIN deliveries d ON ap.delivery_id = d.id
        ORDER BY ap.id
        """
        # Return generator that creates fresh connections for each chunk
        def chunk_generator():
            offset = 0
            while True:
                with sqlite3.connect(self.db_path) as conn:
                    chunk_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                    chunk_df = pd.read_sql_query(chunk_query, conn)
                    if chunk_df.empty:
                        break
                    yield chunk_df
                    offset += batch_size
        
        return chunk_generator()
    
    def recalculate_final_score(self, match_id, delivery_timestamp, moon_sl, moon_sub, moon_ssl):
        """Recalculate final_score using authentic KP methodology."""
        try:
            # Get match coordinates and start time
            lat, lon = self.get_match_coordinates(match_id)
            if lat is None or lon is None:
                return None, "Missing coordinates"
            
            match_start_dt = self.get_match_start_time(match_id)
            if match_start_dt is None:
                return None, "Missing match start time"
            
            # Parse delivery timestamp
            if isinstance(delivery_timestamp, str):
                delivery_dt = pd.to_datetime(delivery_timestamp)
            else:
                delivery_dt = delivery_timestamp
            
            # Generate muhurta chart (match start time)
            muhurta_chart = generate_kp_chart(match_start_dt, lat, lon, self.nakshatra_df)
            if "error" in muhurta_chart:
                return None, f"Muhurta chart error: {muhurta_chart['error']}"
            
            # Generate current chart (delivery time)
            current_chart = generate_kp_chart(delivery_dt, lat, lon, self.nakshatra_df)
            if "error" in current_chart:
                return None, f"Current chart error: {current_chart['error']}"
            
            # Calculate favorability using authentic KP methodology
            favorability_data = evaluate_favorability(muhurta_chart, current_chart, self.nakshatra_df)
            
            final_score = favorability_data.get("final_score")
            if final_score is None:
                return None, "Failed to calculate final_score"
            
            return final_score, "Success"
            
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def update_batch(self, batch_df):
        """Update a batch of records with recalculated final_scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        errors = []
        
        try:
            for _, row in batch_df.iterrows():
                final_score, status = self.recalculate_final_score(
                    row['match_id'], 
                    row['timestamp'],
                    row['moon_sl'],
                    row['moon_sub'], 
                    row['moon_ssl']
                )
                
                if final_score is not None:
                    updates.append((final_score, row['id']))
                else:
                    errors.append((row['id'], status))
            
            # Batch update successful calculations
            if updates:
                cursor.executemany(
                    "UPDATE astrological_predictions SET success_score = ? WHERE id = ?",
                    updates
                )
                conn.commit()
            
            return len(updates), errors
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def update_all_final_scores(self, batch_size=1000):
        """Update all final_score values in the database."""
        print("Starting final_score recalculation using authentic KP methodology...")
        
        # Get total count for progress tracking
        with sqlite3.connect(self.db_path) as conn:
            total_count = pd.read_sql_query("SELECT COUNT(*) as count FROM astrological_predictions", conn).iloc[0]['count']
        
        print(f"Total records to process: {total_count:,}")
        
        total_updated = 0
        total_errors = 0
        
        # Process in batches
        for batch_num, batch_df in enumerate(self.get_records_to_update(batch_size)):
            print(f"\nProcessing batch {batch_num + 1} ({len(batch_df)} records)...")
            
            try:
                updated_count, errors = self.update_batch(batch_df)
                total_updated += updated_count
                total_errors += len(errors)
                
                print(f"Batch {batch_num + 1}: {updated_count} updated, {len(errors)} errors")
                
                if errors and len(errors) <= 5:  # Show first few errors
                    for record_id, error_msg in errors[:5]:
                        print(f"  Error ID {record_id}: {error_msg}")
                
                # Progress update
                processed = (batch_num + 1) * batch_size
                progress = min(100, (processed / total_count) * 100)
                print(f"Progress: {progress:.1f}% ({total_updated:,} updated, {total_errors:,} errors)")
                
            except Exception as e:
                print(f"Error processing batch {batch_num + 1}: {e}")
                continue
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total records processed: {total_count:,}")
        print(f"Successfully updated: {total_updated:,}")
        print(f"Errors: {total_errors:,}")
        print(f"Success rate: {(total_updated/total_count)*100:.1f}%")
        
        return total_updated, total_errors

def main():
    parser = argparse.ArgumentParser(description="Update final_score values using authentic KP methodology")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing (default: 1000)")
    parser.add_argument("--db-path", type=str, default="training_analysis/cricket_predictions.db", help="Path to database file")
    
    args = parser.parse_args()
    
    updater = FinalScoreUpdater(args.db_path)
    
    try:
        updated, errors = updater.update_all_final_scores(args.batch_size)
        
        if errors == 0:
            print("\n✅ All final_score values successfully updated with authentic KP methodology!")
        else:
            print(f"\n⚠️  Update completed with {errors:,} errors. Check logs above for details.")
            
    except KeyboardInterrupt:
        print("\n❌ Update interrupted by user")
    except Exception as e:
        print(f"\n❌ Update failed: {e}")

if __name__ == "__main__":
    main() 