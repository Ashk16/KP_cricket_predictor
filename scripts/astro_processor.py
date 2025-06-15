#!/usr/bin/env python3
"""
Fast Astrological Processor for KP Cricket Predictions
Optimized for high-performance processing of large datasets
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import swisseph as swe
import argparse
import json
from collections import defaultdict

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Removed unnecessary imports - this processor is self-contained

class FastAstroProcessor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
        
        # Cache for coordinates to avoid repeated DataFrame lookups
        self.coords_cache = {}
        
        # Pre-calculate sign boundaries for faster lookups
        self.sign_names = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                          'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
        
        # Pre-calculate nakshatra boundaries using correct column names
        self.nakshatra_boundaries = []
        for _, row in self.nakshatra_df.iterrows():
            self.nakshatra_boundaries.append((row['Start_Degree'], row['End_Degree'], row['Nakshatra']))

    def process_all_deliveries_fast(self, batch_size=5000):
        """Fast processing with optimized batch operations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('''
            SELECT COUNT(*) FROM deliveries d
            LEFT JOIN astrological_predictions ap ON d.id = ap.delivery_id
            WHERE ap.delivery_id IS NULL
        ''')
        total_deliveries = cursor.fetchone()[0]
        
        if total_deliveries == 0:
            print("All deliveries already processed.")
            conn.close()
            return
        
        print(f"Fast processing {total_deliveries} unprocessed deliveries...")
        
        # Load coordinates into cache
        coords_df = pd.read_sql_query("SELECT match_id, latitude, longitude FROM matches", conn)
        for _, row in coords_df.iterrows():
            self.coords_cache[row['match_id']] = (
                float(row['latitude']) if row['latitude'] is not None else 19.0760,
                float(row['longitude']) if row['longitude'] is not None else 72.8777
            )
        
        processed = 0
        errors = 0
        
        while processed < total_deliveries:
            # Get large batch
            cursor.execute('''
                SELECT d.id, d.match_id, d.timestamp
                FROM deliveries d
                LEFT JOIN astrological_predictions ap ON d.id = ap.delivery_id
                WHERE ap.delivery_id IS NULL
                LIMIT ?
            ''', (batch_size,))
            
            batch = cursor.fetchall()
            if not batch:
                break
            
            # Process batch efficiently
            batch_results = self.process_batch_fast(batch)
            
            # Batch insert results
            if batch_results:
                self.batch_insert_results(conn, batch_results)
                processed += len(batch_results)
                print(f"Progress: {processed}/{total_deliveries} processed")
        
        conn.close()
        print(f"Fast processing complete: {processed} processed, {errors} errors")

    def process_batch_fast(self, batch):
        """Process a batch of deliveries with minimal calculations"""
        results = []
        
        # Group by unique datetime/coordinates to minimize Swiss Ephemeris calls
        unique_calculations = {}
        delivery_mapping = {}
        
        for delivery_id, match_id, timestamp in batch:
            if not timestamp:
                continue
                
            try:
                dt = datetime.fromisoformat(timestamp)
                lat, lon = self.coords_cache.get(match_id, (19.0760, 72.8777))
                
                # Create key for unique calculations
                calc_key = (dt.year, dt.month, dt.day, dt.hour, dt.minute, lat, lon)
                
                if calc_key not in unique_calculations:
                    # Calculate once for this unique datetime/location
                    astro_data = self.calculate_essential_astro_data(dt, lat, lon)
                    unique_calculations[calc_key] = astro_data
                
                delivery_mapping[delivery_id] = (match_id, calc_key)
                
            except Exception as e:
                continue
        
        # Build results using cached calculations
        for delivery_id, (match_id, calc_key) in delivery_mapping.items():
            if calc_key in unique_calculations:
                astro_data = unique_calculations[calc_key]
                if 'error' not in astro_data:
                    results.append((delivery_id, match_id, astro_data))
        
        return results

    def calculate_essential_astro_data(self, dt, lat, lon):
        """Calculate only essential astrological data efficiently"""
        try:
            # Single Swiss Ephemeris setup
            swe.set_ephe_path('')
            jd = swe.julday(dt.year, dt.month, dt.day, dt.hour + dt.minute/60.0 + dt.second/3600.0)
            
            # Get all planetary positions in one go
            planets = self.get_all_planets_fast(jd)
            
            # Get house cusps
            cusps, ascmc = swe.houses(jd, lat, lon, b'P')
            
            # Essential KP data only
            moon_longitude = planets['Moon']['longitude']
            moon_nakshatra = self.get_nakshatra_fast(moon_longitude)
            moon_sub_lord = self.get_sub_lord_fast(moon_longitude)
            moon_sub_sub_lord = self.get_sub_sub_lord_fast(moon_longitude)
            moon_star_lord = self.get_star_lord_fast(moon_nakshatra)
            
            # Calculate basic favorability scores
            asc_score, desc_score = self.calculate_basic_favorability(planets, cusps, ascmc[0])
            
            # Essential data structure matching database schema
            essential_data = {
                'timestamp': dt.isoformat(),
                'coordinates': {'latitude': lat, 'longitude': lon},
                'planets': planets,
                'house_cusps': [cusps[i] for i in range(12)],
                'ascendant': ascmc[0],
                'moon_analysis': {
                    'longitude': moon_longitude,
                    'nakshatra': moon_nakshatra,
                    'sub_lord': moon_sub_lord,
                    'sub_sub_lord': moon_sub_sub_lord,
                    'star_lord': moon_star_lord,
                    'sign': self.get_sign_fast(moon_longitude)
                },
                'favorability': {
                    'asc_score': asc_score,
                    'desc_score': desc_score,
                    'moon_sl': moon_star_lord,
                    'moon_sub': moon_sub_lord,
                    'moon_ssl': moon_sub_sub_lord,
                    'moon_sl_score': self.calculate_planet_score(moon_star_lord),
                    'moon_sub_score': self.calculate_planet_score(moon_sub_lord),
                    'moon_ssl_score': self.calculate_planet_score(moon_sub_sub_lord)
                }
            }
            
            return essential_data
            
        except Exception as e:
            return {'error': f"Fast astro calculation failed: {str(e)}"}

    def get_all_planets_fast(self, jd):
        """Get all planetary positions efficiently"""
        planets = {}
        planet_ids = {
            'Sun': swe.SUN, 'Moon': swe.MOON, 'Mars': swe.MARS, 'Mercury': swe.MERCURY,
            'Jupiter': swe.JUPITER, 'Venus': swe.VENUS, 'Saturn': swe.SATURN,
            'Rahu': swe.MEAN_NODE
        }
        
        for planet_name, planet_id in planet_ids.items():
            try:
                pos, ret = swe.calc_ut(jd, planet_id)
                longitude = pos[0]
                
                planets[planet_name] = {
                    'longitude': longitude,
                    'speed': pos[3] if len(pos) > 3 else 0,
                    'sign': self.get_sign_fast(longitude),
                    'nakshatra': self.get_nakshatra_fast(longitude),
                    'sub_lord': self.get_sub_lord_fast(longitude)
                }
            except:
                planets[planet_name] = {'error': 'calculation_failed'}
        
        # Add Ketu
        if 'Rahu' in planets and 'error' not in planets['Rahu']:
            ketu_longitude = (planets['Rahu']['longitude'] + 180) % 360
            planets['Ketu'] = {
                'longitude': ketu_longitude,
                'speed': 0,
                'sign': self.get_sign_fast(ketu_longitude),
                'nakshatra': self.get_nakshatra_fast(ketu_longitude),
                'sub_lord': self.get_sub_lord_fast(ketu_longitude)
            }
        
        return planets

    def get_sign_fast(self, longitude):
        """Fast sign calculation using pre-calculated boundaries"""
        sign_index = int(longitude // 30)
        return self.sign_names[sign_index] if 0 <= sign_index < 12 else 'Unknown'

    def get_nakshatra_fast(self, longitude):
        """Fast nakshatra calculation"""
        for start, end, nakshatra in self.nakshatra_boundaries:
            if start <= longitude < end:
                return nakshatra
        return 'Unknown'

    def get_sub_lord_fast(self, longitude):
        """Fast sub lord calculation using correct column names"""
        nakshatra_row = self.nakshatra_df[
            (self.nakshatra_df['Start_Degree'] <= longitude) & 
            (longitude < self.nakshatra_df['End_Degree'])
        ]
        if not nakshatra_row.empty:
            return nakshatra_row.iloc[0]['Sub_Lord']
        return 'Unknown'

    def get_sub_sub_lord_fast(self, longitude):
        """Fast sub-sub lord calculation using correct column names"""
        nakshatra_row = self.nakshatra_df[
            (self.nakshatra_df['Start_Degree'] <= longitude) & 
            (longitude < self.nakshatra_df['End_Degree'])
        ]
        if not nakshatra_row.empty:
            return nakshatra_row.iloc[0].get('Sub_Sub_Lord', 'Unknown')
        return 'Unknown'

    def get_star_lord_fast(self, nakshatra):
        """Get star lord (nakshatra lord) for given nakshatra"""
        nakshatra_lords = {
            'Ashwini': 'Ketu', 'Bharani': 'Venus', 'Krittika': 'Sun', 'Rohini': 'Moon',
            'Mrigashira': 'Mars', 'Ardra': 'Rahu', 'Punarvasu': 'Jupiter', 'Pushya': 'Saturn',
            'Ashlesha': 'Mercury', 'Magha': 'Ketu', 'Purva Phalguni': 'Venus', 'Uttara Phalguni': 'Sun',
            'Hasta': 'Moon', 'Chitra': 'Mars', 'Swati': 'Rahu', 'Vishakha': 'Jupiter',
            'Anuradha': 'Saturn', 'Jyeshtha': 'Mercury', 'Mula': 'Ketu', 'Purva Ashadha': 'Venus',
            'Uttara Ashadha': 'Sun', 'Shravana': 'Moon', 'Dhanishta': 'Mars', 'Shatabhisha': 'Rahu',
            'Purva Bhadrapada': 'Jupiter', 'Uttara Bhadrapada': 'Saturn', 'Revati': 'Mercury'
        }
        return nakshatra_lords.get(nakshatra, 'Unknown')

    def calculate_basic_favorability(self, planets, cusps, ascendant):
        """Calculate basic favorability scores for ascendant and descendant"""
        # Simplified favorability calculation
        asc_score = 0.0
        desc_score = 0.0
        
        # Basic house weights for cricket
        asc_houses = [1, 5, 9, 10, 11]  # Favorable for ascendant team
        desc_houses = [6, 7, 8, 12]     # Favorable for descendant team
        
        # Check Moon's influence
        if 'Moon' in planets and 'error' not in planets['Moon']:
            moon_house = self.get_house_from_longitude(planets['Moon']['longitude'], cusps)
            if moon_house in asc_houses:
                asc_score += 1.0
            elif moon_house in desc_houses:
                desc_score += 1.0
        
        return asc_score, desc_score

    def get_house_from_longitude(self, longitude, cusps):
        """Determine which house a longitude falls into"""
        for i in range(12):
            next_cusp = cusps[(i + 1) % 12]
            current_cusp = cusps[i]
            
            # Handle zodiac wrap-around
            if current_cusp > next_cusp:  # Crosses 0 degrees
                if longitude >= current_cusp or longitude < next_cusp:
                    return i + 1
            else:
                if current_cusp <= longitude < next_cusp:
                    return i + 1
        return 1  # Default to first house

    def calculate_planet_score(self, planet):
        """Calculate a basic score for a planet"""
        planet_scores = {
            'Jupiter': 1.0, 'Venus': 0.8, 'Mercury': 0.6, 'Moon': 0.5,
            'Sun': 0.4, 'Mars': 0.2, 'Saturn': -0.2, 'Rahu': -0.4, 'Ketu': -0.6
        }
        return planet_scores.get(planet, 0.0)

    def batch_insert_results(self, conn, results):
        """Efficiently insert batch results using correct database schema"""
        cursor = conn.cursor()
        
        astro_data_list = []
        chart_data_list = []
        
        for delivery_id, match_id, astro_data in results:
            # Prepare data for astrological_predictions table
            favorability = astro_data.get('favorability', {})
            moon_analysis = astro_data.get('moon_analysis', {})
            
            astro_data_list.append((
                delivery_id, match_id,
                favorability.get('asc_score', 0), favorability.get('desc_score', 0),
                favorability.get('moon_sl', ''), favorability.get('moon_sub', ''), favorability.get('moon_ssl', ''),
                favorability.get('moon_sl_score', 0), favorability.get('moon_sub_score', 0), favorability.get('moon_ssl_score', 0),
                json.dumps([]), json.dumps([]), json.dumps([]),  # Empty house lists for now
                favorability.get('moon_sl', ''), favorability.get('moon_sub', ''), favorability.get('moon_ssl', ''),
                json.dumps([]), 0.0, 0.0, 0.0  # Empty ruling planets, scores
            ))
            
            # Prepare data for chart_data table
            chart_data_list.append((
                delivery_id, match_id,
                astro_data.get('ascendant', 0), moon_analysis.get('longitude', 0),
                moon_analysis.get('nakshatra', ''), 0,  # pada placeholder
                moon_analysis.get('sign', ''), '',  # sign lord placeholder
                moon_analysis.get('star_lord', ''), moon_analysis.get('sub_lord', ''),
                moon_analysis.get('sub_sub_lord', ''), json.dumps(astro_data)
            ))
        
        # Batch insert into astrological_predictions
        cursor.executemany('''
            INSERT OR REPLACE INTO astrological_predictions 
            (delivery_id, match_id, asc_score, desc_score, moon_sl, moon_sub, moon_ssl,
             moon_sl_score, moon_sub_score, moon_ssl_score, moon_sl_houses, moon_sub_houses, 
             moon_ssl_houses, moon_sl_star_lord, moon_sub_star_lord, moon_ssl_star_lord,
             ruling_planets, success_score, predicted_impact, actual_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', astro_data_list)
        
        # Batch insert into chart_data
        cursor.executemany('''
            INSERT OR REPLACE INTO chart_data 
            (delivery_id, match_id, ascendant_degree, moon_longitude, moon_nakshatra, 
             moon_pada, moon_sign, moon_sign_lord, moon_star_lord, moon_sub_lord, 
             moon_sub_sub_lord, chart_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', chart_data_list)
        
        conn.commit()

def main():
    parser = argparse.ArgumentParser(description='Fast Astrological Processor')
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size for processing')
    parser.add_argument('--db-path', default='training_analysis/cricket_predictions.db', help='Database path')
    
    args = parser.parse_args()
    
    processor = FastAstroProcessor(args.db_path)
    processor.process_all_deliveries_fast(batch_size=args.batch_size)

if __name__ == "__main__":
    main() 