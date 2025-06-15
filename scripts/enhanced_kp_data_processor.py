#!/usr/bin/env python3
"""
Enhanced KP Data Processor
Comprehensive astrological data processing based on muhurta charts and traditional KP principles
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability
import swisseph as swe

class EnhancedKPDataProcessor:
    def __init__(self, db_path="training_analysis/cricket_predictions.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Load nakshatra data
        self.nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
        
        # KP Constants
        self.benefic_planets = ["Jupiter", "Venus"]
        self.malefic_planets = ["Saturn", "Mars", "Sun", "Rahu", "Ketu"]
        self.neutral_planets = ["Moon", "Mercury"]
        
        # Default location (Mumbai for IPL)
        self.default_lat = 19.0760
        self.default_lon = 72.8777
        
        print("Enhanced KP Data Processor initialized")
    
    def _create_enhanced_tables(self):
        """Create enhanced tables for comprehensive KP analysis"""
        
        # Muhurta chart data for each match
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS muhurta_charts (
                match_id TEXT PRIMARY KEY,
                match_start_time TEXT,
                location_lat REAL,
                location_lon REAL,
                
                -- Ascendant details
                ascendant_degree REAL,
                ascendant_sign INTEGER,
                
                -- All 12 house cusps
                cusp_1_degree REAL,
                cusp_2_degree REAL,
                cusp_3_degree REAL,
                cusp_4_degree REAL,
                cusp_5_degree REAL,
                cusp_6_degree REAL,
                cusp_7_degree REAL,
                cusp_8_degree REAL,
                cusp_9_degree REAL,
                cusp_10_degree REAL,
                cusp_11_degree REAL,
                cusp_12_degree REAL,
                
                -- House cuspal sub lords
                cusp_1_sub_lord TEXT,
                cusp_2_sub_lord TEXT,
                cusp_3_sub_lord TEXT,
                cusp_4_sub_lord TEXT,
                cusp_5_sub_lord TEXT,
                cusp_6_sub_lord TEXT,
                cusp_7_sub_lord TEXT,
                cusp_8_sub_lord TEXT,
                cusp_9_sub_lord TEXT,
                cusp_10_sub_lord TEXT,
                cusp_11_sub_lord TEXT,
                cusp_12_sub_lord TEXT,
                
                -- Planetary positions at match start
                sun_longitude REAL,
                moon_longitude REAL,
                mars_longitude REAL,
                mercury_longitude REAL,
                jupiter_longitude REAL,
                venus_longitude REAL,
                saturn_longitude REAL,
                rahu_longitude REAL,
                ketu_longitude REAL,
                
                -- Planetary house positions
                sun_house INTEGER,
                moon_house INTEGER,
                mars_house INTEGER,
                mercury_house INTEGER,
                jupiter_house INTEGER,
                venus_house INTEGER,
                saturn_house INTEGER,
                rahu_house INTEGER,
                ketu_house INTEGER,
                
                -- Retrograde status
                sun_retrograde BOOLEAN,
                moon_retrograde BOOLEAN,
                mars_retrograde BOOLEAN,
                mercury_retrograde BOOLEAN,
                jupiter_retrograde BOOLEAN,
                venus_retrograde BOOLEAN,
                saturn_retrograde BOOLEAN,
                rahu_retrograde BOOLEAN,
                ketu_retrograde BOOLEAN,
                
                -- Combustion status
                moon_combust BOOLEAN,
                mars_combust BOOLEAN,
                mercury_combust BOOLEAN,
                jupiter_combust BOOLEAN,
                venus_combust BOOLEAN,
                saturn_combust BOOLEAN,
                
                -- Moon's hierarchical lords
                moon_sign TEXT,
                moon_sign_lord TEXT,
                moon_nakshatra TEXT,
                moon_star_lord TEXT,
                moon_sub_lord TEXT,
                moon_sub_sub_lord TEXT,
                
                -- Day lord
                day_lord TEXT,
                
                -- Overall muhurta strength
                muhurta_strength_score REAL,
                ascendant_favorability REAL,
                descendant_favorability REAL
            )
        """)
        
        self.conn.commit()
        print("Enhanced KP tables created successfully")
    
    def calculate_combustion(self, sun_longitude, planet_longitude, planet_name):
        """Calculate if a planet is combust (too close to Sun)"""
        
        # Combustion distances in degrees
        combustion_distances = {
            'Moon': 12,
            'Mars': 17,
            'Mercury': 14,
            'Jupiter': 11,
            'Venus': 10,
            'Saturn': 15
        }
        
        if planet_name not in combustion_distances:
            return False
        
        # Calculate angular distance
        diff = abs(sun_longitude - planet_longitude)
        if diff > 180:
            diff = 360 - diff
        
        return diff <= combustion_distances[planet_name]
    
    def calculate_house_position(self, planet_longitude, house_cusps):
        """Calculate which house a planet is positioned in"""
        
        for house in range(1, 13):
            cusp_start = house_cusps[house - 1]
            cusp_end = house_cusps[house % 12]
            
            # Handle zodiac wrap-around
            if cusp_start <= cusp_end:
                if cusp_start <= planet_longitude < cusp_end:
                    return house
            else:  # Cusp crosses 0 degrees
                if planet_longitude >= cusp_start or planet_longitude < cusp_end:
                    return house
        
        return 1  # Default to 1st house
    
    def get_ruling_planets_for_longitude(self, longitude):
        """Get ruling planets for a given longitude"""
        
        # Find the matching row in nakshatra data
        match = self.nakshatra_df[
            (self.nakshatra_df['Start_Degree'] <= longitude) & 
            (self.nakshatra_df['End_Degree'] > longitude)
        ]
        
        if match.empty:
            # Handle edge case for 360 degrees
            if longitude >= 359.999:
                match = self.nakshatra_df.iloc[-1:]
            if match.empty:
                return {'star_lord': 'Unknown', 'sub_lord': 'Unknown', 'sub_sub_lord': 'Unknown'}
        
        row = match.iloc[0]
        return {
            'star_lord': row['Star_Lord'],
            'sub_lord': row['Sub_Lord'],
            'sub_sub_lord': row['Sub_Sub_Lord']
        }
    
    def generate_muhurta_chart(self, match_id, match_start_time, lat=None, lon=None):
        """Generate comprehensive muhurta chart for a match"""
        
        if lat is None:
            lat = self.default_lat
        if lon is None:
            lon = self.default_lon
        
        try:
            # Parse match start time
            if isinstance(match_start_time, str):
                start_dt = pd.to_datetime(match_start_time).to_pydatetime()
            else:
                start_dt = match_start_time
            
            # Generate KP chart
            chart = generate_kp_chart(start_dt, lat, lon, self.nakshatra_df)
            
            if 'error' in chart:
                print(f"Error generating muhurta chart for match {match_id}: {chart['error']}")
                return None
            
            # Calculate house cusps
            jd = swe.julday(start_dt.year, start_dt.month, start_dt.day, 
                           start_dt.hour + start_dt.minute / 60 + start_dt.second / 3600 - 5.5)
            houses, ascmc = swe.houses(jd, lat, lon, b'P')
            
            # Enhanced muhurta data
            muhurta_data = {
                'match_id': match_id,
                'match_start_time': match_start_time,
                'location_lat': lat,
                'location_lon': lon,
                
                # Ascendant details
                'ascendant_degree': chart['ascendant_degree'],
                'ascendant_sign': int(chart['ascendant_degree'] // 30),
                'moon_longitude': chart['moon_longitude'],
                'moon_nakshatra': chart['moon_nakshatra'],
                'moon_star_lord': chart['moon_star_lord'],
                'moon_sub_lord': chart['moon_sub_lord'],
                'moon_sub_sub_lord': chart['moon_sub_sub_lord'],
                'moon_sign': chart['moon_sign'],
                
                # Day lord
                'day_lord': ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'][start_dt.weekday()],
                
                # House cusps
                'houses': houses,
                'planets': chart['planets']
            }
            
            # Add house positions for planets
            for planet, planet_data in chart['planets'].items():
                house_pos = self.calculate_house_position(planet_data['longitude'], houses)
                muhurta_data['planets'][planet]['house'] = house_pos
                muhurta_data[f'{planet.lower()}_house'] = house_pos
                
                # Add combustion status
                if planet != 'Sun':
                    is_combust = self.calculate_combustion(
                        chart['planets']['Sun']['longitude'],
                        planet_data['longitude'],
                        planet
                    )
                    muhurta_data[f'{planet.lower()}_combust'] = is_combust
            
            # Calculate house cuspal sub lords
            for house_num in range(1, 13):
                cusp_degree = houses[house_num - 1]
                cusp_ruling = self.get_ruling_planets_for_longitude(cusp_degree)
                muhurta_data[f'cusp_{house_num}_degree'] = cusp_degree
                muhurta_data[f'cusp_{house_num}_sub_lord'] = cusp_ruling.get('sub_lord', 'Unknown')
            
            return muhurta_data
            
        except Exception as e:
            print(f"Error generating muhurta chart for match {match_id}: {str(e)}")
            return None
    
    def process_match_muhurta(self, match_id, match_start_time):
        """Process complete muhurta analysis for a match"""
        
        print(f"Processing muhurta for match {match_id}...")
        
        # Generate muhurta chart
        muhurta_data = self.generate_muhurta_chart(match_id, match_start_time)
        if not muhurta_data:
            return False
        
        # Calculate overall muhurta strength
        asc_favorability = 0
        desc_favorability = 0
        
        # Simple favorability based on key house sub lords
        key_asc_houses = [1, 5, 6, 10, 11]  # Favorable for ascendant
        key_desc_houses = [7, 8, 12]        # Favorable for descendant
        
        for house in key_asc_houses:
            sub_lord = muhurta_data.get(f'cusp_{house}_sub_lord', 'Unknown')
            if sub_lord in self.benefic_planets:
                asc_favorability += 2
            elif sub_lord in self.malefic_planets:
                asc_favorability -= 1
        
        for house in key_desc_houses:
            sub_lord = muhurta_data.get(f'cusp_{house}_sub_lord', 'Unknown')
            if sub_lord in self.benefic_planets:
                desc_favorability += 2
            elif sub_lord in self.malefic_planets:
                desc_favorability -= 1
        
        muhurta_strength = asc_favorability - desc_favorability
        
        # Save to database
        try:
            # Insert muhurta chart data
            muhurta_insert_data = {
                'match_id': match_id,
                'match_start_time': match_start_time,
                'location_lat': self.default_lat,
                'location_lon': self.default_lon,
                'ascendant_degree': muhurta_data['ascendant_degree'],
                'ascendant_sign': muhurta_data['ascendant_sign'],
                'moon_longitude': muhurta_data['moon_longitude'],
                'moon_nakshatra': muhurta_data['moon_nakshatra'],
                'moon_star_lord': muhurta_data['moon_star_lord'],
                'moon_sub_lord': muhurta_data['moon_sub_lord'],
                'moon_sub_sub_lord': muhurta_data['moon_sub_sub_lord'],
                'moon_sign': muhurta_data['moon_sign'],
                'day_lord': muhurta_data['day_lord'],
                'muhurta_strength_score': muhurta_strength,
                'ascendant_favorability': asc_favorability,
                'descendant_favorability': desc_favorability
            }
            
            # Add house cusps
            for i in range(12):
                muhurta_insert_data[f'cusp_{i+1}_degree'] = muhurta_data['houses'][i]
                muhurta_insert_data[f'cusp_{i+1}_sub_lord'] = muhurta_data.get(f'cusp_{i+1}_sub_lord', 'Unknown')
            
            # Add planetary data
            for planet, planet_data in muhurta_data['planets'].items():
                planet_lower = planet.lower()
                muhurta_insert_data[f'{planet_lower}_longitude'] = planet_data['longitude']
                muhurta_insert_data[f'{planet_lower}_house'] = planet_data.get('house', 1)
                muhurta_insert_data[f'{planet_lower}_retrograde'] = planet_data.get('retrograde', False)
                if planet != 'Sun':
                    muhurta_insert_data[f'{planet_lower}_combust'] = muhurta_data.get(f'{planet_lower}_combust', False)
            
            # Insert muhurta chart
            columns = ', '.join(muhurta_insert_data.keys())
            placeholders = ', '.join(['?' for _ in muhurta_insert_data])
            
            self.conn.execute(f"""
                INSERT OR REPLACE INTO muhurta_charts ({columns})
                VALUES ({placeholders})
            """, list(muhurta_insert_data.values()))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving muhurta data for match {match_id}: {str(e)}")
            return False
    
    def process_all_matches(self):
        """Process muhurta charts for all matches in database"""
        
        print("Processing muhurta charts for all matches...")
        
        # Create enhanced tables
        self._create_enhanced_tables()
        
        # Get all matches with start times
        matches_df = pd.read_sql_query("""
            SELECT DISTINCT 
                match_id,
                MIN(timestamp) as match_start_time
            FROM deliveries 
            WHERE timestamp IS NOT NULL
            GROUP BY match_id
            HAVING COUNT(*) > 100
            ORDER BY match_start_time
        """, self.conn)
        
        print(f"Found {len(matches_df)} matches to process")
        
        success_count = 0
        for _, match in matches_df.iterrows():
            if self.process_match_muhurta(match['match_id'], match['match_start_time']):
                success_count += 1
            
            if success_count % 10 == 0:
                print(f"Processed {success_count} matches...")
        
        print(f"Successfully processed {success_count}/{len(matches_df)} matches")
        return success_count
    
    def close(self):
        """Close database connection"""
        self.conn.close()

def main():
    """Main function to run enhanced KP data processing"""
    
    processor = EnhancedKPDataProcessor()
    
    try:
        # Process all matches
        success_count = processor.process_all_matches()
        print(f"Enhanced KP data processing complete. Processed {success_count} matches.")
        
    except Exception as e:
        print(f"Error in enhanced KP processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        processor.close()

if __name__ == "__main__":
    main() 