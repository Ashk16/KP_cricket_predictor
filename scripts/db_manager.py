import sqlite3
import json
from datetime import datetime

def init_database(db_path):
    """Initialize the database with required tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create matches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            start_datetime TEXT,
            venue TEXT,
            team1 TEXT,
            team2 TEXT,
            winner TEXT,
            total_overs INTEGER,
            latitude REAL,
            longitude REAL,
            processed_at TEXT
        )
    ''')
    
    # Create deliveries table with batting_team and bowling_team columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            timestamp TEXT,
            inning INTEGER,
            over INTEGER,
            ball INTEGER,
            batting_team TEXT,
            bowling_team TEXT,
            striker TEXT,
            non_striker TEXT,
            bowler TEXT,
            runs_off_bat INTEGER,
            extras INTEGER,
            wicket_kind TEXT,
            player_out TEXT,
            FOREIGN KEY (match_id) REFERENCES matches (match_id)
        )
    ''')
    
    # Create astrological_predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS astrological_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            delivery_id INTEGER,
            match_id TEXT,
            asc_score REAL,
            desc_score REAL,
            moon_sl TEXT,
            moon_sub TEXT,
            moon_ssl TEXT,
            moon_sl_score REAL,
            moon_sub_score REAL,
            moon_ssl_score REAL,
            moon_sl_houses TEXT,
            moon_sub_houses TEXT,
            moon_ssl_houses TEXT,
            moon_sl_star_lord TEXT,
            moon_sub_star_lord TEXT,
            moon_ssl_star_lord TEXT,
            ruling_planets TEXT,
            success_score REAL,
            predicted_impact REAL,
            actual_impact REAL,
            FOREIGN KEY (delivery_id) REFERENCES deliveries (id),
            FOREIGN KEY (match_id) REFERENCES matches (match_id)
        )
    ''')
    
    # Create chart_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chart_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            delivery_id INTEGER,
            match_id TEXT,
            ascendant_degree REAL,
            moon_longitude REAL,
            moon_nakshatra TEXT,
            moon_pada INTEGER,
            moon_sign TEXT,
            moon_sign_lord TEXT,
            moon_star_lord TEXT,
            moon_sub_lord TEXT,
            moon_sub_sub_lord TEXT,
            chart_json TEXT,
            FOREIGN KEY (delivery_id) REFERENCES deliveries (id),
            FOREIGN KEY (match_id) REFERENCES matches (match_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}")

def save_match_data(conn, match_id, match_data, match_start_str, lat=None, lon=None):
    """Save match metadata to the database"""
    cursor = conn.cursor()
    
    # Extract match information
    info = match_data.get('info', {})
    teams = info.get('teams', [])
    team1 = teams[0] if len(teams) > 0 else ''
    team2 = teams[1] if len(teams) > 1 else ''
    winner = info.get('outcome', {}).get('winner', '')
    venue = info.get('venue', '')
    total_overs = info.get('overs', 20)
    processed_at = datetime.now().isoformat()
    
    # Insert or replace match data
    cursor.execute('''
        INSERT OR REPLACE INTO matches 
        (match_id, start_datetime, venue, team1, team2, winner, total_overs, latitude, longitude, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (match_id, match_start_str, venue, team1, team2, winner, total_overs, lat, lon, processed_at))
    
    conn.commit()
    print(f"Match data saved: {match_id}")

def save_delivery_data(conn, match_id, deliveries):
    """Save delivery data to the database"""
    cursor = conn.cursor()
    
    # Clear existing deliveries for this match
    cursor.execute('DELETE FROM deliveries WHERE match_id = ?', (match_id,))
    
    # Insert new delivery data
    for delivery in deliveries:
        cursor.execute('''
            INSERT INTO deliveries 
            (match_id, timestamp, inning, over, ball, batting_team, bowling_team, 
             striker, non_striker, bowler, runs_off_bat, extras, wicket_kind, player_out)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_id,
            delivery['timestamp'],
            delivery['inning'],
            delivery['over'],
            delivery['ball'],
            delivery['batting_team'],
            delivery['bowling_team'],
            delivery['batsman'],  # striker
            delivery.get('non_striker', ''),  # non_striker
            delivery['bowler'],
            delivery['runs_off_bat'],
            delivery['extras'],
            delivery['wicket_type'],  # wicket_kind
            delivery['dismissal']  # player_out
        ))
    
    conn.commit()
    print(f"Saved {len(deliveries)} deliveries for match {match_id}")

def save_astrological_data(conn, delivery_id, match_id, chart, favorability_data):
    """Save astrological predictions and chart data to the database"""
    cursor = conn.cursor()
    
    # Save astrological predictions
    cursor.execute('''
        INSERT INTO astrological_predictions 
        (delivery_id, match_id, asc_score, desc_score, moon_sl, moon_sub, moon_ssl,
         moon_sl_score, moon_sub_score, moon_ssl_score, moon_sl_houses, moon_sub_houses, 
         moon_ssl_houses, moon_sl_star_lord, moon_sub_star_lord, moon_ssl_star_lord,
         ruling_planets, success_score, predicted_impact, actual_impact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        delivery_id,
        match_id,
        favorability_data.get('asc_score', 0),
        favorability_data.get('desc_score', 0),
        favorability_data.get('moon_sl', ''),
        favorability_data.get('moon_sub', ''),
        favorability_data.get('moon_ssl', ''),
        favorability_data.get('moon_sl_score', 0),
        favorability_data.get('moon_sub_score', 0),
        favorability_data.get('moon_ssl_score', 0),
        json.dumps(favorability_data.get('moon_sl_houses', [])),
        json.dumps(favorability_data.get('moon_sub_houses', [])),
        json.dumps(favorability_data.get('moon_ssl_houses', [])),
        favorability_data.get('moon_sl_star_lord', ''),
        favorability_data.get('moon_sub_star_lord', ''),
        favorability_data.get('moon_ssl_star_lord', ''),
        favorability_data.get('ruling_planets', ''),
        favorability_data.get('success_score', 0),
        favorability_data.get('predicted_impact', 0),
        favorability_data.get('actual_impact', 0)
    ))
    
    # Save chart data
    cursor.execute('''
        INSERT INTO chart_data 
        (delivery_id, match_id, ascendant_degree, moon_longitude, moon_nakshatra, 
         moon_pada, moon_sign, moon_sign_lord, moon_star_lord, moon_sub_lord, 
         moon_sub_sub_lord, chart_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        delivery_id,
        match_id,
        chart.get('ascendant_degree', 0),
        chart.get('moon_longitude', 0),
        chart.get('moon_nakshatra', ''),
        chart.get('moon_pada', 0),
        chart.get('moon_sign', ''),
        chart.get('moon_sign_lord', ''),
        chart.get('moon_star_lord', ''),
        chart.get('moon_sub_lord', ''),
        chart.get('moon_sub_sub_lord', ''),
        json.dumps(chart)
    ))
    
    conn.commit() 