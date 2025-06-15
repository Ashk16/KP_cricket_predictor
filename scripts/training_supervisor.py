import os
import sys
import json
import pandas as pd
from datetime import datetime
import swisseph as swe
from scripts.chart_generator import generate_kp_chart
from scripts.kp_favorability_rules import evaluate_favorability
from scripts.db_manager import init_database, save_match_data, save_delivery_data, save_astrological_data

# Add project root to the Python path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
ROOT_DIR = os.path.dirname(scripts_dir)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_training_analysis(match_id, match_data, match_start_str, db_path, lat=0, lon=0):
    print(f"--- Starting Analysis for: {match_id}.json ---")
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        # Save match metadata
        save_match_data(conn, match_id, match_data, match_start_str)
        
        # Get match location from metadata
        venue = match_data.get('info', {}).get('venue', '')
        if not lat or not lon:
            print(f"Warning: No location data for match {match_id} at {venue}. Using default coordinates.")
        
        # Extract team information from match data
        teams = match_data.get('info', {}).get('teams', [])
        if len(teams) != 2:
            print(f"Warning: Expected 2 teams, found {len(teams)} for match {match_id}")
            teams = ['Team1', 'Team2']  # Fallback
        
        team1, team2 = teams[0], teams[1]
        print(f"Teams: {team1} vs {team2}")
        
        # Prepare delivery data with realistic timestamps
        deliveries = []
        
        # Parse match start time from match_start_str
        try:
            match_start_dt = datetime.fromisoformat(match_start_str)
        except:
            # Fallback: use match date from JSON + default time
            match_date_str = match_data.get('info', {}).get('dates', ['2020-01-01'])[0]
            match_start_dt = datetime.fromisoformat(f"{match_date_str}T19:30:00")
        
        # Known spinner bowlers (common names - this could be expanded)
        known_spinners = {
            'R Ashwin', 'Harbhajan Singh', 'Yuzvendra Chahal', 'Kuldeep Yadav', 
            'Washington Sundar', 'Ravindra Jadeja', 'Amit Mishra', 'Piyush Chawla',
            'Imran Tahir', 'Rashid Khan', 'Mujeeb Ur Rahman', 'Sunil Narine',
            'Andre Russell', 'Kieron Pollard', 'MS Dhoni'  # Part-time spinners
        }
        
        current_time = match_start_dt
        prev_over = -1
        prev_inning = 0
        
        for inning_num, inning in enumerate(match_data['innings'], 1):
            if 'overs' not in inning:
                continue
            
            # Determine batting and bowling teams for this innings
            batting_team = inning.get('team', '')
            if batting_team == team1:
                bowling_team = team2
            elif batting_team == team2:
                bowling_team = team1
            else:
                # Fallback if team name doesn't match
                batting_team = team1 if inning_num == 1 else team2
                bowling_team = team2 if inning_num == 1 else team1
                print(f"Warning: Could not match innings team '{inning.get('team', '')}' to known teams. Using fallback.")
            
            print(f"Innings {inning_num}: {batting_team} batting, {bowling_team} bowling")
                
            # Innings break (except for first inning)
            if inning_num > 1:
                current_time += pd.Timedelta(minutes=20)  # 20-minute innings break
                
            for over_data in inning['overs']:
                over_num = over_data.get('over', 0)
                
                # Over change time (except for first over of innings)
                if prev_inning == inning_num and over_num != prev_over:
                    current_time += pd.Timedelta(seconds=60)  # 60 seconds for over change
                
                for ball_num, delivery in enumerate(over_data.get('deliveries', []), 1):
                    bowler = delivery.get('bowler', '')
                    
                    # Determine delivery time based on bowler type
                    if any(spinner in bowler for spinner in known_spinners):
                        base_delivery_time = 30  # 30 seconds for spinners
                    else:
                        base_delivery_time = 45  # 45 seconds for fast bowlers
                    
                    # Additional time for special events
                    extra_time = 0
                    
                    # Wicket taken - extra time for celebrations and new batsman
                    if delivery.get('wickets'):
                        extra_time += 90  # 1.5 minutes extra for wicket
                    
                    # Boundary - slight extra time for celebrations
                    runs = delivery.get('runs', {}).get('total', 0)
                    if runs >= 4:
                        extra_time += 15  # 15 seconds extra for boundary
                    
                    # Wide or no-ball - extra time for field adjustments
                    if delivery.get('runs', {}).get('extras', 0) > 0:
                        extra_time += 10  # 10 seconds extra for extras
                    
                    # Add the time
                    current_time += pd.Timedelta(seconds=base_delivery_time + extra_time)
                    
                    deliveries.append({
                        'inning': inning_num,
                        'over': over_num + 1,  # Convert 0-based to 1-based
                        'ball': ball_num,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'batsman': delivery.get('batter', ''),
                        'bowler': delivery.get('bowler', ''),
                        'runs_off_bat': delivery.get('runs', {}).get('batter', 0),
                        'extras': delivery.get('runs', {}).get('extras', 0),
                        'wicket_type': delivery.get('wickets', [{}])[0].get('kind', '') if delivery.get('wickets') else '',
                        'dismissal': delivery.get('wickets', [{}])[0].get('player_out', '') if delivery.get('wickets') else '',
                        'timestamp': current_time.isoformat()
                    })
                
                prev_over = over_num
                prev_inning = inning_num
        save_delivery_data(conn, match_id, deliveries)
        
        # Get delivery IDs
        cursor = conn.cursor()
        cursor.execute('SELECT id, timestamp FROM deliveries WHERE match_id = ? ORDER BY inning, over, ball', (match_id,))
        delivery_rows = cursor.fetchall()
        print(f"Enriched {len(delivery_rows)} deliveries with timestamps.")
        print("Generating astrological predictions...")
        
        nakshatra_df = pd.read_csv("config/nakshatra_sub_lords_longitudes.csv")
        error_count = 0
        max_errors = 10  # Maximum number of errors before skipping the match
        
        for i, (delivery_id, timestamp) in enumerate(delivery_rows, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(delivery_rows)} deliveries...")
            if not timestamp:
                continue
                
            try:
                dt = datetime.fromisoformat(timestamp)
                chart = generate_kp_chart(dt, lat, lon, nakshatra_df)
                
                if "error" in chart:
                    error_count += 1
                    print(f"Error generating chart for delivery {delivery_id}: {chart['error']}")
                    if error_count >= max_errors:
                        print(f"Too many errors ({error_count}) for match {match_id}. Skipping remaining deliveries.")
                        break
                    continue
                    
                favorability_data = evaluate_favorability(chart, nakshatra_df)
                if "error" in favorability_data:
                    error_count += 1
                    print(f"Error evaluating favorability for delivery {delivery_id}: {favorability_data['error']}")
                    if error_count >= max_errors:
                        print(f"Too many errors ({error_count}) for match {match_id}. Skipping remaining deliveries.")
                        break
                    continue
                
                # Calculate asc_score and desc_score for rule optimizer compatibility
                # Extract individual lord scores and calculate weighted totals
                moon_sl_score = favorability_data.get('moon_sl_score', 0)
                moon_sub_score = favorability_data.get('moon_sub_score', 0) 
                moon_ssl_score = favorability_data.get('moon_ssl_score', 0)
                
                # Apply the same weights as in evaluate_favorability (50/30/20)
                total_asc_score = 0
                total_desc_score = 0
                
                # For positive scores, add to asc_score; for negative, add absolute value to desc_score
                for score, weight in [(moon_sl_score, 0.5), (moon_sub_score, 0.3), (moon_ssl_score, 0.2)]:
                    if score > 0:
                        total_asc_score += score * weight
                    else:
                        total_desc_score += abs(score) * weight
                
                # Add calculated scores to favorability_data
                favorability_data['asc_score'] = total_asc_score
                favorability_data['desc_score'] = total_desc_score
                
                # Add lord names for rule optimizer
                favorability_data['moon_sl'] = chart.get('moon_star_lord', '')
                favorability_data['moon_sub'] = chart.get('moon_sub_lord', '')
                favorability_data['moon_ssl'] = chart.get('moon_sub_sub_lord', '')
                    
                save_astrological_data(conn, delivery_id, match_id, chart, favorability_data)
                
            except Exception as e:
                error_count += 1
                print(f"Error processing delivery {delivery_id}: {str(e)}")
                if error_count >= max_errors:
                    print(f"Too many errors ({error_count}) for match {match_id}. Skipping remaining deliveries.")
                    break
                continue
                
        print("Prediction generation complete.")
        if error_count > 0:
            print(f"Completed with {error_count} errors.")
        print(f"--- Analysis complete. Data saved to database: {db_path} ---\n")
        
    except Exception as e:
        print(f"Error processing match {match_id}: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def run_basic_data_processing():
    """Process only basic match and delivery data without astrological calculations"""
    db_path = os.path.join(ROOT_DIR, 'training_analysis', 'cricket_predictions.db')
    init_database(db_path)
    index_path = os.path.join(ROOT_DIR, 'match_data', 'match_index.csv')
    
    if not os.path.exists(index_path):
        print(f"Error: Match index file not found at {index_path}")
        return
    
    index_df = pd.read_csv(index_path, comment='#')
    total_matches = len(index_df)
    processed_matches = 0
    
    print(f"Starting basic data processing for {total_matches} matches...")
    
    for _, row in index_df.iterrows():
        match_id_raw = str(row['match_id']) if 'match_id' in row else str(row['cricsheet_id'])
        match_id = match_id_raw.replace('.json', '')
        
        # Get coordinates from index
        lat = row.get('latitude') if 'latitude' in row and not pd.isnull(row.get('latitude')) else None
        lon = row.get('longitude') if 'longitude' in row and not pd.isnull(row.get('longitude')) else None
        
        json_path = os.path.join(ROOT_DIR, 'match_data', 'cricsheet_t20_json', f"{match_id}.json")
        if not os.path.exists(json_path):
            print(f"Warning: Match file not found: {json_path}")
            continue
            
        try:
            with open(json_path, 'r') as f:
                match_data = json.load(f)
            
            run_basic_match_processing(
                match_id=match_id,
                match_data=match_data,
                match_start_str=row['start_datetime'],
                db_path=db_path,
                lat=lat,
                lon=lon
            )
            processed_matches += 1
            
            if processed_matches % 100 == 0:
                print(f"Progress: {processed_matches}/{total_matches} matches processed")
                
        except Exception as e:
            print(f"Error processing match {match_id}: {str(e)}")
            continue
    
    print(f"Basic data processing complete: {processed_matches}/{total_matches} matches processed")

def run_basic_match_processing(match_id, match_data, match_start_str, db_path, lat=None, lon=None):
    """Process basic match data without astrological calculations"""
    print(f"--- Processing Basic Data for: {match_id}.json ---")
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    try:
        # Save match metadata with coordinates
        save_match_data(conn, match_id, match_data, match_start_str, lat, lon)
        
        # Extract team information from match data
        teams = match_data.get('info', {}).get('teams', [])
        if len(teams) != 2:
            print(f"Warning: Expected 2 teams, found {len(teams)} for match {match_id}")
            teams = ['Team1', 'Team2']  # Fallback
        
        team1, team2 = teams[0], teams[1]
        
        # Parse match start time
        try:
            match_start_dt = datetime.fromisoformat(match_start_str)
        except:
            match_date_str = match_data.get('info', {}).get('dates', ['2020-01-01'])[0]
            match_start_dt = datetime.fromisoformat(f"{match_date_str}T19:30:00")
        
        # Known spinner bowlers
        known_spinners = {
            'R Ashwin', 'Harbhajan Singh', 'Yuzvendra Chahal', 'Kuldeep Yadav', 
            'Washington Sundar', 'Ravindra Jadeja', 'Amit Mishra', 'Piyush Chawla',
            'Imran Tahir', 'Rashid Khan', 'Mujeeb Ur Rahman', 'Sunil Narine',
            'Andre Russell', 'Kieron Pollard', 'MS Dhoni'
        }
        
        current_time = match_start_dt
        prev_over = -1
        prev_inning = 0
        deliveries = []
        
        for inning_num, inning in enumerate(match_data['innings'], 1):
            if 'overs' not in inning:
                continue
            
            # Determine batting and bowling teams
            batting_team = inning.get('team', '')
            if batting_team == team1:
                bowling_team = team2
            elif batting_team == team2:
                bowling_team = team1
            else:
                batting_team = team1 if inning_num == 1 else team2
                bowling_team = team2 if inning_num == 1 else team1
                
            # Innings break
            if inning_num > 1:
                current_time += pd.Timedelta(minutes=20)
                
            for over_data in inning['overs']:
                over_num = over_data.get('over', 0)
                
                # Over change time
                if prev_inning == inning_num and over_num != prev_over:
                    current_time += pd.Timedelta(seconds=60)
                
                for ball_num, delivery in enumerate(over_data.get('deliveries', []), 1):
                    bowler = delivery.get('bowler', '')
                    
                    # Calculate delivery timing
                    if any(spinner in bowler for spinner in known_spinners):
                        base_delivery_time = 30
                    else:
                        base_delivery_time = 45
                    
                    extra_time = 0
                    if delivery.get('wickets'):
                        extra_time += 90
                    
                    runs = delivery.get('runs', {}).get('total', 0)
                    if runs >= 4:
                        extra_time += 15
                    
                    if delivery.get('runs', {}).get('extras', 0) > 0:
                        extra_time += 10
                    
                    current_time += pd.Timedelta(seconds=base_delivery_time + extra_time)
                    
                    deliveries.append({
                        'inning': inning_num,
                        'over': over_num + 1,
                        'ball': ball_num,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'batsman': delivery.get('batter', ''),
                        'bowler': delivery.get('bowler', ''),
                        'runs_off_bat': delivery.get('runs', {}).get('batter', 0),
                        'extras': delivery.get('runs', {}).get('extras', 0),
                        'wicket_type': delivery.get('wickets', [{}])[0].get('kind', '') if delivery.get('wickets') else '',
                        'dismissal': delivery.get('wickets', [{}])[0].get('player_out', '') if delivery.get('wickets') else '',
                        'timestamp': current_time.isoformat()
                    })
                
                prev_over = over_num
                prev_inning = inning_num
        
        save_delivery_data(conn, match_id, deliveries)
        print(f"--- Basic processing complete for {match_id}: {len(deliveries)} deliveries ---")
        
    except Exception as e:
        print(f"Error processing match {match_id}: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def run_bulk_analysis():
    """Legacy function - now calls basic processing"""
    run_basic_data_processing()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--basic':
        run_basic_data_processing()
    else:
        run_bulk_analysis() 