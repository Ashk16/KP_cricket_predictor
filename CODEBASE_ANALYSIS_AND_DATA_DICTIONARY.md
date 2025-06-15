# KP Cricket Predictor - Codebase Analysis & Data Dictionary

## 🚨 CRITICAL ERRORS IDENTIFIED

### 1. **Variable Naming Inconsistencies**

#### **Database Column Mismatches:**
- **deliveries table**: Uses `inning` (singular) but code often references `innings` (plural)
- **astrological_predictions table**: Missing `favorability_score`, `ascendant_strength`, `descendant_strength` columns that code expects
- **matches table**: Uses `match_id` as primary key, but some code expects `id`

#### **Nakshatra Data Access:**
- **Error**: `KeyError: 'Start_Degree'` occurs when nakshatra DataFrame is not properly loaded
- **Root Cause**: Multiple functions try to access `nakshatra_df['Start_Degree']` and `nakshatra_df['End_Degree']` without checking if DataFrame is loaded
- **Files Affected**: `chart_generator.py`, `kp_favorability_rules.py`, `kp_timeline_predictor.py`

#### **Chart Data Structure:**
- **Error**: Code expects `chart['planets']` but sometimes gets string error messages
- **Root Cause**: Chart generation functions return `{"error": "message"}` but calling code doesn't check for errors
- **Files Affected**: `kp_timeline_predictor.py`, `training_supervisor.py`

### 2. **Missing Database Columns**
Several scripts expect columns that don't exist in the current database schema:
- `team1`, `team2`, `batting_team` in deliveries table
- `favorability_score`, `ascendant_strength`, `descendant_strength` in astrological_predictions
- `date` column in matches table (should be `start_datetime`)

### 3. **Import Path Issues**
- Inconsistent relative imports across different script directories
- Some scripts fail to import required modules due to path issues

---

## 📊 COMPLETE DATABASE SCHEMA DICTIONARY

### **Core Tables (Currently Implemented)**

#### **1. matches**
```sql
CREATE TABLE matches (
    match_id TEXT PRIMARY KEY,           -- Unique match identifier
    start_datetime TEXT,                 -- Match start time (ISO format)
    venue TEXT,                          -- Match venue name
    team1 TEXT,                          -- First team name
    team2 TEXT,                          -- Second team name  
    winner TEXT,                         -- Winning team name
    total_overs INTEGER,                 -- Total overs in match
    processed_at TEXT                    -- Processing timestamp
)
```

#### **2. deliveries**
```sql
CREATE TABLE deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique delivery ID
    match_id TEXT,                          -- Foreign key to matches
    inning INTEGER,                         -- Inning number (1 or 2)
    over INTEGER,                           -- Over number (0-based)
    ball INTEGER,                           -- Ball number (1-6)
    batsman TEXT,                           -- Batsman name
    bowler TEXT,                            -- Bowler name
    runs_off_bat INTEGER,                   -- Runs scored off bat
    extras INTEGER,                         -- Extra runs
    wicket_type TEXT,                       -- Type of dismissal
    dismissal TEXT,                         -- Dismissed player name
    timestamp TEXT,                         -- Delivery timestamp
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
)
```

#### **3. astrological_predictions**
```sql
CREATE TABLE astrological_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique prediction ID
    delivery_id INTEGER,                    -- Foreign key to deliveries
    match_id TEXT,                          -- Foreign key to matches
    asc_score REAL,                         -- Ascendant favorability score
    desc_score REAL,                        -- Descendant favorability score
    moon_sl TEXT,                           -- Moon sub lord
    moon_sub TEXT,                          -- Moon sub
    moon_ssl TEXT,                          -- Moon sub-sub lord
    moon_sl_score REAL,                     -- Moon sub lord score
    moon_sub_score REAL,                    -- Moon sub score
    moon_ssl_score REAL,                    -- Moon sub-sub lord score
    moon_sl_houses TEXT,                    -- JSON: Moon SL house significances
    moon_sub_houses TEXT,                   -- JSON: Moon sub house significances
    moon_ssl_houses TEXT,                   -- JSON: Moon SSL house significances
    moon_sl_star_lord TEXT,                 -- Moon SL star lord
    moon_sub_star_lord TEXT,                -- Moon sub star lord
    moon_ssl_star_lord TEXT,                -- Moon SSL star lord
    ruling_planets TEXT,                    -- Ruling planets string
    success_score REAL,                     -- Success probability score
    predicted_impact REAL,                  -- Predicted impact value
    actual_impact REAL,                     -- Actual impact value
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
)
```

#### **4. chart_data**
```sql
CREATE TABLE chart_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique chart ID
    delivery_id INTEGER,                    -- Foreign key to deliveries
    match_id TEXT,                          -- Foreign key to matches
    ascendant_degree REAL,                  -- Ascendant degree
    moon_longitude REAL,                    -- Moon longitude
    moon_nakshatra TEXT,                    -- Moon nakshatra
    moon_pada INTEGER,                      -- Moon pada
    moon_sign TEXT,                         -- Moon sign
    moon_sign_lord TEXT,                    -- Moon sign lord
    moon_star_lord TEXT,                    -- Moon star lord
    moon_sub_lord TEXT,                     -- Moon sub lord
    moon_sub_sub_lord TEXT,                 -- Moon sub-sub lord
    chart_json TEXT,                        -- Complete chart as JSON
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
)
```

### **Enhanced Tables (Planned/Partially Implemented)**

#### **5. muhurta_charts**
```sql
CREATE TABLE muhurta_charts (
    match_id TEXT PRIMARY KEY,             -- Foreign key to matches
    match_start_time TEXT,                 -- Match start time
    location_lat REAL,                     -- Latitude
    location_lon REAL,                     -- Longitude
    ascendant_degree REAL,                 -- Ascendant degree
    ascendant_sign INTEGER,                -- Ascendant sign (0-11)
    
    -- House cusps (1-12)
    cusp_1_degree REAL, cusp_2_degree REAL, cusp_3_degree REAL,
    cusp_4_degree REAL, cusp_5_degree REAL, cusp_6_degree REAL,
    cusp_7_degree REAL, cusp_8_degree REAL, cusp_9_degree REAL,
    cusp_10_degree REAL, cusp_11_degree REAL, cusp_12_degree REAL,
    
    -- House cuspal sub lords
    cusp_1_sub_lord TEXT, cusp_2_sub_lord TEXT, cusp_3_sub_lord TEXT,
    cusp_4_sub_lord TEXT, cusp_5_sub_lord TEXT, cusp_6_sub_lord TEXT,
    cusp_7_sub_lord TEXT, cusp_8_sub_lord TEXT, cusp_9_sub_lord TEXT,
    cusp_10_sub_lord TEXT, cusp_11_sub_lord TEXT, cusp_12_sub_lord TEXT,
    
    -- Planetary positions
    sun_longitude REAL, moon_longitude REAL, mars_longitude REAL,
    mercury_longitude REAL, jupiter_longitude REAL, venus_longitude REAL,
    saturn_longitude REAL, rahu_longitude REAL, ketu_longitude REAL,
    
    -- Planetary house positions
    sun_house INTEGER, moon_house INTEGER, mars_house INTEGER,
    mercury_house INTEGER, jupiter_house INTEGER, venus_house INTEGER,
    saturn_house INTEGER, rahu_house INTEGER, ketu_house INTEGER,
    
    -- Retrograde status
    sun_retrograde BOOLEAN, moon_retrograde BOOLEAN, mars_retrograde BOOLEAN,
    mercury_retrograde BOOLEAN, jupiter_retrograde BOOLEAN, venus_retrograde BOOLEAN,
    saturn_retrograde BOOLEAN, rahu_retrograde BOOLEAN, ketu_retrograde BOOLEAN,
    
    -- Combustion status
    moon_combust BOOLEAN, mars_combust BOOLEAN, mercury_combust BOOLEAN,
    jupiter_combust BOOLEAN, venus_combust BOOLEAN, saturn_combust BOOLEAN,
    
    -- Moon hierarchical data
    moon_sign TEXT, moon_sign_lord TEXT, moon_nakshatra TEXT,
    moon_star_lord TEXT, moon_sub_lord TEXT, moon_sub_sub_lord TEXT,
    
    -- Additional data
    day_lord TEXT,                          -- Day ruling planet
    muhurta_strength_score REAL,            -- Overall muhurta strength
    ascendant_favorability REAL,            -- Ascendant team favorability
    descendant_favorability REAL            -- Descendant team favorability
)
```

#### **6. raw_planetary_data**
```sql
CREATE TABLE raw_planetary_data (
    delivery_id INTEGER PRIMARY KEY,       -- Foreign key to deliveries
    match_id TEXT,                         -- Foreign key to matches
    delivery_timestamp TEXT,               -- Delivery timestamp
    
    -- Moon specific data
    moon_longitude REAL, moon_nakshatra TEXT, moon_pada INTEGER,
    moon_sub_lord TEXT, moon_sub_sub_lord TEXT, moon_sign TEXT,
    moon_sign_lord TEXT, moon_star_lord TEXT,
    
    -- Ascendant data
    ascendant_degree REAL, ascendant_sign TEXT, ascendant_lord TEXT,
    
    -- All planetary longitudes
    sun_longitude REAL, moon_longitude_dup REAL, mars_longitude REAL,
    mercury_longitude REAL, jupiter_longitude REAL, venus_longitude REAL,
    saturn_longitude REAL, rahu_longitude REAL, ketu_longitude REAL,
    
    -- Planetary signs
    sun_sign TEXT, mars_sign TEXT, mercury_sign TEXT, jupiter_sign TEXT,
    venus_sign TEXT, saturn_sign TEXT, rahu_sign TEXT, ketu_sign TEXT,
    
    -- Retrograde status
    sun_retrograde BOOLEAN, moon_retrograde BOOLEAN, mars_retrograde BOOLEAN,
    mercury_retrograde BOOLEAN, jupiter_retrograde BOOLEAN, venus_retrograde BOOLEAN,
    saturn_retrograde BOOLEAN, rahu_retrograde BOOLEAN, ketu_retrograde BOOLEAN,
    
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id)
)
```

#### **7. house_data**
```sql
CREATE TABLE house_data (
    delivery_id INTEGER PRIMARY KEY,       -- Foreign key to deliveries
    
    -- House cusps (degrees)
    house_1_cusp REAL, house_2_cusp REAL, house_3_cusp REAL,
    house_4_cusp REAL, house_5_cusp REAL, house_6_cusp REAL,
    house_7_cusp REAL, house_8_cusp REAL, house_9_cusp REAL,
    house_10_cusp REAL, house_11_cusp REAL, house_12_cusp REAL,
    
    -- House signs
    house_1_sign TEXT, house_2_sign TEXT, house_3_sign TEXT,
    house_4_sign TEXT, house_5_sign TEXT, house_6_sign TEXT,
    house_7_sign TEXT, house_8_sign TEXT, house_9_sign TEXT,
    house_10_sign TEXT, house_11_sign TEXT, house_12_sign TEXT,
    
    -- House lords
    house_1_lord TEXT, house_2_lord TEXT, house_3_lord TEXT,
    house_4_lord TEXT, house_5_lord TEXT, house_6_lord TEXT,
    house_7_lord TEXT, house_8_lord TEXT, house_9_lord TEXT,
    house_10_lord TEXT, house_11_lord TEXT, house_12_lord TEXT,
    
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id)
)
```

#### **8. kp_cuspal_sub_lords**
```sql
CREATE TABLE kp_cuspal_sub_lords (
    delivery_id INTEGER PRIMARY KEY,       -- Foreign key to deliveries
    
    -- Cuspal sub lords for all 12 houses
    cusp_1_sub_lord TEXT, cusp_2_sub_lord TEXT, cusp_3_sub_lord TEXT,
    cusp_4_sub_lord TEXT, cusp_5_sub_lord TEXT, cusp_6_sub_lord TEXT,
    cusp_7_sub_lord TEXT, cusp_8_sub_lord TEXT, cusp_9_sub_lord TEXT,
    cusp_10_sub_lord TEXT, cusp_11_sub_lord TEXT, cusp_12_sub_lord TEXT,
    
    -- Cuspal sub-sub lords
    cusp_1_sub_sub_lord TEXT, cusp_2_sub_sub_lord TEXT, cusp_3_sub_sub_lord TEXT,
    cusp_4_sub_sub_lord TEXT, cusp_5_sub_sub_lord TEXT, cusp_6_sub_sub_lord TEXT,
    cusp_7_sub_sub_lord TEXT, cusp_8_sub_sub_lord TEXT, cusp_9_sub_sub_lord TEXT,
    cusp_10_sub_sub_lord TEXT, cusp_11_sub_sub_lord TEXT, cusp_12_sub_sub_lord TEXT,
    
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id)
)
```

#### **9. planetary_aspects**
```sql
CREATE TABLE planetary_aspects (
    delivery_id INTEGER PRIMARY KEY,       -- Foreign key to deliveries
    
    -- Aspect types (JSON arrays of planet pairs)
    conjunctions TEXT,                     -- Planets within 5° of each other
    oppositions TEXT,                      -- Planets 180° apart (±5°)
    trines TEXT,                          -- Planets 120° apart (±5°)
    squares TEXT,                         -- Planets 90° apart (±5°)
    sextiles TEXT,                        -- Planets 60° apart (±5°)
    
    -- Aspect strength scores
    total_aspect_strength REAL,           -- Combined aspect strength
    positive_aspect_strength REAL,        -- Beneficial aspects strength
    negative_aspect_strength REAL,        -- Challenging aspects strength
    
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id)
)
```

#### **10. kp_dasha_data**
```sql
CREATE TABLE kp_dasha_data (
    delivery_id INTEGER PRIMARY KEY,       -- Foreign key to deliveries
    
    -- Current dasha lords
    mahadasha_lord TEXT,                   -- Major period lord
    antardasha_lord TEXT,                  -- Sub period lord
    pratyantardasha_lord TEXT,             -- Sub-sub period lord
    
    -- Dasha balance (years remaining)
    mahadasha_balance REAL,                -- Years left in major period
    antardasha_balance REAL,               -- Years left in sub period
    pratyantardasha_balance REAL,          -- Years left in sub-sub period
    
    -- Dasha significators (JSON arrays)
    mahadasha_significators TEXT,          -- Houses signified by major lord
    antardasha_significators TEXT,         -- Houses signified by sub lord
    pratyantardasha_significators TEXT,    -- Houses signified by sub-sub lord
    
    FOREIGN KEY (delivery_id) REFERENCES deliveries(id)
)
```

---

## 🔧 CRITICAL FIXES NEEDED

### **1. Database Schema Alignment**
```sql
-- Add missing columns to existing tables
ALTER TABLE astrological_predictions ADD COLUMN favorability_score REAL;
ALTER TABLE astrological_predictions ADD COLUMN ascendant_strength REAL;
ALTER TABLE astrological_predictions ADD COLUMN descendant_strength REAL;

-- Add team information to deliveries table
ALTER TABLE deliveries ADD COLUMN team1 TEXT;
ALTER TABLE deliveries ADD COLUMN team2 TEXT;
ALTER TABLE deliveries ADD COLUMN batting_team TEXT;
```

### **2. Variable Name Standardization**
- **Standardize on `inning` (singular)** throughout codebase
- **Use `match_id` consistently** as primary key reference
- **Use `start_datetime` instead of `date`** for match timing

### **3. Error Handling Improvements**
```python
# Always check for chart generation errors
if 'error' in chart:
    print(f"Chart generation failed: {chart['error']}")
    return None

# Always verify DataFrame loading
if nakshatra_df is None or nakshatra_df.empty:
    print("Nakshatra data not loaded properly")
    return None

# Check required columns exist
required_columns = ['Start_Degree', 'End_Degree', 'Sub_Lord']
if not all(col in nakshatra_df.columns for col in required_columns):
    print(f"Missing required columns in nakshatra data")
    return None
```

### **4. Import Path Fixes**
```python
# Standardize import pattern across all scripts
import os
import sys
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
ROOT_DIR = os.path.dirname(scripts_dir)
sys.path.insert(0, ROOT_DIR)
```

---

## 📋 CONFIGURATION FILES

### **nakshatra_sub_lords_longitudes.csv**
```csv
Start_Degree,End_Degree,Nakshatra,Pada,Sub_Lord,Sub_Sub_Lord,Star_Lord,Sign,Sign_Lord
0.0000,0.0411,Ashwini,1,Ketu,Ketu,Ketu,Aries,Mars
0.0411,0.0823,Ashwini,1,Ketu,Venus,Ketu,Aries,Mars
...
```
**Required Columns:**
- `Start_Degree` (REAL): Starting degree of sub-division
- `End_Degree` (REAL): Ending degree of sub-division  
- `Nakshatra` (TEXT): Nakshatra name
- `Pada` (INTEGER): Pada number (1-4)
- `Sub_Lord` (TEXT): KP sub lord planet
- `Sub_Sub_Lord` (TEXT): KP sub-sub lord planet
- `Star_Lord` (TEXT): Nakshatra ruling planet
- `Sign` (TEXT): Zodiac sign name
- `Sign_Lord` (TEXT): Sign ruling planet

---

## 🎯 RECOMMENDED ACTIONS

### **Immediate Fixes (Priority 1)**
1. **Fix nakshatra DataFrame loading** in all chart generation functions
2. **Add error checking** for chart generation results
3. **Standardize database column references** across all scripts
4. **Add missing database columns** or update code to use existing ones

### **Medium Priority (Priority 2)**
1. **Implement comprehensive error logging** system
2. **Create database migration scripts** for schema updates
3. **Add data validation** for all database operations
4. **Standardize import paths** across all modules

### **Long Term (Priority 3)**
1. **Complete enhanced table implementation**
2. **Add comprehensive unit tests** for all database operations
3. **Create automated schema validation** tools
4. **Implement database backup and recovery** procedures

---

## 🔍 TESTING CHECKLIST

### **Database Connectivity**
- [ ] All tables exist and are accessible
- [ ] All required columns are present
- [ ] Foreign key relationships work correctly
- [ ] Data types match expectations

### **Chart Generation**
- [ ] Nakshatra data loads correctly
- [ ] Chart generation handles edge cases
- [ ] Error messages are properly formatted
- [ ] All planetary calculations work

### **Timeline Prediction**
- [ ] Astrological periods generate correctly
- [ ] Probability calculations work
- [ ] Database queries return expected results
- [ ] Team assignment validation functions

This comprehensive analysis should prevent future KeyError issues and provide a clear roadmap for system improvements. 