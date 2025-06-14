# KP Cricket Predictor - Database Schema Dictionary

## 📊 COMPLETE DATABASE SCHEMA REFERENCE

This document provides a comprehensive reference for all database tables, columns, and their purposes in the KP Cricket Predictor system. Use this to avoid KeyError issues and ensure consistent database operations.

---

## 🚨 CRITICAL ERRORS IDENTIFIED & FIXED

### **1. Variable Naming Inconsistencies**
- **deliveries table**: Uses `inning` (singular) - code should NOT use `innings` (plural)
- **matches table**: Uses `match_id` as primary key - code should NOT use `id`
- **matches table**: Uses `start_datetime` - code should NOT use `date`
- **nakshatra data**: Uses `Start_Degree`, `End_Degree` - code should NOT use lowercase versions

### **2. Chart Generation Errors**
- Always check for `'error'` key in chart results before accessing data
- Verify nakshatra DataFrame is loaded before chart generation
- Use `.get()` method with defaults instead of direct dictionary access

---

## 🏗️ CORE TABLES (Currently Implemented)

### **1. matches**
**Primary Key**: `match_id`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `match_id` | TEXT | Unique match identifier (PRIMARY KEY) | "1011497" |
| `start_datetime` | TEXT | Match start time (ISO format) | "2023-10-15 14:30:00" |
| `venue` | TEXT | Match venue name | "Wankhede Stadium" |
| `team1` | TEXT | First team name (Ascendant by default) | "Mumbai Indians" |
| `team2` | TEXT | Second team name (Descendant by default) | "Chennai Super Kings" |
| `winner` | TEXT | Winning team name | "Mumbai Indians" |
| `total_overs` | INTEGER | Total overs in match | 20 |
| `processed_at` | TEXT | Processing timestamp | "2023-10-15 16:45:00" |

### **2. deliveries**
**Primary Key**: `id` (auto-increment)
**Foreign Key**: `match_id` → matches.match_id

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | INTEGER | Unique delivery ID (PRIMARY KEY) | 12345 |
| `match_id` | TEXT | Foreign key to matches table | "1011497" |
| `inning` | INTEGER | Inning number (1 or 2) | 1 |
| `over` | INTEGER | Over number (0-based) | 5 |
| `ball` | INTEGER | Ball number (1-6) | 3 |
| `batsman` | TEXT | Batsman name | "Rohit Sharma" |
| `bowler` | TEXT | Bowler name | "Jasprit Bumrah" |
| `runs_off_bat` | INTEGER | Runs scored off bat | 4 |
| `extras` | INTEGER | Extra runs (wides, byes, etc.) | 0 |
| `wicket_type` | TEXT | Type of dismissal | "caught" |
| `dismissal` | TEXT | Dismissed player name | "Rohit Sharma" |
| `timestamp` | TEXT | Delivery timestamp (ISO format) | "2023-10-15 15:23:45" |

**⚠️ CRITICAL**: Uses `inning` (singular), NOT `innings` (plural)

### **3. astrological_predictions**
**Primary Key**: `id` (auto-increment)
**Foreign Keys**: `delivery_id` → deliveries.id, `match_id` → matches.match_id

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | INTEGER | Unique prediction ID (PRIMARY KEY) | 67890 |
| `delivery_id` | INTEGER | Foreign key to deliveries table | 12345 |
| `match_id` | TEXT | Foreign key to matches table | "1011497" |
| `asc_score` | REAL | Ascendant favorability score | 0.234 |
| `desc_score` | REAL | Descendant favorability score | -0.156 |
| `moon_sl` | TEXT | Moon sub lord planet | "Jupiter" |
| `moon_sub` | TEXT | Moon sub planet | "Venus" |
| `moon_ssl` | TEXT | Moon sub-sub lord planet | "Mercury" |
| `moon_sl_score` | REAL | Moon sub lord score | 0.123 |
| `moon_sub_score` | REAL | Moon sub score | -0.045 |
| `moon_ssl_score` | REAL | Moon sub-sub lord score | 0.078 |
| `moon_sl_houses` | TEXT | JSON: Moon SL house significances | "[1, 5, 9]" |
| `moon_sub_houses` | TEXT | JSON: Moon sub house significances | "[2, 6, 10]" |
| `moon_ssl_houses` | TEXT | JSON: Moon SSL house significances | "[3, 7, 11]" |
| `moon_sl_star_lord` | TEXT | Moon SL star lord | "Jupiter" |
| `moon_sub_star_lord` | TEXT | Moon sub star lord | "Venus" |
| `moon_ssl_star_lord` | TEXT | Moon SSL star lord | "Mercury" |
| `ruling_planets` | TEXT | Ruling planets string | "Jupiter-Venus-Mercury" |
| `success_score` | REAL | Success probability score | 0.678 |
| `predicted_impact` | REAL | Predicted impact value | 0.234 |
| `actual_impact` | REAL | Actual impact value | 0.189 |

### **4. chart_data**
**Primary Key**: `id` (auto-increment)
**Foreign Keys**: `delivery_id` → deliveries.id, `match_id` → matches.match_id

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | INTEGER | Unique chart ID (PRIMARY KEY) | 98765 |
| `delivery_id` | INTEGER | Foreign key to deliveries table | 12345 |
| `match_id` | TEXT | Foreign key to matches table | "1011497" |
| `ascendant_degree` | REAL | Ascendant degree | 123.4567 |
| `moon_longitude` | REAL | Moon longitude in degrees | 234.5678 |
| `moon_nakshatra` | TEXT | Moon nakshatra name | "Rohini" |
| `moon_pada` | INTEGER | Moon pada (1-4) | 2 |
| `moon_sign` | TEXT | Moon zodiac sign | "Taurus" |
| `moon_sign_lord` | TEXT | Moon sign ruling planet | "Venus" |
| `moon_star_lord` | TEXT | Moon nakshatra ruling planet | "Moon" |
| `moon_sub_lord` | TEXT | Moon KP sub lord | "Jupiter" |
| `moon_sub_sub_lord` | TEXT | Moon KP sub-sub lord | "Mercury" |
| `chart_json` | TEXT | Complete chart data as JSON | "{...}" |

---

## 📋 CONFIGURATION FILES

### **nakshatra_sub_lords_longitudes.csv**
**Location**: `config/nakshatra_sub_lords_longitudes.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Start_Degree` | REAL | Starting degree of subdivision | 0.0000 |
| `End_Degree` | REAL | Ending degree of subdivision | 0.0411 |
| `Nakshatra` | TEXT | Nakshatra name | "Ashwini" |
| `Pada` | INTEGER | Pada number (1-4) | 1 |
| `Sub_Lord` | TEXT | KP sub lord planet | "Ketu" |
| `Sub_Sub_Lord` | TEXT | KP sub-sub lord planet | "Ketu" |
| `Star_Lord` | TEXT | Nakshatra ruling planet | "Ketu" |
| `Sign` | TEXT | Zodiac sign name | "Aries" |
| `Sign_Lord` | TEXT | Sign ruling planet | "Mars" |

**⚠️ CRITICAL**: Column names use underscores and proper case: `Start_Degree`, `End_Degree`

---

## 🔧 CORRECT VARIABLE NAMING PATTERNS

### **✅ CORRECT Database References**
```python
# Database columns
match_row['match_id']           # PRIMARY KEY
match_row['start_datetime']     # NOT 'date'
delivery_row['inning']          # NOT 'innings'

# Nakshatra data
nakshatra_df['Start_Degree']    # NOT 'start_degree'
nakshatra_df['End_Degree']      # NOT 'end_degree'
nakshatra_df['Sub_Lord']        # NOT 'sub_lord'
```

### **✅ CORRECT Chart Data Access**
```python
# Always check for errors first
if 'error' in chart:
    print(f"Chart generation failed: {chart['error']}")
    return None

# Use .get() with defaults
planets = chart.get('planets', {})
moon_long = chart.get('moon_longitude', 0)
nakshatra = chart.get('moon_nakshatra', 'Unknown')
```

### **✅ CORRECT Favorability Access**
```python
# Use .get() with defaults
kp_score = favorability.get('final_score', 0)
asc_score = favorability.get('asc_score', 0)
desc_score = favorability.get('desc_score', 0)
```

### **❌ INCORRECT Patterns to Avoid**
```python
# Wrong column names
match_row['id']                 # Should be 'match_id'
match_row['date']               # Should be 'start_datetime'
delivery_row['innings']         # Should be 'inning'

# Wrong case in nakshatra data
nakshatra_df['start_degree']    # Should be 'Start_Degree'
nakshatra_df['end_degree']      # Should be 'End_Degree'

# Direct access without error checking
planets = chart['planets']      # May cause KeyError
kp_score = favorability['final_score']  # May cause KeyError
```

---

## 📈 PROBABILITY SCORES (Enhanced Timeline Predictor)

The fixed timeline predictor includes these probability scores:

| Probability Type | Range | Description |
|------------------|-------|-------------|
| `high_scoring_probability` | 0.0 - 1.0 | Likelihood of high run scoring |
| `collapse_probability` | 0.0 - 1.0 | Likelihood of batting collapse |
| `wicket_pressure_probability` | 0.0 - 1.0 | Likelihood of wicket-taking pressure |
| `momentum_shift_probability` | 0.0 - 1.0 | Likelihood of momentum change |

**✅ CORRECT Usage**:
```python
result = predictor.generate_timeline_with_probabilities(match_id)
if 'error' not in result:
    for period in result['period_predictions']:
        probs = period['dynamics_probabilities']
        high_scoring = probs['high_scoring_probability']
        collapse_risk = probs['collapse_probability']
        
        if high_scoring > 0.6:
            print("High scoring period expected")
        elif collapse_risk > 0.6:
            print("Collapse risk period")
```

---

## 🎯 ERROR PREVENTION CHECKLIST

### **Before Database Operations**
- [ ] Use correct table names: `matches`, `deliveries`, `astrological_predictions`, `chart_data`
- [ ] Use correct column names: `match_id`, `start_datetime`, `inning`
- [ ] Check if table exists: `PRAGMA table_info(table_name)`
- [ ] Handle empty result sets gracefully

### **Before Chart Generation**
- [ ] Verify nakshatra DataFrame is loaded: `if nakshatra_df is not None`
- [ ] Check required columns: `['Start_Degree', 'End_Degree', 'Sub_Lord']`
- [ ] Validate datetime format
- [ ] Always check for `'error'` key in results

### **Before Data Access**
- [ ] Use `.get()` method with default values
- [ ] Check for `'error'` key in dictionaries
- [ ] Validate data types before calculations
- [ ] Handle None/null values

### **Import Path Management**
```python
# ✅ STANDARD PATTERN for all scripts
import os
import sys
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
ROOT_DIR = os.path.dirname(scripts_dir)
sys.path.insert(0, ROOT_DIR)
```

---

## 🔍 TESTING COMMANDS

### **Test Database Schema**
```python
from scripts.kp_timeline_predictor_fixed import KPTimelinePredictorFixed
predictor = KPTimelinePredictorFixed()
validation = predictor.validate_database_schema()
print(validation)
```

### **Test Timeline with Probabilities**
```python
result = predictor.generate_timeline_with_probabilities("1011497", hours=3)
if 'error' not in result:
    print(f"Success: {result['total_periods']} periods generated")
    # Show probability scores
    first_period = result['period_predictions'][0]
    print(first_period['dynamics_probabilities'])
else:
    print(f"Error: {result['error']}")
```

---

This comprehensive data dictionary eliminates KeyError issues and ensures consistent database operations across the entire KP Cricket Predictor system. The fixed timeline predictor (`kp_timeline_predictor_fixed.py`) demonstrates proper error handling and variable naming patterns.
