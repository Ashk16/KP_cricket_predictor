# KP Cricket Predictor - Project Documentation

## 🎯 **Project Overview**

The KP Cricket Predictor is an advanced astrological prediction system that combines traditional Krishnamurti Paddhati (KP) astrology with modern machine learning to predict cricket match dynamics, team favorability, and match outcomes. The system uses authentic astrological charts generated from match start times (Muhurta charts) and provides real-time timeline predictions with probability scores.

**Key Innovation**: Unlike traditional fixed-time predictions, our system uses **dynamic astrological periods** based on actual sub lord changes, ensuring authentic KP methodology while providing precise temporal predictions.

---

## 🌟 **Core System Components**

### **1. Live Prediction Engine**
- **File**: `scripts/live_predictor.py`
- **Purpose**: Main prediction interface with multi-target ML models
- **Key Features**:
  - Dynamic astrological periods based on planetary movements (15-90 minutes)
  - Multi-target predictions: KP favorability, high scoring, wicket pressure, momentum shift
  - Team favorability analysis (Ascendant/Descendant)
  - Real-time timeline generation with comprehensive reporting

### **2. Chart Generation System**
- **File**: `scripts/chart_generator.py`
- **Purpose**: Creates precise KP astrological charts using Swiss Ephemeris
- **Key Features**:
  - Planetary positions with retrograde and combustion analysis
  - House cusps calculation using Placidus system
  - Nakshatra mapping with sub lord and sub-sub lord determination
  - 249-point KP mapping system

### **3. Favorability Rules Engine**
- **File**: `scripts/kp_favorability_rules.py`
- **Purpose**: Implements traditional KP astrological principles for cricket prediction
- **Key Features**:
  - House significance analysis (1st, 5th, 6th, 7th, 8th, 10th, 11th, 12th houses)
  - Planetary strength calculation with benefic/malefic classifications
  - Hierarchical weighting: Cuspal Sub Lord (60%), Planet in House (25%), Aspects (15%)
  - Combustion and dignity analysis

### **4. Machine Learning Pipeline**
- **Files**: `scripts/predictive_ml_training.py`, `scripts/ml_data_preparation.py`
- **Purpose**: Complete ML pipeline for multi-target prediction
- **Key Features**:
  - Pre-match feature engineering (Time + Location + KP only)
  - Multi-target training: KP prediction, high scoring, wicket pressure, momentum
  - 216,859 astrological periods from 3,992 matches
  - 79.3% accuracy for KP predictions, 77.8% for wicket pressure

### **5. Astrological Data Processing**
- **File**: `scripts/astro_processor.py`
- **Purpose**: Processes delivery-level astrological data
- **Key Features**:
  - Ball-by-ball astrological analysis with KP calculations
  - Batch processing with error handling and progress tracking
  - Database integration with comprehensive validation
  - Coordinate-based chart generation for accurate predictions

### **6. Database Management**
- **Files**: `scripts/db_manager.py`, `scripts/update_final_scores.py`, `scripts/verify_score_update.py`
- **Purpose**: Database operations and score management
- **Key Features**:
  - Database initialization and schema management
  - Score updates with authentic KP methodology
  - Data verification and integrity checking
  - Comprehensive error handling and logging

### **7. Web Interface**
- **Files**: `app/app.py`, `app/live_predictor_app.py`
- **Purpose**: Interactive web applications for predictions
- **Key Features**:
  - Streamlit-based user interfaces
  - Real-time timeline prediction with visualizations
  - Interactive charts and comprehensive reporting
  - Multi-match prediction management

---

## 🔮 **KP Astrological Methodology**

### **Traditional KP Principles Applied**

#### **House Significances for Cricket**
- **1st House (Ascendant)**: Home team, overall strength, match vitality
- **5th House**: Sports, games, entertainment, creative expression
- **6th House**: Competition, opponents, daily struggles, service
- **7th House (Descendant)**: Away team, partnerships, open enemies
- **8th House**: Obstacles, wickets, sudden changes, defeats, transformations
- **10th House**: Success, achievement, reputation, career, status
- **11th House**: Gains, profits, fulfillment of desires, victory
- **12th House**: Losses, expenses, hidden enemies, endings, foreign influence

#### **Planetary Classifications**
- **Benefics**: Jupiter (wisdom, expansion), Venus (harmony, beauty)
- **Malefics**: Saturn (obstacles, delays), Mars (aggression, conflict), Sun (authority, ego), Rahu/Ketu (karmic nodes, sudden events)
- **Neutrals**: Moon (mind, emotions), Mercury (communication, intellect)

#### **KP Hierarchy (Importance Order)**
1. **Cuspal Sub Lord** (60% weight) - Most important
2. **Planet in House** (25% weight) - Secondary influence
3. **Aspecting Planets** (15% weight) - Tertiary influence

#### **Combustion Analysis**
Planets lose strength when too close to Sun:
- Moon: 12°, Mars: 17°, Mercury: 14°
- Jupiter: 11°, Venus: 10°, Saturn: 15°

---

## 📊 **Database Architecture**

### **Core Tables**
- **matches**: Match metadata with start times and team information
- **deliveries**: Ball-by-ball data with precise timestamps
- **astrological_predictions**: KP predictions with favorability scores
- **chart_data**: Astrological chart information for each delivery
- **muhurta_charts**: Match start charts with planetary positions

### **Key Features**
- **Schema Validation**: Built-in validation prevents KeyError issues
- **Timestamp Precision**: Millisecond-accurate for astrological calculations
- **Error Handling**: Comprehensive error recovery and logging
- **Data Integrity**: Foreign key relationships and constraint validation

---

## 🚀 **System Workflow**

### **1. Match Prediction Workflow**
```
Match Start Time → Swiss Ephemeris Chart → KP Analysis → 
Period Calculation → Favorability Scoring → Probability Analysis → 
Timeline Generation → User Interface Display
```

### **2. Training Data Workflow**
```
Historical Matches → Timestamp Enrichment → Chart Generation → 
KP Feature Engineering → Period-based Aggregation → 
ML Model Training → Validation & Testing → Model Deployment
```

### **3. Validation Workflow**
```
Historical Performance → Team Assignment Analysis → 
Confidence Scoring → Assignment Recommendation → 
Statistical Validation → User Notification
```

---

## 📈 **Performance Metrics**

### **Current System Status**
- ✅ **Database**: 3,558 matches, 820,620 deliveries processed
- ✅ **Chart Generation**: 100% success rate with error handling
- ✅ **Timeline Prediction**: Dynamic periods with 4 probability scores
- ✅ **Validation**: Automated team assignment with confidence scoring
- ✅ **Error Handling**: Comprehensive validation and recovery mechanisms

### **Prediction Accuracy Targets**
- **Timeline Favorability**: >65% accuracy in period predictions
- **Team Assignment**: >75% confidence in ascendant/descendant validation
- **Probability Scores**: Calibrated to actual match dynamics
- **Period Duration**: 15-90 minute authentic astrological periods

---

## 🔧 **Configuration & Customization**

### **Key Configuration Files**
- `config/nakshatra_sub_lords_longitudes.csv`: 249-point KP mapping
- `training_analysis/cricket_predictions.db`: Main SQLite database
- Default coordinates: Mumbai (19.0760°N, 72.8777°E)

### **Customizable Parameters**
- **Location**: Venue-specific coordinates for accurate chart generation
- **Period Duration**: Timeline length (default: 5 hours)
- **Probability Thresholds**: Customizable scoring thresholds
- **Model Parameters**: ML model hyperparameters and feature weights

---

## 🧪 **Testing & Validation Framework**

### **Automated Testing**
- **Schema Validation**: Database structure verification
- **Chart Generation**: Swiss Ephemeris calculation testing
- **Period Calculation**: Sub lord change timing validation
- **Probability Scoring**: Score range and calibration testing

### **Manual Validation**
- **Historical Backtesting**: Performance against known match outcomes
- **Expert Review**: KP astrology principle compliance
- **User Acceptance**: Interface usability and prediction clarity

---

## 🔄 **Development Cycle**

### **Phase 1: Data Collection & Processing**
1. Historical match data ingestion
2. Timestamp enrichment and validation
3. Chart generation for all deliveries
4. Database population with error handling

### **Phase 2: Model Development & Training**
1. KP feature engineering with period weighting
2. ML model training using astrological periods
3. Team assignment validation and correction
4. Model evaluation and hyperparameter tuning

### **Phase 3: Prediction System Deployment**
1. Timeline prediction engine implementation
2. Probability scoring system development
3. User interface integration
4. Real-time prediction capability

### **Phase 4: Continuous Improvement**
1. Performance monitoring and analysis
2. Rule refinement based on statistical feedback
3. Model retraining with new data
4. Feature enhancement and optimization

---

## 📚 **Documentation References**

- **DATABASE_SCHEMA_DICTIONARY.md**: Complete database schema reference
- **CODEBASE_ANALYSIS_AND_DATA_DICTIONARY.md**: Variable naming and error prevention
- **ENHANCED_KP_SYSTEM_SUMMARY.md**: Technical implementation details
- **README.md**: User guide and quick start instructions

---

## 🛠️ **Technical Stack**

- **Core Language**: Python 3.8+
- **Astronomical Calculations**: Swiss Ephemeris (pyswisseph)
- **Database**: SQLite with comprehensive schema
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow
- **Data Processing**: Pandas, NumPy
- **Web Interface**: Flask with responsive design
- **Version Control**: Git with comprehensive history

---

## 🎯 **Future Enhancements**

### **Short-term Goals**
- Real-time match integration with live data feeds
- Mobile-responsive web interface
- Advanced visualization of astrological periods
- API development for third-party integration

### **Long-term Vision**
- Multi-sport astrological prediction system
- Advanced ML models with deep learning
- Real-time chart updates during matches
- Community-driven rule validation and improvement

---

## 📞 **Support & Contribution**

### **Getting Help**
- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Comprehensive guides in project documentation
- **Community**: Join discussions for feature requests and improvements

### **Contributing**
- **Code**: Follow established patterns in `kp_timeline_predictor_fixed.py`
- **Documentation**: Update relevant documentation files
- **Testing**: Add comprehensive tests for new features
- **Validation**: Ensure KP astrological principle compliance

---

**🏏 "Where ancient wisdom meets modern technology for cricket prediction" 🔮** 