#!/usr/bin/env python3
"""
Debug Level 2 Contradiction Implementation
"""

import sys
sys.path.append('.')
from scripts.unified_kp_predictor import UnifiedKPPredictor

def debug_contradiction():
    """Debug why Level 2 contradictions aren't working"""
    
    predictor = UnifiedKPPredictor('comprehensive')
    team_a, team_b = 'Panthers', 'Nellai'
    match_date, start_time = '2025-06-18', '19:51:00'
    lat, lon = 11.6469616, 78.2106958

    print('🔧 DEBUGGING LEVEL 2 CONTRADICTIONS')
    print('=' * 50)

    try:
        # Generate just one prediction
        df = predictor.predict_match_timeline(team_a, team_b, match_date, start_time, lat, lon, 0.1)
        
        if len(df) > 0:
            row = df.iloc[0]
            print(f'Sample Entry: {row.moon_star_lord}-{row.sub_lord}-{row.sub_sub_lord}')
            print(f'Contradictions detected: "{row.contradictions}"')
            print(f'Weights: SL={row.star_lord_weight:.3f}, Sub={row.sub_lord_weight:.3f}, SSL={row.sub_sub_lord_weight:.3f}')
            
            # Manual check of Level 2 logic
            contradictions = str(row.contradictions).split(',') if row.contradictions else []
            level2_contradictions = ['Mars-Rahu', 'Rahu-Mars', 'Mars-Saturn', 'Saturn-Mars', 
                                   'Sun-Saturn', 'Saturn-Sun', 'Jupiter-Rahu', 'Rahu-Jupiter']
            
            print(f'\\nContradiction list: {contradictions}')
            print(f'Level 2 patterns: {level2_contradictions}')
            
            # Check if any Level 2 is active
            level2_active = any(any(l2 in c for l2 in level2_contradictions) for c in contradictions)
            print(f'Level 2 active: {level2_active}')
            
            # Check each contradiction
            for contradiction in contradictions:
                if contradiction.strip():
                    print(f'\\nChecking contradiction: "{contradiction.strip()}"')
                    is_level2 = any(l2 in contradiction for l2 in level2_contradictions)
                    print(f'Is Level 2: {is_level2}')
                    
                    if is_level2:
                        planets = contradiction.strip().split('-')
                        print(f'Planets in contradiction: {planets}')
                        if len(planets) == 2:
                            planet1, planet2 = planets
                            print(f'Would apply negative to: {planet1} and {planet2}')
                            print(f'Current lords: SL={row.moon_star_lord}, Sub={row.sub_lord}, SSL={row.sub_sub_lord}')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_contradiction() 