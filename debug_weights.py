#!/usr/bin/env python3
"""
Debug Weight Calculations Step by Step
"""

import sys
sys.path.append('.')
from scripts.unified_kp_predictor import UnifiedKPPredictor

def debug_weights():
    """Debug weight calculations in detail"""
    
    predictor = UnifiedKPPredictor('comprehensive')
    team_a, team_b = 'Panthers', 'Nellai'
    match_date, start_time = '2025-06-18', '19:51:00'
    lat, lon = 11.6469616, 78.2106958

    print('🔧 DETAILED WEIGHT DEBUG')
    print('=' * 50)

    try:
        # Generate just one prediction
        df = predictor.predict_match_timeline(team_a, team_b, match_date, start_time, lat, lon, 0.1)
        
        if len(df) > 0:
            row = df.iloc[0]
            star_lord = row.moon_star_lord
            sub_lord = row.sub_lord
            sub_sub_lord = row.sub_sub_lord
            contradictions = str(row.contradictions).split(',') if row.contradictions else []
            
            print(f'Lords: SL={star_lord}, Sub={sub_lord}, SSL={sub_sub_lord}')
            print(f'Contradictions: {contradictions}')
            print(f'Final weights: SL={row.star_lord_weight:.3f}, Sub={row.sub_lord_weight:.3f}, SSL={row.sub_sub_lord_weight:.3f}')
            
            # Manual Level 2 check
            level2_contradictions = ['Mars-Rahu', 'Rahu-Mars', 'Mars-Saturn', 'Saturn-Mars', 
                                   'Sun-Saturn', 'Saturn-Sun', 'Jupiter-Rahu', 'Rahu-Jupiter']
            
            level2_active = any(any(l2 in c for l2 in level2_contradictions) for c in contradictions)
            print(f'\nLevel 2 active: {level2_active}')
            
            if level2_active:
                print('\n🔴 LEVEL 2 CONTRADICTIONS FOUND!')
                
                # Simulate the logic
                weights = {'star_lord': row.star_lord_weight, 'sub_lord': row.sub_lord_weight, 'sub_sub_lord': row.sub_sub_lord_weight}
                print(f'Original weights: {weights}')
                
                for contradiction in contradictions:
                    if contradiction.strip() and any(l2 in contradiction for l2 in level2_contradictions):
                        print(f'\nProcessing Level 2: "{contradiction.strip()}"')
                        planets = contradiction.strip().split('-')
                        
                        if len(planets) == 2:
                            planet1, planet2 = planets
                            print(f'Conflict planets: {planet1}, {planet2}')
                            
                            # Check each position
                            print(f'\nChecking positions:')
                            print(f'Star Lord ({star_lord}) == {planet1}? {star_lord == planet1}')
                            print(f'Sub Lord ({sub_lord}) == {planet1}? {sub_lord == planet1}')
                            print(f'Sub Sub Lord ({sub_sub_lord}) == {planet1}? {sub_sub_lord == planet1}')
                            print(f'Star Lord ({star_lord}) == {planet2}? {star_lord == planet2}')
                            print(f'Sub Lord ({sub_lord}) == {planet2}? {sub_lord == planet2}')
                            print(f'Sub Sub Lord ({sub_sub_lord}) == {planet2}? {sub_sub_lord == planet2}')
                            
                            # Apply negative weights manually
                            if star_lord == planet1:
                                weights['star_lord'] *= -1
                                print(f'Applied negative to Star Lord: {weights["star_lord"]}')
                            if sub_lord == planet1:
                                weights['sub_lord'] *= -1
                                print(f'Applied negative to Sub Lord: {weights["sub_lord"]}')
                            if sub_sub_lord == planet1:
                                weights['sub_sub_lord'] *= -1
                                print(f'Applied negative to Sub Sub Lord: {weights["sub_sub_lord"]}')
                                
                            if star_lord == planet2:
                                weights['star_lord'] *= -1
                                print(f'Applied negative to Star Lord: {weights["star_lord"]}')
                            if sub_lord == planet2:
                                weights['sub_lord'] *= -1
                                print(f'Applied negative to Sub Lord: {weights["sub_lord"]}')
                            if sub_sub_lord == planet2:
                                weights['sub_sub_lord'] *= -1
                                print(f'Applied negative to Sub Sub Lord: {weights["sub_sub_lord"]}')
                
                print(f'\nExpected final weights: {weights}')
                print(f'Actual final weights: SL={row.star_lord_weight:.3f}, Sub={row.sub_lord_weight:.3f}, SSL={row.sub_sub_lord_weight:.3f}')
                
                # Check if any are negative
                negative_weights = []
                if row.star_lord_weight < 0:
                    negative_weights.append(f'SL({row.star_lord_weight:.3f})')
                if row.sub_lord_weight < 0:
                    negative_weights.append(f'Sub({row.sub_lord_weight:.3f})')
                if row.sub_sub_lord_weight < 0:
                    negative_weights.append(f'SSL({row.sub_sub_lord_weight:.3f})')
                
                print(f'\nNegative weights found: {negative_weights if negative_weights else "None"}')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_weights() 