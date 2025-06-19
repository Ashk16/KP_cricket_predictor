#!/usr/bin/env python3
"""
Test All Level 2 Contradictions
"""

import sys
sys.path.append('.')
from scripts.unified_kp_predictor import UnifiedKPPredictor

def test_all_level2_contradictions():
    """Test all Level 2 contradictions to verify they apply negative weights correctly"""
    
    predictor = UnifiedKPPredictor('comprehensive')
    team_a, team_b = 'Panthers', 'Nellai'
    match_date, start_time = '2025-06-18', '19:51:00'
    lat, lon = 11.6469616, 78.2106958

    print('🔍 TESTING ALL LEVEL 2 CONTRADICTIONS')
    print('=' * 60)
    
    # Level 2 contradiction patterns
    level2_patterns = ['Mars-Rahu', 'Mars-Saturn', 'Sun-Saturn', 'Jupiter-Rahu']

    try:
        # Generate longer timeline to find more cases
        df = predictor.predict_match_timeline(team_a, team_b, match_date, start_time, lat, lon, 5)
        print(f'✅ Generated {len(df)} predictions')
        
        # Find all Level 2 contradiction cases
        level2_cases = {}
        for pattern in level2_patterns:
            cases = []
            for i, row in df.iterrows():
                contradictions = str(row.contradictions)
                if pattern in contradictions or f'{pattern.split("-")[1]}-{pattern.split("-")[0]}' in contradictions:
                    cases.append(row)
            level2_cases[pattern] = cases
            print(f'🎯 {pattern}: {len(cases)} cases found')
        
        # Analyze each contradiction type
        for pattern, cases in level2_cases.items():
            if cases:
                print(f'\n--- {pattern.upper()} CONTRADICTION ANALYSIS ---')
                planet1, planet2 = pattern.split('-')
                
                for i, row in enumerate(cases[:2]):  # Show first 2 cases
                    print(f'\nCase {i+1}: {row.moon_star_lord}-{row.sub_lord}-{row.sub_sub_lord}')
                    print(f'Individual Scores: {row.moon_sl_score:.3f}, {row.moon_sub_score:.3f}, {row.moon_ssl_score:.3f}')
                    print(f'Weights: {row.star_lord_weight:.3f}, {row.sub_lord_weight:.3f}, {row.sub_sub_lord_weight:.3f}')
                    print(f'Final Score: {row.final_score:.3f}')
                    
                    # Check which planets got negative weights
                    negative_weights = []
                    if row.star_lord_weight < 0:
                        negative_weights.append(f'{row.moon_star_lord}(SL)')
                    if row.sub_lord_weight < 0:
                        negative_weights.append(f'{row.sub_lord}(Sub)')
                    if row.sub_sub_lord_weight < 0:
                        negative_weights.append(f'{row.sub_sub_lord}(SSL)')
                    
                    print(f'Negative weights: {", ".join(negative_weights) if negative_weights else "None"}')
                    
                    # Verify both conflicting planets got negative weights
                    planet1_negative = (
                        (row.moon_star_lord == planet1 and row.star_lord_weight < 0) or
                        (row.sub_lord == planet1 and row.sub_lord_weight < 0) or
                        (row.sub_sub_lord == planet1 and row.sub_sub_lord_weight < 0)
                    )
                    planet2_negative = (
                        (row.moon_star_lord == planet2 and row.star_lord_weight < 0) or
                        (row.sub_lord == planet2 and row.sub_lord_weight < 0) or
                        (row.sub_sub_lord == planet2 and row.sub_sub_lord_weight < 0)
                    )
                    
                    print(f'✅ Level 2 Effect: {planet1} negative: {planet1_negative}, {planet2} negative: {planet2_negative}')
                    
                    # Astrological interpretation
                    if row.final_score < 0:
                        print('🎯 STRONG MALEFIC: Favors Descendant (negative score)')
                    elif row.final_score < 3:
                        print('⚠️ MALEFIC EFFECT: Significantly reduced score')
                    else:
                        print('❓ MILD EFFECT: Other positive influences dominate')
        
        # Summary statistics
        print(f'\n📊 LEVEL 2 CONTRADICTION SUMMARY:')
        total_level2 = sum(len(cases) for cases in level2_cases.values())
        descendant_favoring = 0
        reduced_impact = 0
        
        for cases in level2_cases.values():
            for case in cases:
                if case.final_score < 0:
                    descendant_favoring += 1
                elif case.final_score < 3:
                    reduced_impact += 1
        
        print(f'Total Level 2 cases: {total_level2}')
        print(f'Cases favoring Descendant: {descendant_favoring}')
        print(f'Cases with reduced impact: {reduced_impact}')
        print(f'Effectiveness: {(descendant_favoring + reduced_impact)}/{total_level2} cases show malefic effect')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_level2_contradictions() 