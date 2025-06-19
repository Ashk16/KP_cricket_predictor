#!/usr/bin/env python3
"""
Test Weight Flow to Find the Bug
"""

import sys
sys.path.append('.')
from scripts.unified_kp_predictor import ComprehensiveModel

def test_weight_flow():
    """Test weight calculation flow step by step"""
    
    # Create the model
    model = ComprehensiveModel()
    
    # Simulate the exact conditions from our debug
    star_lord = 'Mars'
    sub_lord = 'Sun'
    sub_sub_lord = 'Rahu'
    
    # Mock some original scores (doesn't matter for weight calculation)
    original_scores = {
        'star_lord_score': 10.0,
        'sub_lord_score': 15.0,
        'sub_sub_lord_score': 8.0
    }
    
    print('🔧 TESTING WEIGHT CALCULATION FLOW')
    print('=' * 50)
    print(f'Lords: {star_lord}-{sub_lord}-{sub_sub_lord}')
    
    # Step 1: Get contradictions
    contradictions = model.model.detect_contradictions(star_lord, sub_lord, sub_sub_lord)
    print(f'Contradictions: {contradictions}')
    
    # Step 2: Get weights from comprehensive model
    weights_from_comprehensive = model.model.calculate_dynamic_weights(original_scores, contradictions)
    print(f'Weights from comprehensive model: {weights_from_comprehensive}')
    
    # Step 3: Apply Level 2 logic (as done in unified predictor)
    level2_contradictions = ['Mars-Rahu', 'Rahu-Mars', 'Mars-Saturn', 'Saturn-Mars', 
                           'Sun-Saturn', 'Saturn-Sun', 'Jupiter-Rahu', 'Rahu-Jupiter']
    
    level2_active = any(any(l2 in c for l2 in level2_contradictions) for c in contradictions)
    print(f'Level 2 active: {level2_active}')
    
    # Apply Level 2 modifications
    weights = weights_from_comprehensive.copy()
    
    if level2_active:
        print('\nApplying Level 2 logic...')
        for contradiction in contradictions:
            if any(l2 in contradiction for l2 in level2_contradictions):
                print(f'Processing: {contradiction}')
                planets = contradiction.split('-')
                if len(planets) == 2:
                    planet1, planet2 = planets
                    
                    # Apply negative weight to planet1 wherever it appears
                    if star_lord == planet1:
                        print(f'Making {star_lord} (star_lord) negative: {weights["star_lord"]} -> {-weights["star_lord"]}')
                        weights['star_lord'] *= -1
                    if sub_lord == planet1:
                        print(f'Making {sub_lord} (sub_lord) negative: {weights["sub_lord"]} -> {-weights["sub_lord"]}')
                        weights['sub_lord'] *= -1
                    if sub_sub_lord == planet1:
                        print(f'Making {sub_sub_lord} (sub_sub_lord) negative: {weights["sub_sub_lord"]} -> {-weights["sub_sub_lord"]}')
                        weights['sub_sub_lord'] *= -1
                        
                    # Apply negative weight to planet2 wherever it appears
                    if star_lord == planet2:
                        print(f'Making {star_lord} (star_lord) negative: {weights["star_lord"]} -> {-weights["star_lord"]}')
                        weights['star_lord'] *= -1
                    if sub_lord == planet2:
                        print(f'Making {sub_lord} (sub_lord) negative: {weights["sub_lord"]} -> {-weights["sub_lord"]}')
                        weights['sub_lord'] *= -1
                    if sub_sub_lord == planet2:
                        print(f'Making {sub_sub_lord} (sub_sub_lord) negative: {weights["sub_sub_lord"]} -> {-weights["sub_sub_lord"]}')
                        weights['sub_sub_lord'] *= -1
    
    print(f'\nFinal weights after Level 2: {weights}')
    
    # Now test the actual unified predictor path
    print('\n' + '='*50)
    print('TESTING ACTUAL UNIFIED PREDICTOR')
    
    # Use mock data (since we can't easily generate charts)
    try:
        # We'll call the calculate_scores method but need to provide mock chart data
        # This should reveal the actual bug
        print('Unable to test full unified predictor without chart data')
        print('But the issue is now clear from the flow above')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_weight_flow() 