import sys
import os

# Append parent path to import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

print("PYTHON PATH:", sys.path)

try:
    from scripts.kp_favorability_rules import (
        create_chart,
        get_significator_houses,
        get_conjunctions,
        get_ruling_planets,
        evaluate_period
    )
    print("✅ Import successful")
except ModuleNotFoundError as e:
    print("❌ Import failed:", e)