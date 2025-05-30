import sys
import os

# Append parent path to import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

print("PYTHON PATH:", sys.path)

# test_import_chart_generator.py
from scripts.chart_generator import generate_kp_chart

print("✅ Import successful")