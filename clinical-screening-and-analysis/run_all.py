"""Run every analysis in order. Usage:  python run_all.py"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = ["dengue_analysis.py", "ratios_analysis.py", "dx_profile.py", "panel_comparison.py",
           "compare_scores.py", "internal_validation.py", "secondary_analyses.py", "figures.py",
           "clustering.py"]
here = Path(__file__).resolve().parent / "analysis"
for s in SCRIPTS:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, str(here / s)], check=True, cwd=here)
