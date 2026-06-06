"""
main.py
=======
Entry point per la pipeline con Penalty Sweep.

Uso:
    cd PenaltySweep
    python main.py
"""

from pipeline import run_complete_pipeline
from config import TEST_CSV, OUTPUT_DIR

if __name__ == "__main__":
    results = run_complete_pipeline(
        test_csv=TEST_CSV,
        output_dir=OUTPUT_DIR,
        save_plots=True,
    )
