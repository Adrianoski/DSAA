"""
eval_unsup_from_clf_csv.py
==========================
Use:
    python3 eval_unsup_from_clf_csv.py <input_clf_csv> [output_csv]

Esempio:
    python3 eval_unsup_from_clf_csv.py \\
        cpd_output_classified_knee/test_results.csv \\
        cpd_output_classified_knee/test_results_unsup_view.csv

    # Poi:
    python3 pen_sweep_paper.py --mode eval --include-rare \\
        --results-csv cpd_output_classified_knee/test_results_unsup_view.csv \\
        --test-csv 84_months.test.unsup_std.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_csv = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else \
        in_csv.replace(".csv", "_unsup_view.csv")

    if not Path(in_csv).exists():
        print(f"ERROR: {in_csv} non trovato.")
        sys.exit(1)

    print(f"[unsup-view] Carico: {in_csv}")
    df = pd.read_csv(in_csv)
    print(f"  {len(df)} righe, colonne: {list(df.columns)[:10]}...")

    if "n_cps_initial" not in df.columns:
        print(f"ERROR: il CSV non contiene la colonna 'n_cps_initial'. "
              f"Probabilmente non è un output del pipeline classified.")
        sys.exit(1)

    # Sostituisci n_changepoints con n_cps_initial (= pre-classifier)
    df["n_changepoints"] = df["n_cps_initial"]

    df.to_csv(out_csv, index=False)
    print(f"[unsup-view] ✓ Salvato: {out_csv}")
    print(f"  n_changepoints ora riflette n_cps_initial (pre-classifier).")

    n_triggered = int((df["n_changepoints"] > 0).sum())
    print(f"  Finestre triggered (post-gate, pre-classifier): {n_triggered}")


if __name__ == "__main__":
    main()
