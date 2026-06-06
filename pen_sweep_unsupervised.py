from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_dataset, CHANNELS_41_46  # noqa: E402

from config import CHANNELS, TRAIN_CSV, TEST_CSV  # noqa: E402

# Riusiamo le primitive della versione paper-aligned
from pen_sweep_paper import (  # noqa: E402
    compute_nominal_stats,
    standardize_in_place,
    evaluate,
)

UNSUP_TRAIN_CSV  = "84_months.train.unsup_std.csv"
UNSUP_TEST_CSV   = "84_months.test.unsup_std.csv"
UNSUP_OUTPUT_DIR = "cpd_output_unsupervised"


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing unsupervised
# ─────────────────────────────────────────────────────────────────────────────

def build_unsupervised_csvs(train_csv: str, test_csv: str,
                            out_train: str, out_test: str) -> None:

    print(f"[unsup] Carico training: {train_csv}")
    df_train, _, _ = load_dataset(
        csv_path=train_csv, use_channels=CHANNELS_41_46,
        include_labels=False, interpolate=True, fill_strategy="both",
    )
    channels = [c for c in CHANNELS if c in df_train.columns]

    print(f"[unsup] Calcolo mean/std su TUTTI i punti del training "
          f"({len(channels)} canali, nessun uso delle label)")
    # labels=[] → compute_nominal_stats usa tutti i punti
    mean, std = compute_nominal_stats(df_train, channels, labels=[])
    for c in channels:
        print(f"    {c}: μ={mean[c]:.6f}  σ={std[c]:.6f}")

    standardize_in_place(df_train, channels, mean, std)
    df_train[channels].to_csv(out_train, index_label="timestamp")
    print(f"[unsup] ✓ Train salvato SENZA label: {out_train}")

    print(f"[unsup] Carico test: {test_csv}")
    df_test, _, _ = load_dataset(
        csv_path=test_csv, use_channels=CHANNELS_41_46,
        include_labels=False, interpolate=True, fill_strategy="both",
    )
    standardize_in_place(df_test, channels, mean, std)
    df_test[channels].to_csv(out_test, index_label="timestamp")
    print(f"[unsup] ✓ Test salvato: {out_test}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from pen_sweep_update import run_complete_pipeline

    print(f"\n{'#'*70}")
    print(f"# CPD — VARIANTE COMPLETAMENTE UNSUPERVISED")
    print(f"# Nessun uso delle label in training (standardizzazione + soglia)")
    print(f"{'#'*70}\n")

    # 1) Pre-processing unsupervised (standardizzazione su tutti i punti)
    build_unsupervised_csvs(TRAIN_CSV, TEST_CSV,
                            UNSUP_TRAIN_CSV, UNSUP_TEST_CSV)

    # 2) Pipeline CPD. Il training CSV non ha label → build_nominal_gain_
    #    threshold calibra la soglia su TUTTE le finestre (unsupervised).
    print(f"\n[unsup] Pipeline CPD su dati unsupervised → {UNSUP_OUTPUT_DIR}")
    print("[unsup] NB: la calibrazione stamperà 'finestre nominali' ma in realtà")
    print("        processa OGNI finestra (il CSV di training non ha label).\n")
    run_complete_pipeline(
        test_csv=UNSUP_TEST_CSV,
        train_csv=UNSUP_TRAIN_CSV,
        output_dir=UNSUP_OUTPUT_DIR,
        save_plots=True,
    )

    # 3) Valutazione paper-aligned (stesse metriche della versione semi-sup)
    evaluate(
        test_csv=UNSUP_TEST_CSV,
        results_csv=str(Path(UNSUP_OUTPUT_DIR) / "test_results.csv"),
        beta=0.5,
    )


if __name__ == "__main__":
    main()
