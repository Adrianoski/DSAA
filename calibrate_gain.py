"""
calibrate_gain.py
=================
Use:
    python3 calibrate_gain.py [--train-csv 84_months.train.unsup_std.csv]
                              [--out-prefix gain_distribution]

Output:
    <prefix>_max_gains.npy       array (n_windows,) di max_gain
    <prefix>_summary.json        statistiche + proposte di soglia
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import ruptures as rpt

# matplotlib non-interactive: salva su file senza display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_dataset, CHANNELS_41_46  # noqa: E402

from config import (  # noqa: E402
    CHANNELS,
    WINDOW_DAYS,
    CPD_MIN_SIZE, CPD_JUMP,
    PENALTY_SWEEP_STEPS, PENALTY_JUMP_THRESHOLD,
)
from penalty_sweep import (  # noqa: E402
    find_optimal_penalty,
    per_dimension_cost_analysis,
    elbow_contributing_dims,
)
from stft_utils import extract_all_frequency_timeseries_window  # noqa: E402
from pipeline import create_fixed_windows  # noqa: E402


DEFAULT_TRAIN_CSV   = "84_months.train.unsup_std.csv"
DEFAULT_OUT_PREFIX  = "gain_distribution"


# ─────────────────────────────────────────────────────────────────────────────
# Calibrazione: passa CPD su ogni finestra, raccoglie max(gain_aggregate)
# ─────────────────────────────────────────────────────────────────────────────

def compute_max_gains(train_csv: str) -> np.ndarray:
    """
    Per ogni finestra del training set:
      STFT → X_full → sweep → PELT → gain per-dim → max(gain).

    Ritorna np.array(n,) con il max_gain di ogni finestra in cui PELT ha
    trovato almeno un CP. Le finestre senza CP o con sweep fallito sono
    scartate. Nessun filtro per label (uso unsupervised).
    """
    print(f"[calib] Carico training: {train_csv}")
    df_train, _, _ = load_dataset(
        csv_path=train_csv, use_channels=CHANNELS_41_46,
        include_labels=False, interpolate=True, fill_strategy="both",
    )
    df_train = df_train.sort_index()
    present_channels = [c for c in CHANNELS if c in df_train.columns]
    print(f"[calib] {len(df_train)} samples, {len(present_channels)} canali")

    windows = create_fixed_windows(df_train, window_days=WINDOW_DAYS)
    n_win = len(windows)
    print(f"[calib] {n_win} finestre da {WINDOW_DAYS} giorni\n")

    max_gains = []
    n_skipped = 0

    for w_id, (win_start, win_end) in enumerate(windows, 1):
        W = df_train.loc[win_start:win_end, present_channels]
        if W.dropna(how="all").empty:
            n_skipped += 1
            continue

        freq_series_dict, _ = extract_all_frequency_timeseries_window(W, present_channels)
        if not freq_series_dict:
            n_skipped += 1
            continue

        feats = []
        for freq_hz in sorted(freq_series_dict.keys()):
            for ch in present_channels:
                if ch in freq_series_dict[freq_hz]:
                    feats.append(freq_series_dict[freq_hz][ch])
        if not feats:
            n_skipped += 1
            continue

        X_full = np.column_stack(feats)
        if X_full.shape[0] < 2:
            n_skipped += 1
            continue

        try:
            pen_opt, sweep_info = find_optimal_penalty(
                X_full,
                n_steps=PENALTY_SWEEP_STEPS,
                jump_threshold=PENALTY_JUMP_THRESHOLD,
            )
        except Exception:
            n_skipped += 1
            continue

        if sweep_info["k_optimal"] == 0:
            n_skipped += 1
            continue

        try:
            algo = rpt.Pelt(model="l2", min_size=CPD_MIN_SIZE, jump=CPD_JUMP).fit(X_full)
            bkps = algo.predict(pen=pen_opt)
        except Exception:
            n_skipped += 1
            continue

        if len(bkps) <= 1:
            n_skipped += 1
            continue

        gains = per_dimension_cost_analysis(X_full, bkps)
        max_g = float(gains.max())
        if max_g > 0:
            max_gains.append(max_g)
        else:
            n_skipped += 1

        if w_id % 20 == 0:
            print(f"  [calib] {w_id}/{n_win} — raccolti {len(max_gains)} "
                  f"max_gain (skip={n_skipped})")

    arr = np.array(max_gains, dtype=float)
    print(f"\n[calib] Calibrazione completata: {len(arr)} max_gain raccolti, "
          f"{n_skipped} finestre scartate.")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Proposte di soglia da una distribuzione di max_gain
# ─────────────────────────────────────────────────────────────────────────────

def propose_thresholds(max_gains: np.ndarray) -> dict:
    """
    A partire dall'array max_gains, propone diverse soglie:
      - percentili classici
      - k-σ rule
      - knee della curva ordinata decrescente (metodo della corda)
    """
    if len(max_gains) < 3:
        raise ValueError(f"Servono almeno 3 max_gain, ricevuti {len(max_gains)}.")

    mu = float(np.mean(max_gains))
    sigma = float(np.std(max_gains))
    median = float(np.median(max_gains))
    maximum = float(np.max(max_gains))
    minimum = float(np.min(max_gains))

    percentiles = {
        f"p{p}": float(np.percentile(max_gains, p))
        for p in [50, 70, 75, 80, 85, 90, 95, 99]
    }

    k_sigma = {
        f"mu+{k}sigma": mu + k * sigma
        for k in [1.0, 1.5, 2.0, 2.5, 3.0]
    }

    # KNEE: riusa elbow_contributing_dims, che internamente ordina decrescente
    # e applica il metodo della corda. Ritorna (n_above_knee, knee_value, mask).
    n_above_knee, knee_value, _ = elbow_contributing_dims(
        max_gains, min_contributing=1
    )
    # Percentile a cui cade il knee (per riferimento)
    knee_percentile = float(np.mean(max_gains <= knee_value) * 100.0)

    knee = {
        "knee_gain":          float(knee_value),
        "n_above_knee":       int(n_above_knee),
        "fraction_above_knee": float(n_above_knee / len(max_gains)),
        "knee_at_percentile":  knee_percentile,
    }

    return {
        "stats": {
            "n_windows": int(len(max_gains)),
            "mu":        mu,
            "sigma":     sigma,
            "median":    median,
            "min":       minimum,
            "max":       maximum,
        },
        "percentiles": percentiles,
        "k_sigma":     k_sigma,
        "knee":        knee,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grafico: distribuzione + knee + percentili di riferimento
# ─────────────────────────────────────────────────────────────────────────────

def save_distribution_plot(
    max_gains: np.ndarray,
    proposals: dict,
    out_path: str,
) -> None:
    """
    Salva un grafico con due pannelli:
      sinistra  — curva max_gain ordinata in decrescente, con knee e corda
      destra    — istogramma con knee + 95° e 50° percentile come riferimento
    """
    sorted_desc = np.sort(max_gains)[::-1]
    n = len(sorted_desc)
    knee_value = proposals["knee"]["knee_gain"]
    n_above_knee = proposals["knee"]["n_above_knee"]
    knee_idx = max(0, min(n - 1, n_above_knee))  # posizione knee nel ranking

    p50 = proposals["percentiles"]["p50"]
    p95 = proposals["percentiles"]["p95"]
    knee_perc = proposals["knee"]["knee_at_percentile"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Pannello SX: sorted descending + knee + chord ────────────────────────
    ax = axes[0]
    ax.plot(np.arange(n), sorted_desc, "b-", linewidth=1.2,
            label="max_gain (sorted desc)")
    # Corda dalla cima al fondo della curva
    if n >= 2:
        ax.plot([0, n - 1], [sorted_desc[0], sorted_desc[-1]],
                color="gray", linestyle="--", alpha=0.6, label="chord")
    # Knee
    ax.axhline(knee_value, color="red", linestyle="--", linewidth=1,
               label=f"knee = {knee_value:.2f}  (~{knee_perc:.1f}° pct)")
    ax.axvline(knee_idx, color="red", linestyle=":", alpha=0.5)
    ax.scatter([knee_idx], [knee_value], color="red", s=70, zorder=5)
    ax.set_xlabel("Window rank (sorted descending)")
    ax.set_ylabel("max_gain")
    ax.set_title(f"Sorted max_gain distribution — knee at rank {knee_idx}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Pannello DX: istogramma + knee + percentili ──────────────────────────
    ax = axes[1]
    ax.hist(max_gains, bins=50, color="steelblue", edgecolor="black",
            alpha=0.75)
    ax.axvline(knee_value, color="red", linestyle="--", linewidth=2,
               label=f"knee = {knee_value:.2f}")
    ax.axvline(p50, color="gray", linestyle=":",
               label=f"50° percentile = {p50:.2f}")
    ax.axvline(p95, color="orange", linestyle=":",
               label=f"95° percentile = {p95:.2f}")
    ax.set_xlabel("max_gain")
    ax.set_ylabel("Count")
    ax.set_title(f"max_gain histogram — N = {n} windows")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training-set gain distribution & knee detection",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[calib] ✓ Plot salvato: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV,
                    help=f"Path CSV training standardizzato "
                         f"(default: {DEFAULT_TRAIN_CSV}).")
    ap.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX,
                    help=f"Prefix dei file di output (default: "
                         f"{DEFAULT_OUT_PREFIX}).")
    args = ap.parse_args()

    print(f"{'#'*70}")
    print(f"# CALIBRATE_GAIN — distribuzione max_gain training")
    print(f"# Train CSV:  {args.train_csv}")
    print(f"# Out prefix: {args.out_prefix}")
    print(f"{'#'*70}\n")

    max_gains = compute_max_gains(args.train_csv)

    # Persisti l'array per future analisi
    npy_path = f"{args.out_prefix}_max_gains.npy"
    np.save(npy_path, max_gains)
    print(f"[calib] ✓ {len(max_gains)} valori salvati: {npy_path}")

    # Calcola e salva proposte
    proposals = propose_thresholds(max_gains)
    json_path = f"{args.out_prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(proposals, f, indent=2)
    print(f"[calib] ✓ Summary salvato: {json_path}")

    # Salva il grafico della distribuzione con il knee marcato
    plot_path = f"{args.out_prefix}_plot.png"
    try:
        save_distribution_plot(max_gains, proposals, plot_path)
    except Exception as e:
        print(f"[calib] ⚠ Impossibile generare il plot ({e}). "
              f"Continuo senza.")

    # ── Riepilogo a video ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"GAIN DISTRIBUTION SUMMARY")
    print(f"{'='*70}")
    s = proposals["stats"]
    print(f"  N windows:      {s['n_windows']}")
    print(f"  μ:              {s['mu']:.3f}")
    print(f"  σ:              {s['sigma']:.3f}")
    print(f"  median:         {s['median']:.3f}")
    print(f"  min / max:      {s['min']:.3f} / {s['max']:.3f}")

    print(f"\n  ── Percentili (operating points classici) ──")
    for k, v in proposals["percentiles"].items():
        print(f"    {k:8s}  →  τ = {v:8.3f}")

    print(f"\n  ── Regola k-σ (convenzione GlobalSTD) ──")
    for k, v in proposals["k_sigma"].items():
        print(f"    {k:14s}  →  τ = {v:8.3f}")

    print(f"\n  ── KNEE (metodo della corda, parameter-free) ──")
    knee = proposals["knee"]
    print(f"    knee_gain          = {knee['knee_gain']:.3f}")
    print(f"    n above knee       = {knee['n_above_knee']} / {s['n_windows']}")
    print(f"    fraction above     = {knee['fraction_above_knee']:.4f}")
    print(f"    knee at percentile = {knee['knee_at_percentile']:.2f}°")
    print(f"{'='*70}\n")
    print(f"→ Soglia raccomandata (knee, parameter-free): τ ≈ "
          f"{knee['knee_gain']:.2f}")
    print(f"  Lancia: ./run_pipeline.sh {knee['knee_gain']:.2f} knee")


if __name__ == "__main__":
    main()
