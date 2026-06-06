import numpy as np
import ruptures as rpt

from config import (
    MAX_FEATURES,
    PENALTY_SWEEP_STEPS,
    PENALTY_JUMP_THRESHOLD,
    CPD_MIN_SIZE,
    CPD_JUMP,
)
from penalty_sweep import find_optimal_penalty, per_dimension_cost_analysis, elbow_contributing_dims
from stft_utils import extract_all_frequency_timeseries_window


def select_frequencies_for_window(
    df_window,
    channels: list,
    max_features: int = MAX_FEATURES,
    n_steps: int = PENALTY_SWEEP_STEPS,
    jump_threshold: int = PENALTY_JUMP_THRESHOLD,
    verbose: bool = True,
) -> tuple:
    """
    Selezione frequenze con approccio globale del professore.

    Returns
    -------
    selected_freqs : list[float]   frequenze selezionate
    sweep_info     : dict | None   info sweep (pen_optimal, cp_global,
                                   per_dim_gains, dim_labels, ...)
    """
    n_samples = len(df_window)
    if verbose:
        print(f"    [FreqSel] {n_samples} campioni "
              f"[{df_window.index[0]} → {df_window.index[-1]}]")

    # ── Step 1: STFT → tutte le (freq, channel) disponibili ──────────────
    freq_series_dict, _ = extract_all_frequency_timeseries_window(df_window, channels)
    if not freq_series_dict:
        if verbose:
            print("    [FreqSel] STFT vuota — nessuna frequenza disponibile")
        return [], None

    # Costruisce X_full e mantiene il mapping colonna → (freq_hz, channel)
    feats: list = []
    dim_labels: list = []   # (freq_hz, ch) per ogni colonna di X_full
    for freq_hz in sorted(freq_series_dict.keys()):
        ch_dict = freq_series_dict[freq_hz]
        for ch in channels:
            if ch in ch_dict:
                feats.append(ch_dict[ch])
                dim_labels.append((freq_hz, ch))

    if not feats:
        return [], None

    X_full = np.column_stack(feats)   # shape (T, D_full)
    nT, D_full = X_full.shape

    if verbose:
        n_freqs = len(freq_series_dict)
        print(f"    [FreqSel] X_full: {nT}×{D_full} "
              f"({n_freqs} freq × {len(channels)} canali)")

    # ── Step 2: sweep beta su X_full → penalty ottimale ──────────────────
    try:
        pen_optimal, sweep_info = find_optimal_penalty(
            X_full, n_steps=n_steps, jump_threshold=jump_threshold
        )
    except Exception as e:
        if verbose:
            print(f"    [FreqSel] Sweep fallito: {e}")
        return [], None

    pen_bic        = sweep_info["pen_bic"]
    is_significant = sweep_info["is_significant"]
    k_optimal      = sweep_info["k_optimal"]

    if verbose:
        print(f"    [FreqSel] pen*={pen_optimal:.2f}  BIC={pen_bic:.2f}  "
              f"k_knee={k_optimal}  → {'SIG' if is_significant else 'NOISE'}")

    # ── Step 3: controllo soglia BIC ──────────────────────────────────────
    if not is_significant or k_optimal == 0:
        if verbose:
            print("    [FreqSel] Nessuna struttura rilevante → 0 frequenze")
        return [], sweep_info

    # ── Step 4: PELT globale con pen_optimal → CP globali ─────────────────
    algo = rpt.Pelt(model="l2", min_size=CPD_MIN_SIZE, jump=CPD_JUMP).fit(X_full)
    bkps   = algo.predict(pen=pen_optimal)
    cp_idx = [b - 1 for b in bkps[:-1]]

    if not cp_idx:
        return [], sweep_info

    if verbose:
        print(f"    [FreqSel] CP globali trovati: {cp_idx}")

    # ── Step 5: gain per-dimensione ai CP globali ─────────────────────────
    # "calcoli costi prima e dopo per i CP globali ma il costo lo calcoli
    #  individualmente" — professore.
    per_dim_gains = per_dimension_cost_analysis(X_full, bkps)

    # ── Step 6: elbow sulle dimensioni ────────────────────────────────────
    # "thresholding non è abbastanza, guarda l'elbow"
    n_contributing, threshold_gain, contributing_mask = elbow_contributing_dims(
        per_dim_gains, min_contributing=1
    )

    if verbose:
        sorted_idx = np.argsort(per_dim_gains)[::-1]
        print(f"    [FreqSel] Elbow: {n_contributing}/{D_full} dim sopra soglia "
              f"(gain_min={threshold_gain:.2f})")
        print("    [FreqSel] Gain per-dim (top):")
        for rank, d in enumerate(sorted_idx[:min(6, D_full)], 1):
            marker = "✓" if contributing_mask[d] else "✗"
            fhz, ch = dim_labels[d]
            print(f"      {marker} {rank}. {fhz:.6f}Hz/{ch}  gain={per_dim_gains[d]:.3f}")

    if n_contributing == 0:
        return [], sweep_info

    # ── Step 7: mappa dim → frequenze uniche ─────────────────────────────
    # Ordina le dimensioni contributing per gain decrescente → prendi le freq
    contributing_idx = np.where(contributing_mask)[0]
    contributing_idx_sorted = contributing_idx[
        np.argsort(per_dim_gains[contributing_idx])[::-1]
    ]

    selected_freqs: list = []
    for d_idx in contributing_idx_sorted:
        freq_hz, _ = dim_labels[d_idx]
        if freq_hz not in selected_freqs:
            selected_freqs.append(freq_hz)
        if len(selected_freqs) >= max_features:
            break

    if verbose:
        print(f"    [FreqSel] Frequenze selezionate: {len(selected_freqs)}")
        for f in selected_freqs:
            print(f"      → {f:.6f} Hz")

    # Arricchisce sweep_info con i dati per-dimensione
    sweep_info["per_dim_gains"]   = per_dim_gains
    sweep_info["dim_labels"]      = dim_labels
    sweep_info["cp_global"]       = cp_idx
    sweep_info["bkps_global"]     = bkps
    sweep_info["contributing_mask"] = contributing_mask

    return selected_freqs, sweep_info
