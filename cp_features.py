"""
cp_features.py
==============
Le 17 feature per ogni CP individuale b_i con bordi (b_{i-1}, b_i, b_{i+1}):

  Gain locale per-dim (quale dimensione "vota" per QUESTO CP):
    - local_gain_top1      gain della dim più forte
    - local_gain_top3_sum  somma top-3 gain
    - local_gain_sum       somma totale (forza del CP)
    - n_contributing_dims  dim sopra elbow locale
    - n_channels_voting    canali distinti tra le contributing
    - n_freqs_voting       frequenze distinte tra le contributing
    - has_dc               1 se DC è tra le contributing
    - freq_min/max/mean    statistiche delle frequenze contributing

  Durate regimi (un CP che separa due regimi molto lunghi è meno "anomalo"):
    - dur_before           lunghezza regime prima del CP (frame)
    - dur_after            lunghezza regime dopo il CP (frame)
    - dur_ratio            max(b,a) / max(min(b,a), 1)

  Contesto window-level (broadcast a tutti i CP della finestra):
    - cp_position_norm     b_i / n_frames ∈ [0, 1]
    - n_cps_in_window      numero totale CP nella finestra
    - pen_optimal          penalty del PELT per la finestra
    - k_knee               k del knee nel sweep
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import numpy as np

from penalty_sweep import _l2_cost_segment, elbow_contributing_dims


FEATURE_NAMES: List[str] = [
    "local_gain_top1",
    "local_gain_top3_sum",
    "local_gain_sum",
    "n_contributing_dims",
    "n_channels_voting",
    "n_freqs_voting",
    "has_dc",
    "freq_min",
    "freq_max",
    "freq_mean",
    "dur_before",
    "dur_after",
    "dur_ratio",
    "cp_position_norm",
    "n_cps_in_window",
    "pen_optimal",
    "k_knee",
]


def per_cp_per_dim_gain(X: np.ndarray, bkps_global: List[int]) -> np.ndarray:
    """
    Per ogni CP interno b_i e ogni dimensione d, calcola il GAIN LOCALE:

        gain_{i,d} = L2(X_d[b_{i-1}:b_{i+1}])
                     − L2(X_d[b_{i-1}:b_i])
                     − L2(X_d[b_i:b_{i+1}])

    Misura quanto SPECIFICAMENTE il CP b_i riduce il costo L2 nella
    dimensione d, a parità degli altri breakpoint. Diverso da
    `per_dimension_cost_analysis` di penalty_sweep, che aggrega su tutti i CP.

    Parameters
    ----------
    X : np.ndarray  shape (T, D)
    bkps_global : list  breakpoints PELT (include T come ultimo elemento)

    Returns
    -------
    gains_local : np.ndarray  shape (n_cp, D)
                  dove n_cp = numero di CP interni (esclude 0 e T).
                  Se n_cp == 0 ritorna un array shape (0, D).
    """
    nT, D = X.shape
    boundaries = [0] + sorted(bkps_global)        # [0, b_1, ..., b_K, T]
    n_cp = len(boundaries) - 2                    # CP interni (escludono 0 e T)
    if n_cp <= 0:
        return np.zeros((0, D))

    gains_local = np.zeros((n_cp, D))
    for i in range(n_cp):
        b_prev = boundaries[i]
        b_cur  = boundaries[i + 1]
        b_next = boundaries[i + 2]
        for d in range(D):
            x_d = X[:, d]
            c_merged = _l2_cost_segment(x_d[b_prev:b_next])
            c_left   = _l2_cost_segment(x_d[b_prev:b_cur])
            c_right  = _l2_cost_segment(x_d[b_cur:b_next])
            gains_local[i, d] = max(0.0, c_merged - c_left - c_right)
    return gains_local


def extract_cp_features(
    X_full: np.ndarray,
    bkps_global: List[int],
    dim_labels: List[Tuple[float, str]],
    pen_optimal: float,
    k_knee: int,
) -> List[Dict[str, Any]]:
    """
    Estrae le 17 feature per ogni CP interno della finestra.

    Returns
    -------
    list[dict]  una entry per CP, ognuna con tutte le chiavi di FEATURE_NAMES
                + 'cp_idx_in_window' (posizione b_i intera nei frame STFT,
                usata a valle per ricostruire il timestamp).
                Se non ci sono CP interni, ritorna [].
    """
    nT, D = X_full.shape
    boundaries = [0] + sorted(bkps_global)
    n_cp = len(boundaries) - 2
    if n_cp <= 0:
        return []

    gains_local = per_cp_per_dim_gain(X_full, bkps_global)

    rows: List[Dict[str, Any]] = []
    for i in range(n_cp):
        b_prev = boundaries[i]
        b_cur  = boundaries[i + 1]
        b_next = boundaries[i + 2]

        g = gains_local[i]                      # shape (D,)
        sorted_g = np.sort(g)[::-1]

        # Elbow locale: dimensioni che spiegano QUESTO CP
        try:
            _, _, contributing_mask = elbow_contributing_dims(g, min_contributing=1)
        except Exception:
            contributing_mask = np.zeros(D, dtype=bool)
        contrib_dims = np.where(contributing_mask)[0]

        contrib_freqs = [dim_labels[d][0] for d in contrib_dims]
        contrib_chs   = [dim_labels[d][1] for d in contrib_dims]

        if len(contrib_freqs) > 0:
            f_min  = float(min(contrib_freqs))
            f_max  = float(max(contrib_freqs))
            f_mean = float(np.mean(contrib_freqs))
            has_dc = int(any(f < 1e-6 for f in contrib_freqs))
        else:
            f_min = f_max = f_mean = 0.0
            has_dc = 0

        dur_before = int(b_cur - b_prev)
        dur_after  = int(b_next - b_cur)
        dur_ratio  = float(max(dur_before, dur_after) / max(1, min(dur_before, dur_after)))

        rows.append({
            "local_gain_top1":      float(sorted_g[0]) if len(sorted_g) > 0 else 0.0,
            "local_gain_top3_sum":  float(sorted_g[:3].sum()),
            "local_gain_sum":       float(g.sum()),
            "n_contributing_dims":  int(len(contrib_dims)),
            "n_channels_voting":    int(len(set(contrib_chs))),
            "n_freqs_voting":       int(len(set(contrib_freqs))),
            "has_dc":               has_dc,
            "freq_min":             f_min,
            "freq_max":             f_max,
            "freq_mean":            f_mean,
            "dur_before":           dur_before,
            "dur_after":            dur_after,
            "dur_ratio":            dur_ratio,
            "cp_position_norm":     float(b_cur / nT),
            "n_cps_in_window":      int(n_cp),
            "pen_optimal":          float(pen_optimal),
            "k_knee":               int(k_knee),
            # Campo extra (non in FEATURE_NAMES): posizione del CP nel grid
            # STFT — serve per ricostruire il timestamp del CP.
            "cp_idx_in_window":     int(b_cur),
        })
    return rows
