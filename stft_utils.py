"""
stft_utils.py
=============
Calcolo STFT, estrazione serie temporali per frequenza,
estrazione feature matrix per CPD.
"""

import numpy as np
import pandas as pd

from config import (
    STFT_NPERSEG, STFT_OVERLAP_FRAC, STFT_WINDOW,
    MIN_BAND_HZ, MAX_BAND_HZ,
)


# ===========================
# Finestra temporale
# ===========================

def _window_vec(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)
    return np.ones(n)


# ===========================
# STFT numpy-only
# ===========================

def stft_numpy(
    x: np.ndarray,
    fs: float,
    nperseg: int,
    overlap_frac: float,
    window: str = "hann",
):
    """Ritorna freqs[Hz], times_idx[int], STFT complex [F, T]."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    noverlap = int(round(nperseg * np.clip(overlap_frac, 0.0, 0.98)))
    step = max(1, nperseg - noverlap)
    if n < nperseg:
        x = np.pad(x, (0, nperseg - n), mode="edge")
        n = len(x)
    w = _window_vec(window, nperseg)
    n_frames = 1 + (n - nperseg) // step
    Z = np.empty((nperseg // 2 + 1, n_frames), dtype=complex)
    t_idx = np.empty(n_frames, dtype=int)
    for k in range(n_frames):
        start = k * step
        seg = x[start : start + nperseg] * w
        Z[:, k] = np.fft.rfft(seg)
        t_idx[k] = start + nperseg // 2
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, t_idx, Z


def median_step_ns(index_like) -> int:
    idx = pd.DatetimeIndex(index_like)
    if len(idx) >= 2:
        diffs = np.diff(idx.asi8)
        if len(diffs):
            return int(np.median(diffs))
    return 1_000_000_000


# ===========================
# Estrazione serie per frequenza su una finestra
# ===========================

def extract_all_frequency_timeseries_window(df_window: pd.DataFrame, channels: list):
    """
    Per ogni canale e ogni frequenza nella banda [MIN_BAND_HZ, MAX_BAND_HZ],
    estrae la serie temporale della potenza spettrale (normalizzata z-score).

    Returns
    -------
    freq_series_dict : dict  {freq_hz: {channel: array[n_frames]}}
    freqs_b          : np.ndarray  frequenze nella banda
    """
    step_ns = median_step_ns(df_window.index)
    fs = 1.0 / (step_ns / 1e9)

    freq_series_dict: dict = {}
    freqs_b = None

    for ch in channels:
        s_ch = pd.to_numeric(df_window[ch], errors="coerce").astype(float)
        s_ch = s_ch.interpolate(limit_direction="both")

        if s_ch.dropna().empty or len(s_ch) < STFT_NPERSEG:
            continue

        freqs, t_idx, Z = stft_numpy(
            s_ch.values,
            fs=fs,
            nperseg=STFT_NPERSEG,
            overlap_frac=STFT_OVERLAP_FRAC,
            window=STFT_WINDOW,
        )
        P = np.abs(Z) ** 2

        fmask = (freqs >= MIN_BAND_HZ) & (freqs <= MAX_BAND_HZ)
        P_b = P[fmask, :]
        freqs_b_ch = freqs[fmask]

        if freqs_b is None:
            freqs_b = freqs_b_ch

        for i, freq_hz in enumerate(freqs_b_ch):
            if freq_hz not in freq_series_dict:
                freq_series_dict[freq_hz] = {}
            series = P_b[i, :]
            series_norm = (series - series.mean()) / (series.std() + 1e-12)
            freq_series_dict[freq_hz][ch] = series_norm

    if freqs_b is None:
        freqs_b = np.array([])

    return freq_series_dict, freqs_b


# ===========================
# Estrazione feature matrix per CPD
# ===========================

def extract_features_from_selected_freqs(
    df: pd.DataFrame,
    win_start,
    win_end,
    selected_freqs_list: list,
    present_channels: list,
):
    """
    Costruisce la matrice X [T, D] per PELT dalle frequenze selezionate.

    Returns
    -------
    X          : np.ndarray [T, D] o None
    t_stamps   : np.ndarray di datetime64
    feat_labels: list[str]
    """
    W = df.loc[win_start:win_end, present_channels]

    step_ns = median_step_ns(W.index)
    fs = 1.0 / (step_ns / 1e9)

    stft_cache: dict = {}
    freqs_base = None
    t_idx_base = None

    for ch in present_channels:
        s_ch = pd.to_numeric(W[ch], errors="coerce").astype(float)
        s_ch = s_ch.interpolate(limit_direction="both")

        freqs, t_idx, Z = stft_numpy(
            s_ch.values,
            fs=fs,
            nperseg=STFT_NPERSEG,
            overlap_frac=STFT_OVERLAP_FRAC,
            window=STFT_WINDOW,
        )
        P = np.abs(Z) ** 2

        if freqs_base is None:
            freqs_base = freqs
            t_idx_base = t_idx
        else:
            T_common = min(t_idx_base.shape[0], t_idx.shape[0])
            t_idx_base = t_idx_base[:T_common]
            P = P[:, :T_common]

        fmask = (freqs_base >= MIN_BAND_HZ) & (freqs_base <= MAX_BAND_HZ)
        P_b = P[fmask, :]
        freqs_b = freqs_base[fmask]
        stft_cache[ch] = (P_b, freqs_b)

    feats = []
    feat_labels = []

    for target_freq in selected_freqs_list:
        for ch in present_channels:
            P_b, freqs_b = stft_cache[ch]
            diffs = np.abs(freqs_b - target_freq)
            closest_idx = int(np.argmin(diffs))
            if diffs[closest_idx] <= 0.001:
                series = P_b[closest_idx, :]
                series = (series - series.mean()) / (series.std() + 1e-12)
                feats.append(series)
                actual_freq = freqs_b[closest_idx]
                label = f"{ch}@DC" if actual_freq < 0.0001 else f"{ch}@{actual_freq:.6f}Hz"
                feat_labels.append(label)

    if not feats:
        return None, None, None

    X = np.stack(feats, axis=1)

    focus_ch = present_channels[0]
    s_ref = W[focus_ch].astype(float)
    t_idx_base = np.clip(t_idx_base, 0, len(s_ref) - 1)
    t_stamps = s_ref.index.values[t_idx_base]

    return X, t_stamps, feat_labels
