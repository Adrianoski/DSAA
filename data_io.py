from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# Canali multivariati di interesse
CHANNELS_41_46: List[str] = [f"channel_{i}" for i in range(41, 47)]

# Pattern per colonne etichetta tipo: is_anomaly_channel_41
_LABEL_RE = re.compile(r"^is_anomaly_channel_(\d+)$")


def _coerce_timestamp_utc(series: pd.Series) -> pd.DatetimeIndex:
    """Parsa la colonna 'timestamp' e la converte in UTC, rimuovendo valori non parsabili."""
    idx = pd.to_datetime(series, utc=True, errors="coerce")
    if idx.isna().all():
        raise ValueError("Nessun timestamp valido nel dataset.")
    return idx


def _select_available(columns: List[str], desired: List[str]) -> List[str]:
    """Ritorna l'intersezione preservando l'ordine di 'desired'."""
    colset = set(columns)
    return [c for c in desired if c in colset]


def _find_label_columns(columns: List[str], restrict_to: Optional[List[str]] = None) -> List[str]:
    """Trova le colonne etichetta is_anomaly_channel_XX, opzionalmente filtrando per i canali forniti."""
    labels = []
    for c in columns:
        m = _LABEL_RE.match(c)
        if not m:
            continue
        ch_name = f"channel_{m.group(1)}"
        if restrict_to is None or ch_name in restrict_to:
            labels.append(c)
    return labels


def load_dataset(
    csv_path: str,
    use_channels: Optional[List[str]] = None,
    include_labels: bool = True,
    interpolate: bool = True,
    fill_strategy: str = "both",  # "forward", "backward", "both", "none"
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Carica un CSV con almeno le colonne: 'timestamp' e channel_*.
    Seleziona i canali 41–46 (o quelli passati in use_channels) e, se richiesto, le colonne etichetta is_anomaly_*.

    Ritorna:
        df: DataFrame indicizzato per timestamp UTC, ordinato
        channels: lista canali effettivamente caricati
        labels: lista colonne etichetta incluse (può essere vuota)

    Note:
        - Converte i canali in float, sostituisce ±inf con NaN
        - Opzionalmente interpola per colmare buchi
        - Le etichette, se presenti, sono convertite a Int64
    """
    # 1) Leggi CSV
    df_raw = pd.read_csv(csv_path)

    if "timestamp" not in df_raw.columns:
        raise ValueError("Colonna 'timestamp' assente nel CSV.")

    # 2) Timestamp → UTC e ordinamento
    idx = _coerce_timestamp_utc(df_raw["timestamp"])
    ok = idx.notna()
    df_raw = df_raw.loc[ok].copy()
    idx = idx[ok]

    # 3) Determina i canali da usare
    desired_channels = use_channels if use_channels is not None else CHANNELS_41_46
    channels = _select_available(df_raw.columns.tolist(), desired_channels)
    if not channels:
        raise ValueError("Nessun canale tra 41–46 trovato nel dataset.")

    # 4) Determina le colonne etichetta
    labels: List[str] = []
    if include_labels:
        labels = _find_label_columns(df_raw.columns.tolist(), restrict_to=channels)

    # 5) Costruisci df con colonne selezionate
    cols: List[str] = channels + labels
    df = df_raw[cols].copy()
    df.index = idx
    df = df.sort_index()

    # 6) Numerici e pulizia
    for c in channels:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[channels] = df[channels].replace([np.inf, -np.inf], np.nan)

    # 7) Interpolazione/riempimento opzionali sui canali
    if interpolate:
        df[channels] = df[channels].interpolate(limit_direction="both")

    if fill_strategy in ("forward", "both"):
        df[channels] = df[channels].ffill()
    if fill_strategy in ("backward", "both"):
        df[channels] = df[channels].bfill()

    # 8) Etichette a Int64 se presenti
    for c in labels:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df, channels, labels


def summarize_dataframe(df: pd.DataFrame, channels: List[str], labels: List[str]) -> Dict[str, object]:
    """Ritorna un piccolo sommario utile da stampare nel notebook."""
    return {
        "start": df.index.min(),
        "end": df.index.max(),
        "n_rows": int(df.shape[0]),
        "n_channels": len(channels),
        "n_labels": len(labels),
        "channels": channels,
        "labels": labels,
        "nan_ratio_per_channel": {c: float(df[c].isna().mean()) for c in channels},
    }
