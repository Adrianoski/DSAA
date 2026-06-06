"""

Use:
    python3 relabel_dataset_allevents.py <input_csv> <output_csv> [margin_seconds]

Example:
    python3 relabel_dataset_allevents.py \\
        cp_training_dataset_290.csv \\
        cp_training_dataset_290_allevt.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pen_sweep_paper import (
    load_paper_events, TARGET_CATEGORY, RARE_CATEGORY, CHANNELS,
)


def relabel(in_csv: str, out_csv: str, margin_s: float = 3600.0) -> None:
    if not Path(in_csv).exists():
        print(f"ERROR: {in_csv} non trovato.")
        sys.exit(1)

    print(f"[relabel] Carico dataset: {in_csv}")
    df = pd.read_csv(in_csv)
    print(f"  {len(df)} CP — positivi attuali (Anomaly only): {int(df['label'].sum())}")

    if "cp_timestamp" not in df.columns:
        print("ERROR: il dataset non contiene la colonna 'cp_timestamp'.")
        sys.exit(1)

    ts = pd.to_datetime(df["cp_timestamp"], utc=True)
    t_min = ts.min() - pd.Timedelta(days=1)
    t_max = ts.max() + pd.Timedelta(days=1)
    print(f"[relabel] Range CP: {ts.min()} → {ts.max()}")

    anom = load_paper_events(category=TARGET_CATEGORY, channels=CHANNELS,
                             t_start=t_min, t_end=t_max)
    rare = load_paper_events(category=RARE_CATEGORY,  channels=CHANNELS,
                             t_start=t_min, t_end=t_max)
    print(f"[relabel] Eventi nel range — Anomaly: {len(anom)}  Rare Event: {len(rare)}")

    events = pd.concat([anom, rare], ignore_index=True)
    margin = pd.Timedelta(seconds=margin_s)
    starts = pd.to_datetime(events["StartTime"].values, utc=True) - margin
    ends   = pd.to_datetime(events["EndTime"].values,   utc=True) + margin

    new_labels = np.zeros(len(df), dtype=int)
    for i, t in enumerate(ts):
        t_pd = pd.Timestamp(t)
        if t_pd.tzinfo is None:
            t_pd = t_pd.tz_localize("UTC")
        if np.any((starts <= t_pd) & (t_pd <= ends)):
            new_labels[i] = 1

    df_out = df.copy()
    df_out["label_anomonly"] = df["label"].astype(int)
    df_out["label"] = new_labels

    n_pos_anom = int(df_out["label_anomonly"].sum())
    n_pos_all  = int(df_out["label"].sum())
    n_added    = int(((df_out["label"] == 1) & (df_out["label_anomonly"] == 0)).sum())
    print(f"\n[relabel] Positivi (Anomaly only):           {n_pos_anom}")
    print(f"[relabel] Positivi (Anomaly OR Rare Event):  {n_pos_all}  (+{n_added})")

    if "fold" in df_out.columns:
        for f in df_out["fold"].unique():
            sub = df_out[df_out["fold"] == f]
            print(f"  fold={f}: {len(sub)} CP, pos={int(sub['label'].sum())}")

    df_out.to_csv(out_csv, index=False)
    print(f"\n[relabel] ✓ Salvato: {out_csv}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_csv  = sys.argv[1]
    out_csv = sys.argv[2]
    margin  = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
    relabel(in_csv, out_csv, margin)


if __name__ == "__main__":
    main()
