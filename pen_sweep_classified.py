"""
pen_sweep_classified.py

Three modes:

  --mode generate-train
      Runs a full CPD pass over the training set, applies the unsupervised
      gate, extracts the features of every surviving CP, labels each CP by
      overlap (±margin) with 'Category=Anomaly' events from the official
      ESA-ADB annotations, and saves the dataset (cp_training_dataset.csv).

  --mode train-classifier
      Loads the dataset, performs a temporal train/val split (last 3 months
      of training = val, as in the paper), trains a balanced RandomForest,
      tunes the decision threshold on val for F0.5 (NOT on the test set),
      and saves the model and metrics.

  --mode test
      Runs the CPD+classifier pipeline on the test set: for every CP that
      passes the gate, predicts with the classifier and keeps only the CPs
      with prob >= threshold. Builds the pred_mask with the same window-wide
      logic as the baseline (apples-to-apples), saves the CSV in the format
      expected by evaluate() in pen_sweep_paper.py, and prints the corrected
      event-wise F0.5 metrics with rare events excluded.

Default output -> cpd_output_classified/
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import ruptures as rpt
import joblib

# Stesso meccanismo degli altri file: data_io vive in ../Jan/
sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_dataset, CHANNELS_41_46  # noqa: E402

from config import (  # noqa: E402
    CHANNELS,
    WINDOW_DAYS,
    CPD_MIN_SIZE, CPD_JUMP,
    PENALTY_SWEEP_STEPS, PENALTY_JUMP_THRESHOLD,
    STFT_NPERSEG, STFT_OVERLAP_FRAC,
    TRAIN_CSV, TEST_CSV,
)
from penalty_sweep import find_optimal_penalty, per_dimension_cost_analysis  # noqa: E402
from stft_utils import extract_all_frequency_timeseries_window  # noqa: E402
from pipeline import create_fixed_windows  # noqa: E402

from cp_features import extract_cp_features, FEATURE_NAMES  # noqa: E402
from pen_sweep_paper import (  # noqa: E402
    load_paper_events, evaluate,
    TARGET_CATEGORY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Costanti e default
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GAIN_THRESHOLD       = 290.85   # dal log_sweep_unsup.txt (run unsupervised)
DEFAULT_LABEL_MARGIN_NS      = int(3600 * 1e9)   # ±1h per labeling CP↔evento
DEFAULT_TRAIN_CSV_UNSUP      = "84_months.train.unsup_std.csv"
DEFAULT_TEST_CSV_UNSUP       = "84_months.test.unsup_std.csv"
DEFAULT_DATASET_CSV          = "cp_training_dataset.csv"
DEFAULT_MODEL_PATH           = "cp_classifier.joblib"
DEFAULT_METRICS_PATH         = "cp_classifier_metrics.json"
DEFAULT_OUTPUT_DIR           = "cpd_output_classified"

# Validation = ultimi 3 mesi del training (Mission1 train: 2000-01-01 → 2007-01-01).
VAL_START                    = pd.Timestamp("2006-10-01", tz="UTC")
VAL_END                      = pd.Timestamp("2007-01-01", tz="UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Window processing (replica di select_frequencies_and_cps, ma espone
# X_full, bkps_global e t_stamps necessari per estrarre feature per-CP)
# ─────────────────────────────────────────────────────────────────────────────

# Risultato di process_window: tupla di campi necessari downstream + status.
WindowResult = Tuple[
    Optional[np.ndarray],                      # X_full
    Optional[List[int]],                       # bkps_global
    Optional[List[Tuple[float, str]]],         # dim_labels
    float, int,                                # pen_optimal, k_knee
    Optional[pd.DatetimeIndex],                # t_stamps (frame STFT)
    str,                                       # status: "ok" o "skip:reason"
]


def _stft_frame_timestamps(W_index: pd.DatetimeIndex, n_frames: int) -> pd.DatetimeIndex:
    """
    Ricostruisce i timestamp dei frame STFT a partire dall'indice della
    finestra e dalla configurazione (STFT_NPERSEG, STFT_OVERLAP_FRAC).

    Frame i ha centro al campione: i * step + nperseg/2 (in W_index).
    step = nperseg * (1 - overlap_frac)
    """
    step = max(1, int(round(STFT_NPERSEG * (1.0 - STFT_OVERLAP_FRAC))))
    centers = np.arange(n_frames) * step + STFT_NPERSEG // 2
    centers = np.clip(centers, 0, len(W_index) - 1)
    return W_index[centers]


def process_window(
    W: pd.DataFrame,
    channels: List[str],
    gain_threshold: float,
) -> WindowResult:
    """
    Replica della pipeline di select_frequencies_and_cps, ma ritorna
    X_full e bkps_global per la feature extraction.
    Identica logica: STFT → sweep → PELT → gate sul max gain aggregato.
    """
    if W.dropna(how="all").empty:
        return (None, None, None, 0.0, 0, None, "skip:empty")

    # STFT → costruzione X_full
    freq_series_dict, _ = extract_all_frequency_timeseries_window(W, channels)
    if not freq_series_dict:
        return (None, None, None, 0.0, 0, None, "skip:stft_empty")

    feats: List[np.ndarray] = []
    dim_labels: List[Tuple[float, str]] = []
    for freq_hz in sorted(freq_series_dict.keys()):
        for ch in channels:
            if ch in freq_series_dict[freq_hz]:
                feats.append(freq_series_dict[freq_hz][ch])
                dim_labels.append((freq_hz, ch))
    if not feats:
        return (None, None, None, 0.0, 0, None, "skip:no_feats")

    X_full = np.column_stack(feats)
    n_frames = X_full.shape[0]
    if n_frames < 2:
        return (None, None, None, 0.0, 0, None, "skip:too_short")

    t_stamps = _stft_frame_timestamps(W.index, n_frames)

    # Sweep penalty
    try:
        pen_optimal, sweep_info = find_optimal_penalty(
            X_full,
            n_steps=PENALTY_SWEEP_STEPS,
            jump_threshold=PENALTY_JUMP_THRESHOLD,
        )
    except Exception as e:
        return (None, None, None, 0.0, 0, t_stamps, f"skip:sweep_fail:{e}")

    k_knee = int(sweep_info["k_optimal"])
    if k_knee == 0:
        return (None, None, None, pen_optimal, 0, t_stamps, "skip:k_knee_0")

    # PELT
    try:
        algo = rpt.Pelt(model="l2", min_size=CPD_MIN_SIZE, jump=CPD_JUMP).fit(X_full)
        bkps_global = algo.predict(pen=pen_optimal)
    except Exception as e:
        return (None, None, None, pen_optimal, k_knee, t_stamps, f"skip:pelt_fail:{e}")

    cp_idx = [b - 1 for b in bkps_global[:-1]]
    if not cp_idx:
        return (None, None, None, pen_optimal, k_knee, t_stamps, "skip:no_cp")

    # Gate sul max gain aggregato (identico all'unsupervised)
    gains_aggregate = per_dimension_cost_analysis(X_full, bkps_global)
    max_gain = float(gains_aggregate.max())
    if max_gain <= gain_threshold:
        return (None, None, None, pen_optimal, k_knee, t_stamps,
                f"skip:gain_gate:{max_gain:.2f}<={gain_threshold:.2f}")

    return (X_full, bkps_global, dim_labels, pen_optimal, k_knee, t_stamps, "ok")


def cp_timestamps_from_bkps(bkps_global: List[int], t_stamps: pd.DatetimeIndex):
    """Timestamp di ogni CP interno (esclude T finale)."""
    boundaries = sorted(bkps_global)[:-1]
    return [t_stamps[b] for b in boundaries if 0 <= b < len(t_stamps)]


# ─────────────────────────────────────────────────────────────────────────────
# Labeling: overlap CP ↔ eventi ESA-ADB
# ─────────────────────────────────────────────────────────────────────────────

def label_cps(
    cp_timestamps: List[pd.Timestamp],
    events_df: pd.DataFrame,
    margin_ns: int = DEFAULT_LABEL_MARGIN_NS,
) -> List[int]:
    """
    Per ogni CP timestamp:
      label = 1 se ts ∈ [evento.StartTime − margin, evento.EndTime + margin]
                 di un evento in events_df (filtrato a Anomaly upstream).
      label = 0 altrimenti.
    """
    if events_df is None or events_df.empty or len(cp_timestamps) == 0:
        return [0] * len(cp_timestamps)
    margin = pd.Timedelta(margin_ns, unit="ns")
    starts = pd.to_datetime(events_df["StartTime"].values, utc=True) - margin
    ends   = pd.to_datetime(events_df["EndTime"].values,   utc=True) + margin

    labels: List[int] = []
    for ts in cp_timestamps:
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("UTC")
        hit = bool(np.any((starts <= ts_pd) & (ts_pd <= ends)))
        labels.append(1 if hit else 0)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# --mode generate-train
# ─────────────────────────────────────────────────────────────────────────────

def generate_train_dataset(
    train_csv: str,
    gain_threshold: float,
    margin_ns: int,
    out_csv: str,
) -> None:
    print(f"\n{'#'*70}")
    print(f"# MODE: GENERATE-TRAIN")
    print(f"# Train CSV:      {train_csv}")
    print(f"# Gain threshold: {gain_threshold:.4f}")
    print(f"# Label margin:   ±{margin_ns / 1e9 / 60:.1f} min")
    print(f"# Output dataset: {out_csv}")
    print(f"{'#'*70}\n")

    df_train, _, _ = load_dataset(
        csv_path=train_csv, use_channels=CHANNELS_41_46,
        include_labels=False, interpolate=True, fill_strategy="both",
    )
    df_train = df_train.sort_index()
    present_channels = [c for c in CHANNELS if c in df_train.columns]
    print(f"[gen] {len(df_train)} samples, {len(present_channels)} canali")

    train_start = df_train.index[0]
    train_end   = df_train.index[-1]
    if train_start.tzinfo is None:
        train_start = train_start.tz_localize("UTC")
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    print(f"[gen] Periodo training: {train_start} → {train_end}")

    # Annotazioni ufficiali ESA-ADB, ristrette al periodo training
    anom_events = load_paper_events(
        category=TARGET_CATEGORY,
        channels=CHANNELS,
        t_start=train_start,
        t_end=train_end,
    )
    print(f"[gen] Eventi 'Anomaly' nel periodo training: {len(anom_events)}")

    windows = create_fixed_windows(df_train, window_days=WINDOW_DAYS)
    n_win = len(windows)
    print(f"[gen] {n_win} finestre da {WINDOW_DAYS} giorni\n")

    rows: List[dict] = []
    n_passed_gate = 0
    n_skipped     = 0

    for w_id, (win_start, win_end) in enumerate(windows, 1):
        W = df_train.loc[win_start:win_end, present_channels]
        X_full, bkps, dim_labels, pen_opt, k_knee, t_stamps, status = \
            process_window(W, present_channels, gain_threshold)

        if status != "ok" or X_full is None:
            n_skipped += 1
            if w_id % 20 == 0:
                print(f"  [gen] {w_id}/{n_win} — passed={n_passed_gate} "
                      f"skipped={n_skipped} rows={len(rows)}")
            continue

        n_passed_gate += 1
        cp_feats = extract_cp_features(X_full, bkps, dim_labels, pen_opt, k_knee)
        cp_ts    = cp_timestamps_from_bkps(bkps, t_stamps)
        n_cp     = min(len(cp_feats), len(cp_ts))
        if n_cp == 0:
            if w_id % 20 == 0:
                print(f"  [gen] {w_id}/{n_win} — passed={n_passed_gate} "
                      f"skipped={n_skipped} rows={len(rows)}")
            continue

        cp_labels = label_cps(cp_ts[:n_cp], anom_events, margin_ns)

        for feat, ts, lab in zip(cp_feats[:n_cp], cp_ts[:n_cp], cp_labels):
            ts_pd = pd.Timestamp(ts)
            if ts_pd.tzinfo is None:
                ts_pd = ts_pd.tz_localize("UTC")
            in_val = (VAL_START <= ts_pd) and (ts_pd < VAL_END)
            row = dict(feat)
            row["window_id"]    = w_id
            row["cp_timestamp"] = str(ts_pd)
            row["label"]        = int(lab)
            row["fold"]         = "val" if in_val else "train"
            rows.append(row)

        if w_id % 20 == 0:
            n_pos = sum(1 for r in rows if r["label"] == 1)
            print(f"  [gen] {w_id}/{n_win} — passed={n_passed_gate} "
                  f"skipped={n_skipped} rows={len(rows)} pos={n_pos}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"GENERATE-TRAIN done")
    print(f"  Finestre processate: {n_win}")
    print(f"  Passate gate:        {n_passed_gate}")
    print(f"  Skippate:            {n_skipped}")
    print(f"  Righe (CP):          {len(rows)}")
    if rows:
        n_pos       = int(df["label"].sum())
        n_train_fld = int((df["fold"] == "train").sum())
        n_val_fld   = int((df["fold"] == "val").sum())
        n_pos_val   = int(((df["fold"] == "val") & (df["label"] == 1)).sum())
        print(f"  Positivi totali:     {n_pos} ({100 * n_pos / len(rows):.1f}%)")
        print(f"  Train fold:          {n_train_fld}")
        print(f"  Val fold:            {n_val_fld}  (pos={n_pos_val})")
    print(f"  CSV:                 {out_csv}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# --mode train-classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    dataset_csv:  str,
    model_out:    str,
    metrics_out:  str,
    beta:         float = 0.5,
    val_strategy: str   = "stratified-window",
    val_size:     float = 0.25,
    random_state: int   = 42,
) -> None:
    """
    val_strategy:
      "temporal"           — usa la colonna `fold` del CSV (ultimi 3 mesi = val).
                             Funziona solo se il val temporale contiene positivi.
      "stratified-window"  — IGNORA la colonna `fold`. Splitta le `window_id`
                             stratificando per "la finestra contiene almeno un
                             CP positivo". Garantisce positivi in val e
                             nessun leakage CP-finestra tra train e val.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, fbeta_score,
    )

    print(f"\n{'#'*70}")
    print(f"# MODE: TRAIN-CLASSIFIER")
    print(f"# Dataset:      {dataset_csv}")
    print(f"# Model out:    {model_out}")
    print(f"# Metrics:      {metrics_out}")
    print(f"# β:            {beta}")
    print(f"# Val strategy: {val_strategy}  (val_size={val_size}, seed={random_state})")
    print(f"{'#'*70}\n")

    df = pd.read_csv(dataset_csv)
    print(f"[train] Dataset: {len(df)} righe")
    if "label" not in df.columns:
        raise ValueError(f"Dataset {dataset_csv} non contiene colonna 'label'")

    if val_strategy == "temporal":
        if "fold" not in df.columns:
            raise ValueError("val-strategy=temporal richiede colonna 'fold' nel dataset")
        df_tr = df[df["fold"] == "train"].copy()
        df_va = df[df["fold"] == "val"].copy()
    elif val_strategy == "stratified-window":
        if "window_id" not in df.columns:
            raise ValueError("val-strategy=stratified-window richiede colonna 'window_id'")
        # Split sulle window_id stratificato per "la finestra contiene ≥1 positivo"
        win_pos = df.groupby("window_id")["label"].max().reset_index()
        win_pos.columns = ["window_id", "has_pos"]
        n_pos_w = int(win_pos["has_pos"].sum())
        n_neg_w = int((win_pos["has_pos"] == 0).sum())
        print(f"[train] Finestre uniche: {len(win_pos)}  (con pos: {n_pos_w}, senza: {n_neg_w})")
        if n_pos_w < 2:
            raise ValueError(
                f"Solo {n_pos_w} finestre con positivi — impossibile stratificare. "
                "Servono ≥2 finestre positive."
            )
        train_wins, val_wins = train_test_split(
            win_pos["window_id"].values,
            test_size=val_size,
            stratify=win_pos["has_pos"].values,
            random_state=random_state,
        )
        df_tr = df[df["window_id"].isin(train_wins)].copy()
        df_va = df[df["window_id"].isin(val_wins)].copy()
        n_pos_tr_w = int(df_tr.groupby("window_id")["label"].max().sum())
        n_pos_va_w = int(df_va.groupby("window_id")["label"].max().sum())
        print(f"[train] Stratified-window split:")
        print(f"  train: {len(train_wins)} finestre ({n_pos_tr_w} con pos)")
        print(f"  val:   {len(val_wins)} finestre ({n_pos_va_w} con pos)")
    else:
        raise ValueError(f"val_strategy sconosciuta: {val_strategy}")

    print(f"[train] Train: {len(df_tr)} CP  (pos={int(df_tr['label'].sum())})")
    print(f"[train] Val:   {len(df_va)} CP  (pos={int(df_va['label'].sum())})")

    if df_tr.empty or df_va.empty:
        raise ValueError("Train o val fold vuoti — controlla il dataset generato.")

    # Verifica colonne
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne feature nel dataset: {missing}")

    X_tr = df_tr[FEATURE_NAMES].values.astype(float)
    y_tr = df_tr["label"].values.astype(int)
    X_va = df_va[FEATURE_NAMES].values.astype(float)
    y_va = df_va["label"].values.astype(int)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)

    p_va = clf.predict_proba(X_va_s)[:, 1]

    if y_va.sum() > 0 and y_va.sum() < len(y_va):
        roc = float(roc_auc_score(y_va, p_va))
        pr  = float(average_precision_score(y_va, p_va))
    else:
        print("[train] ⚠ Val fold senza variabilità (tutti pos o tutti neg) → AUC=NaN")
        roc = float("nan")
        pr  = float("nan")

    # Tara soglia su val per max F-β (default β=0.5 come il paper)
    best_thr = 0.5
    best_fb  = -1.0
    if y_va.sum() > 0 and y_va.sum() < len(y_va):
        for thr in np.linspace(0.01, 0.99, 99):
            pred = (p_va >= thr).astype(int)
            fb = float(fbeta_score(y_va, pred, beta=beta, zero_division=0))
            if fb > best_fb:
                best_fb = fb
                best_thr = float(thr)

    f1_at_best = float(f1_score(y_va, (p_va >= best_thr).astype(int), zero_division=0))

    print(f"\n[train] Metriche val:")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"  PR-AUC:    {pr:.4f}")
    print(f"  Best thr:  {best_thr:.3f}")
    print(f"  F-β@best:  {best_fb:.4f}  (β={beta})")
    print(f"  F1@best:   {f1_at_best:.4f}")

    importance = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: -x[1],
    )
    print(f"\n[train] Feature importance (top 10):")
    for name, imp in importance[:10]:
        print(f"  {name:25s}  {imp:.4f}")

    # Persisti modello + scaler + soglia + feature names
    joblib.dump({
        "model":         clf,
        "scaler":        scaler,
        "threshold":     best_thr,
        "feature_names": FEATURE_NAMES,
        "beta":          beta,
    }, model_out)

    metrics = {
        "roc_auc_val":         roc,
        "pr_auc_val":          pr,
        "best_threshold":      best_thr,
        "f_beta_val":          best_fb,
        "f1_val":              f1_at_best,
        "beta":                beta,
        "n_train":             int(len(df_tr)),
        "n_val":               int(len(df_va)),
        "n_train_positives":   int(df_tr["label"].sum()),
        "n_val_positives":     int(df_va["label"].sum()),
        "feature_importance":  [
            {"name": n, "importance": float(i)} for n, i in importance
        ],
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[train] ✓ Modello: {model_out}")
    print(f"[train] ✓ Metriche: {metrics_out}")
    print(f"{'='*70}\n")

    if not (isinstance(roc, float) and roc > 0.7):
        print("⚠ ATTENZIONE: ROC-AUC val ≤ 0.7. Il classificatore non distingue bene.")
        print("  → considerare: più feature, più dati, o accettare negative result.\n")


# ─────────────────────────────────────────────────────────────────────────────
# --mode test
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    test_csv:       str,
    gain_threshold: float,
    model_path:     str,
    output_dir:     str,
    beta:           float = 0.5,
) -> None:
    print(f"\n{'#'*70}")
    print(f"# MODE: TEST")
    print(f"# Test CSV:      {test_csv}")
    print(f"# Gain threshold: {gain_threshold:.4f}")
    print(f"# Model:         {model_path}")
    print(f"# Output dir:    {output_dir}")
    print(f"{'#'*70}\n")

    bundle = joblib.load(model_path)
    clf:           Any                = bundle["model"]
    scaler:        Any                = bundle["scaler"]
    decision_thr:  float              = bundle["threshold"]
    feature_names: List[str]          = bundle["feature_names"]
    print(f"[test] Modello caricato. Soglia decisione = {decision_thr:.3f}")

    df_test, _, _ = load_dataset(
        csv_path=test_csv, use_channels=CHANNELS_41_46,
        include_labels=False, interpolate=True, fill_strategy="both",
    )
    df_test = df_test.sort_index()
    present_channels = [c for c in CHANNELS if c in df_test.columns]
    print(f"[test] {len(df_test)} samples")

    windows = create_fixed_windows(df_test, window_days=WINDOW_DAYS)
    n_win = len(windows)
    print(f"[test] {n_win} finestre da {WINDOW_DAYS} giorni\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    n_passed_gate    = 0
    n_cp_total       = 0
    n_cp_kept_total  = 0

    for w_id, (win_start, win_end) in enumerate(windows, 1):
        W = df_test.loc[win_start:win_end, present_channels]
        X_full, bkps, dim_labels, pen_opt, k_knee, t_stamps, status = \
            process_window(W, present_channels, gain_threshold)

        # Riga di base (anche per finestre skipped, così la CSV ha 1 riga/finestra)
        base = {
            "window":             w_id,
            "win_start":          str(win_start),
            "win_end":            str(win_end),
            "gain_gate_passed":   False,
            "status":             status,
            "n_cps_initial":      0,
            "n_cps_kept":         0,
            "n_changepoints":     0,   # ← compat con evaluate()/build_predicted_mask_from_results
            "pen_optimal":        float(pen_opt),
            "k_knee":             int(k_knee),
            "cp_timestamps_kept": "[]",
            "cp_proba":           "[]",
        }

        if status != "ok" or X_full is None:
            results.append(base)
            if w_id % 20 == 0:
                print(f"  [test] {w_id}/{n_win} — gate_passed={n_passed_gate} "
                      f"cp_kept={n_cp_kept_total}/{n_cp_total}")
            continue

        n_passed_gate += 1
        cp_feats = extract_cp_features(X_full, bkps, dim_labels, pen_opt, k_knee)
        cp_ts    = cp_timestamps_from_bkps(bkps, t_stamps)
        n_cp     = min(len(cp_feats), len(cp_ts))
        n_cp_total += n_cp

        if n_cp == 0:
            base["gain_gate_passed"] = True
            base["status"]           = "ok_but_no_cp"
            results.append(base)
            if w_id % 20 == 0:
                print(f"  [test] {w_id}/{n_win} — gate_passed={n_passed_gate} "
                      f"cp_kept={n_cp_kept_total}/{n_cp_total}")
            continue

        # Predizione classificatore
        feat_df = pd.DataFrame(cp_feats[:n_cp])[feature_names].astype(float)
        X_cp    = scaler.transform(feat_df.values)
        proba   = clf.predict_proba(X_cp)[:, 1]
        keep    = proba >= decision_thr
        cp_ts_kept = [cp_ts[i] for i in range(n_cp) if keep[i]]
        n_kept     = int(len(cp_ts_kept))
        n_cp_kept_total += n_kept

        base.update({
            "gain_gate_passed":   True,
            "status":             "ok",
            "n_cps_initial":      n_cp,
            "n_cps_kept":         n_kept,
            "n_changepoints":     n_kept,
            "cp_timestamps_kept": json.dumps([str(t) for t in cp_ts_kept]),
            "cp_proba":           json.dumps([float(p) for p in proba.tolist()]),
        })
        results.append(base)

        if w_id % 20 == 0:
            print(f"  [test] {w_id}/{n_win} — gate_passed={n_passed_gate} "
                  f"cp_kept={n_cp_kept_total}/{n_cp_total}")

    csv_path = Path(output_dir) / "test_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print(f"TEST done")
    print(f"  Finestre processate:   {n_win}")
    print(f"  Passate gate gain:     {n_passed_gate}")
    print(f"  CP totali post-gate:   {n_cp_total}")
    print(f"  CP tenuti dal clf:     {n_cp_kept_total} "
          f"({100 * n_cp_kept_total / max(1, n_cp_total):.1f}%)")
    print(f"  CSV salvato:           {csv_path}")
    print(f"{'='*70}\n")

    print(f"[test] Chiamata evaluate() per metriche paper-aligned...")
    evaluate(test_csv=test_csv, results_csv=str(csv_path), beta=beta)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        choices=["generate-train", "train-classifier", "test"],
        required=True,
        help="Modalità di esecuzione.",
    )
    ap.add_argument(
        "--train-csv", default=DEFAULT_TRAIN_CSV_UNSUP,
        help=f"CSV training (default unsupervised: {DEFAULT_TRAIN_CSV_UNSUP}). "
             "Se non esiste, fallback a TRAIN_CSV originale.",
    )
    ap.add_argument(
        "--test-csv", default=DEFAULT_TEST_CSV_UNSUP,
        help=f"CSV test (default unsupervised: {DEFAULT_TEST_CSV_UNSUP}). "
             "Se non esiste, fallback a TEST_CSV originale.",
    )
    ap.add_argument(
        "--gain-threshold", type=float, default=DEFAULT_GAIN_THRESHOLD,
        help=f"Soglia gain del gate unsupervised (default: {DEFAULT_GAIN_THRESHOLD}).",
    )
    ap.add_argument(
        "--margin-min", type=float, default=60.0,
        help="Margine ±minuti per labeling CP↔evento (default: 60).",
    )
    ap.add_argument(
        "--dataset", default=DEFAULT_DATASET_CSV,
        help=f"Path del dataset training del classificatore (default: {DEFAULT_DATASET_CSV}).",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help=f"Path del modello classificatore (default: {DEFAULT_MODEL_PATH}).",
    )
    ap.add_argument(
        "--metrics", default=DEFAULT_METRICS_PATH,
        help=f"Path delle metriche val (default: {DEFAULT_METRICS_PATH}).",
    )
    ap.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Cartella output test (default: {DEFAULT_OUTPUT_DIR}).",
    )
    ap.add_argument(
        "--beta", type=float, default=0.5,
        help="β dell'F-score (default: 0.5, come il paper).",
    )
    ap.add_argument(
        "--val-strategy",
        choices=["temporal", "stratified-window"],
        default="stratified-window",
        help="Strategia di split train/val per --mode train-classifier. "
             "'temporal' = ultimi 3 mesi (richiede positivi nel val temporale). "
             "'stratified-window' (default) = split sulle window_id stratificato "
             "per 'finestra contiene ≥1 positivo', garantisce positivi in val.",
    )
    ap.add_argument(
        "--val-size", type=float, default=0.25,
        help="Frazione di finestre nel val fold per stratified-window (default: 0.25).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed per lo split stratified-window (default: 42).",
    )
    args = ap.parse_args()

    # Fallback su CSV non-standardizzato se l'unsupervised non esiste
    train_csv = args.train_csv
    if not Path(train_csv).exists():
        print(f"⚠ {train_csv} non trovato → fallback su {TRAIN_CSV}")
        train_csv = TRAIN_CSV

    test_csv = args.test_csv
    if not Path(test_csv).exists():
        print(f"⚠ {test_csv} non trovato → fallback su {TEST_CSV}")
        test_csv = TEST_CSV

    margin_ns = int(args.margin_min * 60 * 1e9)

    if args.mode == "generate-train":
        generate_train_dataset(
            train_csv=train_csv,
            gain_threshold=args.gain_threshold,
            margin_ns=margin_ns,
            out_csv=args.dataset,
        )
    elif args.mode == "train-classifier":
        train_classifier(
            dataset_csv=args.dataset,
            model_out=args.model,
            metrics_out=args.metrics,
            beta=args.beta,
            val_strategy=args.val_strategy,
            val_size=args.val_size,
            random_state=args.seed,
        )
    elif args.mode == "test":
        run_test(
            test_csv=test_csv,
            gain_threshold=args.gain_threshold,
            model_path=args.model,
            output_dir=args.output_dir,
            beta=args.beta,
        )


if __name__ == "__main__":
    main()
