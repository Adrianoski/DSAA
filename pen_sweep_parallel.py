"""
pen_sweep_parallel.py
===========================
Parallel version of pen_sweep_update.py.

Uses multiprocessing.Pool con fino a 20 core per parallelizzare:
  1. Calibrazione soglia gain (finestre nominali training)
  2. Loop test (finestre test indipendenti)

"""

import sys
import os
import io
import contextlib
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import ruptures as rpt

from data_io import load_dataset, CHANNELS_41_46

from config import (
    CHANNELS, LABELS, CATS,
    WINDOW_DAYS,
    CPD_MIN_SIZE, CPD_JUMP,
    PR_MARGIN_FR, ALPHA_RECALL, RECALL_BIAS, PRECISION_BIAS, BETA_SCORE,
    MAX_GAP_SECONDS, MIN_CLUSTER_IOU,
    MAX_FEATURES,
    PENALTY_SWEEP_STEPS, PENALTY_JUMP_THRESHOLD,
    TRAIN_CSV, TEST_CSV,
)
from penalty_sweep import find_optimal_penalty
from stft_utils import (
    extract_all_frequency_timeseries_window,
    extract_features_from_selected_freqs,
)
from metrics import compute_timeseries_metrics, compute_classic_pr, compute_iou_corrected
from pipeline import create_fixed_windows, build_lab_table, _plot_cpd_result

from pen_sweep_update import (
    select_frequencies_and_cps,
    NOMINAL_GAIN_PERCENTILE,
)
from penalty_sweep import per_dimension_cost_analysis

N_WORKERS       = min(6, cpu_count() or 1)
OUTPUT_DIR_PAR  = "cpd_parallel_output"


# ─────────────────────────────────────────────────────────────────────────────
# Stato locale al processo worker (caricato una sola volta per processo)
# ─────────────────────────────────────────────────────────────────────────────

_worker_df       = None
_worker_channels = None
_worker_labels   = None


def _init_worker(csv_path: str, use_labels: bool):
    """Initializer: carica il dataset una volta per processo worker."""
    global _worker_df, _worker_channels, _worker_labels
    df, _, _ = load_dataset(
        csv_path=csv_path, use_channels=CHANNELS_41_46,
        include_labels=use_labels, interpolate=True, fill_strategy="both",
    )
    df = df.sort_index()
    ch = [c for c in CHANNELS if c in df.columns]
    for c in ch:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if use_labels:
        lb = [l for l in LABELS if l in df.columns]
        for l in lb:
            df[l] = pd.to_numeric(df[l], errors="coerce").astype("Int64")
    else:
        lb = []
    _worker_df       = df
    _worker_channels = ch
    _worker_labels   = lb


# ─────────────────────────────────────────────────────────────────────────────
# Task calibrazione: una finestra nominale del training
# ─────────────────────────────────────────────────────────────────────────────

def _calibration_task(args):
    """
    Restituisce max(gain) per la finestra nominale, oppure None se:
    - la finestra è vuota
    - lo sweep non trova struttura (k_optimal == 0)
    - PELT non trova CP
    """
    win_start, win_end = args

    W = _worker_df.loc[win_start:win_end, _worker_channels]
    if W.dropna(how="all").empty:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        freq_series_dict, _ = extract_all_frequency_timeseries_window(W, _worker_channels)
    if not freq_series_dict:
        return None

    feats = []
    for freq_hz in sorted(freq_series_dict.keys()):
        for ch in _worker_channels:
            if ch in freq_series_dict[freq_hz]:
                feats.append(freq_series_dict[freq_hz][ch])
    if not feats:
        return None

    X_full = np.column_stack(feats)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pen_opt, sweep_info = find_optimal_penalty(
                X_full,
                n_steps=PENALTY_SWEEP_STEPS,
                jump_threshold=PENALTY_JUMP_THRESHOLD,
            )
    except Exception:
        return None

    if sweep_info["k_optimal"] == 0:
        return None

    try:
        algo = rpt.Pelt(model="l2", min_size=CPD_MIN_SIZE, jump=CPD_JUMP).fit(X_full)
        bkps = algo.predict(pen=pen_opt)
    except Exception:
        return None

    if len(bkps) <= 1:
        return None

    gains = per_dimension_cost_analysis(X_full, bkps)
    max_g = float(gains.max())
    return max_g if max_g > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Task test: una singola finestra del test set
# ─────────────────────────────────────────────────────────────────────────────

def _test_task(args):
    """
    Esegue la pipeline CPD completa su una finestra di test.
    stdout soppresso; ritorna (w_id, result_dict | None).
    """
    (w_id, win_start, win_end,
     events_records, has_gt,
     gain_threshold, focus_ch,
     output_dir, save_plots) = args

    # Ricostruisce events_in_win dal record serializzato
    if events_records:
        events_in_win = pd.DataFrame(events_records)
        events_in_win["StartTime"] = pd.to_datetime(events_in_win["StartTime"])
        events_in_win["EndTime"]   = pd.to_datetime(events_in_win["EndTime"])
    else:
        events_in_win = pd.DataFrame(columns=["StartTime", "EndTime", "Category", "Channel"])

    W = _worker_df.loc[win_start:win_end, _worker_channels]
    if W.dropna(how="all").empty:
        return w_id, None

    # Pipeline CPD (stdout soppresso)
    with contextlib.redirect_stdout(io.StringIO()):
        selected_freqs, cp_idx, info = select_frequencies_and_cps(
            W, _worker_channels,
            gain_threshold=gain_threshold,
            verbose=False,
        )

    if not selected_freqs:
        return w_id, None

    pen_optimal = info["pen_optimal"]
    k_optimal   = info["k_optimal"]
    n_freqs     = len(selected_freqs)

    with contextlib.redirect_stdout(io.StringIO()):
        X, t_stamps, feat_labels = extract_features_from_selected_freqs(
            _worker_df, win_start, win_end, selected_freqs, _worker_channels
        )
    if X is None:
        return w_id, None

    nT, D = X.shape
    bkps  = info["bkps_global"]

    # Maschera GT
    lab_mask = np.zeros(nT, dtype=bool)
    if has_gt:
        for _, r in events_in_win.iterrows():
            lab_mask |= (
                (t_stamps >= np.datetime64(r["StartTime"])) &
                (t_stamps <= np.datetime64(r["EndTime"]))
            )

    # Metriche
    if has_gt:
        true_cps_frames = []
        for _, r in events_in_win.iterrows():
            s_idx = int(np.clip(np.searchsorted(t_stamps, np.datetime64(r["StartTime"])), 0, nT - 1))
            e_idx = int(np.clip(np.searchsorted(t_stamps, np.datetime64(r["EndTime"])), 0, nT - 1))
            true_cps_frames.extend([s_idx, e_idx])
        true_cps_frames = sorted(set(i for i in true_cps_frames if 0 <= i < nT - 1))
        true_bkps = sorted(set([min(i + 1, nT - 1) for i in true_cps_frames] + [nT]))

        prec_classic, rec_classic = (
            compute_classic_pr(true_bkps, bkps, true_cps_frames, cp_idx)
            if true_cps_frames else (0.0, 0.0)
        )
        f1_classic = (
            2 * prec_classic * rec_classic / (prec_classic + rec_classic)
            if (prec_classic + rec_classic) > 0 else 0.0
        )
        with contextlib.redirect_stdout(io.StringIO()):
            metrics_T = compute_timeseries_metrics(
                cp_idx, lab_mask.astype(int), nT,
                alpha=ALPHA_RECALL, recall_bias=RECALL_BIAS,
                precision_bias=PRECISION_BIAS, beta=BETA_SCORE,
            )
            iou_total, selected_clusters, _, detailed_info = compute_iou_corrected(
                cp_idx, lab_mask.astype(int), nT, t_stamps,
                max_gap_seconds=MAX_GAP_SECONDS,
                min_cluster_iou=MIN_CLUSTER_IOU, verbose=False,
            )
    else:
        prec_classic = rec_classic = f1_classic = 0.0
        metrics_T = {"precision_T": 0.0, "recall_T": 0.0, "f_beta": 0.0,
                     "n_real_ranges": 0, "n_pred_ranges": 0}
        iou_total         = 0.0
        selected_clusters = []
        detailed_info     = {
            "n_anomalies": 0, "n_segments": 0, "n_clusters": 0,
            "n_selected": 0, "gt_anomalies": [],
            "cluster_matches": [], "iou_per_anomaly": [],
        }

    if save_plots:
        with contextlib.redirect_stdout(io.StringIO()):
            _plot_cpd_result(
                w_id, W, X, t_stamps, cp_idx, selected_clusters,
                events_in_win, feat_labels, focus_ch,
                prec_classic, rec_classic, f1_classic,
                metrics_T["precision_T"], metrics_T["recall_T"],
                metrics_T["f_beta"], iou_total,
                pen_optimal, D, detailed_info, output_dir, n_freqs,
                has_gt=has_gt, sweep_info=info["sweep_info"],
            )

    result = {
        "window":              w_id,
        "gt_label":            "ANOMALY" if has_gt else "NOMINAL",
        "detection":           "CORRECT" if (has_gt == (len(cp_idx) > 0)) else "WRONG",
        "win_start":           str(win_start),
        "win_end":             str(win_end),
        "has_gt_anomalies":    has_gt,
        "n_selected_freqs":    n_freqs,
        "selected_freqs":      selected_freqs,
        "n_features":          D,
        "pen_optimal":         pen_optimal,
        "k_knee":              k_optimal,
        "n_changepoints":      len(cp_idx),
        "n_anomalies":         detailed_info["n_anomalies"],
        "n_clusters":          detailed_info.get("n_clusters", 0),
        "n_selected_clusters": detailed_info.get("n_selected", 0),
        "precision_classic":   prec_classic,
        "recall_classic":      rec_classic,
        "f1_classic":          f1_classic,
        "precision_T":         metrics_T["precision_T"],
        "recall_T":            metrics_T["recall_T"],
        "f_beta":              metrics_T["f_beta"],
        "iou_total":           iou_total,
        "n_real_ranges":       metrics_T["n_real_ranges"],
        "n_pred_ranges":       metrics_T["n_pred_ranges"],
    }
    return w_id, result


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1: calibrazione parallela soglia gain
# ─────────────────────────────────────────────────────────────────────────────

def build_nominal_gain_threshold_parallel(
    train_csv: str    = TRAIN_CSV,
    percentile: float = NOMINAL_GAIN_PERCENTILE,
    n_workers: int    = N_WORKERS,
) -> float:
    print(f"\n{'─'*70}")
    print(f"Calibrazione soglia gain nominale (percentile={percentile}, workers={n_workers})")

    # Carica il training nel processo principale solo per costruire la lista finestre
    df_train, _, _ = load_dataset(
        csv_path=train_csv, use_channels=CHANNELS_41_46,
        include_labels=True, interpolate=True, fill_strategy="both",
    )
    df_train = df_train.sort_index()
    for c in [c for c in CHANNELS if c in df_train.columns]:
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
    present_labels = [l for l in LABELS if l in df_train.columns]
    for l in present_labels:
        df_train[l] = pd.to_numeric(df_train[l], errors="coerce").astype("Int64")

    lab_train     = build_lab_table(df_train, present_labels)
    windows_train = create_fixed_windows(df_train, window_days=WINDOW_DAYS)

    # Filtra solo finestre nominali
    nominal_windows = []
    for win_start, win_end in windows_train:
        events = (
            lab_train[
                (lab_train["StartTime"] <= win_end) &
                (lab_train["EndTime"]   >= win_start)
            ]
            if not lab_train.empty else pd.DataFrame()
        )
        if events.empty:
            nominal_windows.append((win_start, win_end))

    print(f"  Finestre training totali: {len(windows_train)}  |  Nominali: {len(nominal_windows)}")

    # Pool con initializer: ogni worker carica df_train una sola volta
    tasks = nominal_windows
    max_gains = []
    completed = 0

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(train_csv, True),
    ) as pool:
        for max_g in pool.imap_unordered(_calibration_task, tasks, chunksize=4):
            completed += 1
            if max_g is not None:
                max_gains.append(max_g)
            if completed % 50 == 0 or completed == len(tasks):
                pct_str = (
                    f"{np.percentile(max_gains, percentile):.3f}"
                    if max_gains else "n/a"
                )
                print(f"  [calibrazione] {completed}/{len(tasks)} nom — "
                      f"{len(max_gains)} con CP — soglia corrente: {pct_str}")

    if not max_gains:
        print("  ⚠ Nessuna finestra nominale con CP — soglia=0 (nessun filtraggio)")
        return 0.0

    threshold = float(np.percentile(max_gains, percentile))
    print(f"  μ={np.mean(max_gains):.3f}  σ={np.std(max_gains):.3f}  "
          f"max={np.max(max_gains):.3f}")
    print(f"  ✓ Soglia gain ({percentile}° percentile): {threshold:.4f}")
    print(f"{'─'*70}\n")
    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Fase 2: test parallelo
# ─────────────────────────────────────────────────────────────────────────────

def run_test_parallel(
    test_csv:       str   = TEST_CSV,
    gain_threshold: float = 0.0,
    output_dir:     str   = OUTPUT_DIR_PAR,
    save_plots:     bool  = True,
    n_workers:      int   = N_WORKERS,
) -> list:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = str(Path(output_dir) / "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Carica test nel processo principale per costruire la lista finestre e lab
    df_test, _, _ = load_dataset(
        csv_path=test_csv, use_channels=CHANNELS_41_46,
        include_labels=True, interpolate=True, fill_strategy="both",
    )
    df_test = df_test.sort_index()
    present_channels = [c for c in CHANNELS if c in df_test.columns]
    present_labels   = [l for l in LABELS   if l in df_test.columns]
    for c in present_channels:
        df_test[c] = pd.to_numeric(df_test[c], errors="coerce")
    for l in present_labels:
        df_test[l] = pd.to_numeric(df_test[l], errors="coerce").astype("Int64")

    lab_sel = build_lab_table(df_test, present_labels)

    cat_to_val  = {"Nominal": 0, "Rare Event": 1, "Anomaly": 2, "Communication gap": 3}
    target_vals = {cat_to_val[c] for c in CATS}
    cov      = {ch: int(df_test.get(f"is_anomaly_{ch}", pd.Series([])).isin(target_vals).sum())
                for ch in present_channels}
    focus_ch = max(cov, key=cov.get)

    windows   = create_fixed_windows(df_test, window_days=WINDOW_DAYS)
    n_windows = len(windows)

    print(f"{'='*70}")
    print(f"TEST PARALLELO CPD — {WINDOW_DAYS}gg/finestra  |  "
          f"{n_windows} finestre  |  {n_workers} worker")
    print(f"  Canale focus: {focus_ch}  |  GT anomalie: {len(lab_sel)}")
    print(f"  Soglia gain: {gain_threshold:.4f}")
    print(f"{'='*70}\n")

    # Costruisce la lista di task
    tasks = []
    for w_id, (win_start, win_end) in enumerate(windows, 1):
        events_in_win = (
            lab_sel[
                (lab_sel["StartTime"] <= win_end) &
                (lab_sel["EndTime"]   >= win_start)
            ]
            if not lab_sel.empty else pd.DataFrame()
        )
        has_gt = not events_in_win.empty
        # Serializza events come lista di dict (picklable)
        events_records = (
            events_in_win.assign(
                StartTime=events_in_win["StartTime"].astype(str),
                EndTime=events_in_win["EndTime"].astype(str),
            ).to_dict("records")
            if not events_in_win.empty else []
        )
        tasks.append((
            w_id, win_start, win_end,
            events_records, has_gt,
            gain_threshold, focus_ch,
            plots_dir, save_plots,
        ))

    # Pool con initializer: ogni worker carica df_test una sola volta
    results_map = {}
    completed   = 0

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(test_csv, True),
    ) as pool:
        for w_id, result in pool.imap_unordered(_test_task, tasks, chunksize=2):
            completed += 1
            if result is not None:
                results_map[w_id] = result
                status = "ANOM" if result["has_gt_anomalies"] else "NOM "
                cp_str = f"CP={result['n_changepoints']}"
                gate   = "" if result["n_changepoints"] > 0 else " [GATE]"
                print(f"  [{status}] win {w_id:03d} ({completed:3d}/{n_windows}) | "
                      f"{cp_str}{gate} | "
                      f"P={result['precision_classic']:.3f} "
                      f"R={result['recall_classic']:.3f} "
                      f"F1={result['f1_classic']:.3f}")
            else:
                print(f"  [SKIP] win {w_id:03d} ({completed:3d}/{n_windows})")

    # Riordina per w_id
    results = [results_map[k] for k in sorted(results_map)]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline completa
# ─────────────────────────────────────────────────────────────────────────────

def run_complete_pipeline_parallel(
    train_csv:  str   = TRAIN_CSV,
    test_csv:   str   = TEST_CSV,
    output_dir: str   = OUTPUT_DIR_PAR,
    save_plots: bool  = True,
    n_workers:  int   = N_WORKERS,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# PIPELINE CPD PARALLELA  ({n_workers} worker)")
    print(f"{'#'*70}\n")

    # ── Fase 1: calibrazione ──────────────────────────────────────────────
    gain_threshold = build_nominal_gain_threshold_parallel(
        train_csv=train_csv,
        percentile=NOMINAL_GAIN_PERCENTILE,
        n_workers=n_workers,
    )

    # ── Fase 2: test ──────────────────────────────────────────────────────
    results = run_test_parallel(
        test_csv=test_csv,
        gain_threshold=gain_threshold,
        output_dir=output_dir,
        save_plots=save_plots,
        n_workers=n_workers,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    n_gt  = sum(1 for r in results if r["has_gt_anomalies"])
    n_nom = len(results) - n_gt

    anom_res = [r for r in results if r["has_gt_anomalies"]]
    g_pc  = [r["precision_classic"] for r in anom_res]
    g_rc  = [r["recall_classic"]    for r in anom_res]
    g_f1c = [r["f1_classic"]        for r in anom_res]
    g_pT  = [r["precision_T"]       for r in anom_res]
    g_rT  = [r["recall_T"]          for r in anom_res]
    g_fbT = [r["f_beta"]            for r in anom_res]
    g_iou = [r["iou_total"]         for r in anom_res]

    print(f"\n{'='*70}")
    print(f"✓ COMPLETATO: {len(results)} finestre  "
          f"(anomale={n_gt}  nominali={n_nom})")

    if g_f1c:
        print(f"\n{'CLASSIC':-^70}")
        print(f"  P={np.mean(g_pc):.3f}  R={np.mean(g_rc):.3f}  F1={np.mean(g_f1c):.3f}")
        print(f"\n{'RANGE-BASED':-^70}")
        print(f"  P_T={np.mean(g_pT):.3f}  R_T={np.mean(g_rT):.3f}  "
              f"F-β={np.mean(g_fbT):.3f}  IoU={np.mean(g_iou):.3f}")

    if results:
        correct = sum(1 for r in results if r["detection"] == "CORRECT")
        nom_with_cp = sum(
            1 for r in results if not r["has_gt_anomalies"] and r["n_changepoints"] > 0
        )
        print(f"\n  Detection accuracy: {correct}/{len(results)} "
              f"({100*correct/len(results):.1f}%)")
        print(f"  Falsi allarmi (nominali con CP): {nom_with_cp}/{n_nom}")
    print(f"{'='*70}\n")

    csv_path = Path(output_dir) / "test_results_parallel.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"✓ Risultati: {csv_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Necessario su Linux con spawn/fork per evitare ricorsione del Pool
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    run_complete_pipeline_parallel(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        output_dir=OUTPUT_DIR_PAR,
        save_plots=True,
        n_workers=N_WORKERS,
    )