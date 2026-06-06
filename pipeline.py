import sys
from pathlib import Path

# Permette di importare data_io dalla cartella Jan
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

from data_io import load_dataset, CHANNELS_41_46

from config import (
    CHANNELS, LABELS, CATS,
    WINDOW_DAYS, DAYS_PAD,
    CPD_MIN_SIZE, CPD_JUMP,
    PR_MARGIN_FR, ALPHA_RECALL, RECALL_BIAS, PRECISION_BIAS, BETA_SCORE,
    MAX_GAP_SECONDS, MIN_CLUSTER_IOU,
    MAX_FEATURES,
    PENALTY_SWEEP_STEPS, PENALTY_JUMP_THRESHOLD,
    PER_DIM_MIN_CONTRIBUTING,
    TEST_CSV, OUTPUT_DIR,
)
from freq_selection import select_frequencies_for_window
from stft_utils import extract_features_from_selected_freqs
from metrics import (
    compute_timeseries_metrics, compute_classic_pr, compute_iou_corrected,
)


# ===========================
# Windowing utilities
# ===========================

def create_fixed_windows(df: pd.DataFrame, window_days: float = WINDOW_DAYS):
    """Finestre fisse di window_days giorni su tutto il dataset."""
    df = df.sort_index()
    delta = pd.Timedelta(days=window_days)
    windows = []
    cur = df.index[0]
    end = df.index[-1]
    while cur < end:
        win_end = min(cur + delta, end)
        windows.append((cur, win_end))
        cur += delta
    return windows


def build_lab_table(df: pd.DataFrame, present_labels: list) -> pd.DataFrame:
    cat_to_val = {"Nominal": 0, "Rare Event": 1, "Anomaly": 2, "Communication gap": 3}
    val_to_cat = {v: k for k, v in cat_to_val.items()}
    target_vals = {cat_to_val[c] for c in CATS}
    rows = []
    for lab_col in present_labels:
        ch_name = lab_col.replace("is_anomaly_", "")
        mask = df[lab_col].isin(target_vals)
        if not mask.any():
            continue
        groups = (mask != mask.shift()).cumsum()
        for _, seg in mask[mask].groupby(groups[mask]):
            v = int(df.loc[seg.index[0], lab_col])
            rows.append({
                "StartTime": seg.index[0],
                "EndTime":   seg.index[-1],
                "Category":  val_to_cat[v],
                "Channel":   ch_name,
            })
    return pd.DataFrame(rows).sort_values("StartTime") if rows else pd.DataFrame()


# ===========================
# CPD su singola finestra (adaptive + penalty sweep)
# ===========================

def _cpd_single_window_adaptive(
    df, w_id, win_start, win_end,
    present_channels, present_labels,
    events_in_win, has_gt,
    focus_ch, output_dir, save_plots,
):
    """
    CPD su singola finestra con approccio del professore:
      1. Sweep su X_full (tutte freq × canali) → penalty ottimale + CP globali
      2. Gain per-dimensione ai CP globali → elbow → selezione frequenze
      3. X ridotto con freq selezionate (solo per metriche/plot)
      4. I CP sono quelli trovati nel passo 1, non un nuovo PELT
    """
    W = df.loc[win_start:win_end, present_channels]
    if W.dropna(how="all").empty:
        return None

    # ── Steps 1-6: sweep globale → CP → gain per-dim → elbow → freq ──────
    selected_freqs_list, sweep_info = select_frequencies_for_window(
        W, present_channels,
        max_features=MAX_FEATURES,
        verbose=True,
    )

    if not selected_freqs_list or sweep_info is None:
        print(f"  [win {w_id:03d}] Nessuna struttura rilevante → skip")
        return None

    n_freqs     = len(selected_freqs_list)
    cp_idx      = sweep_info["cp_global"]
    bkps        = sweep_info["bkps_global"]
    pen_optimal = float(sweep_info["pen_values"][sweep_info["knee_idx"]])
    pen_bic     = sweep_info["pen_bic"]
    k_natural   = sweep_info["k_optimal"]
    per_dim_gains    = sweep_info["per_dim_gains"]
    contributing_mask = sweep_info["contributing_mask"]
    dim_labels   = sweep_info["dim_labels"]

    n_contrib   = int(contributing_mask.sum())
    contrib_labels = [
        f"{fhz:.6f}Hz/{ch}"
        for (fhz, ch), flag in zip(dim_labels, contributing_mask) if flag
    ]

    print(f"  [win {w_id:03d}] Frequenze: " +
          ", ".join(f"{f:.6f}Hz" for f in selected_freqs_list))
    print(f"  [win {w_id:03d}] T D={len(dim_labels)} | pen*={pen_optimal:.2f} "
          f"BIC={pen_bic:.2f} k_knee={k_natural} | {len(cp_idx)} CP globali")
    print(f"  [win {w_id:03d}] Dims sopra elbow: {n_contrib} → {contrib_labels[:4]}")

    # ── Step 7: X con solo le frequenze selezionate (metriche / plot) ─────
    X, t_stamps, feat_labels = extract_features_from_selected_freqs(
        df, win_start, win_end, selected_freqs_list, present_channels
    )
    if X is None:
        return None

    nT, D = X.shape

    # ── Maschera GT ───────────────────────────────────────────────────────
    lab_mask = np.zeros(nT, dtype=bool)
    if has_gt:
        for _, r in events_in_win.iterrows():
            s_lab, e_lab = r["StartTime"], r["EndTime"]
            lab_mask |= (
                (t_stamps >= np.datetime64(s_lab)) &
                (t_stamps <= np.datetime64(e_lab))
            )

    print(f"  [win {w_id:03d}] {'GT:ANOMALY' if has_gt else 'GT:NOMINAL'} "
          f"| {len(cp_idx)} CP rilevati")

    # ── Metriche ──────────────────────────────────────────────────────────
    if has_gt:
        true_cps_frames = []
        for _, r in events_in_win.iterrows():
            s_lab, e_lab = r["StartTime"], r["EndTime"]
            s_idx = int(np.clip(np.searchsorted(t_stamps, np.datetime64(s_lab)), 0, nT - 1))
            e_idx = int(np.clip(np.searchsorted(t_stamps, np.datetime64(e_lab)), 0, nT - 1))
            true_cps_frames.extend([s_idx, e_idx])
        true_cps_frames = sorted(set(i for i in true_cps_frames if 0 <= i < nT - 1))
        true_bkps = sorted(set([min(i + 1, nT - 1) for i in true_cps_frames] + [nT]))

        if true_cps_frames:
            prec_classic, rec_classic = compute_classic_pr(
                true_bkps, bkps, true_cps_frames, cp_idx
            )
        else:
            prec_classic = rec_classic = 0.0

        f1_classic = (
            2 * prec_classic * rec_classic / (prec_classic + rec_classic)
            if (prec_classic + rec_classic) > 0 else 0.0
        )

        metrics_T = compute_timeseries_metrics(
            cp_idx, lab_mask.astype(int), nT,
            alpha=ALPHA_RECALL,
            recall_bias=RECALL_BIAS,
            precision_bias=PRECISION_BIAS,
            beta=BETA_SCORE,
        )

        iou_total, selected_clusters, pred_mask, detailed_info = compute_iou_corrected(
            cp_idx, lab_mask.astype(int), nT, t_stamps,
            max_gap_seconds=MAX_GAP_SECONDS,
            min_cluster_iou=MIN_CLUSTER_IOU,
            verbose=True,
        )

        print(f"  [win {w_id:03d}] Classic: P={prec_classic:.3f} R={rec_classic:.3f} "
              f"F1={f1_classic:.3f}")
        print(f"  [win {w_id:03d}] Range-based: P_T={metrics_T['precision_T']:.3f} "
              f"R_T={metrics_T['recall_T']:.3f} F_β={metrics_T['f_beta']:.3f}")
        print(f"  [win {w_id:03d}] IoU corretta: {iou_total:.3f}")
    else:
        prec_classic = rec_classic = f1_classic = 0.0
        metrics_T = {"precision_T": 0.0, "recall_T": 0.0, "f_beta": 0.0,
                     "n_real_ranges": 0, "n_pred_ranges": 0}
        iou_total = 0.0
        selected_clusters = []
        detailed_info = {
            "n_anomalies": 0, "n_segments": 0, "n_clusters": 0,
            "n_selected": 0, "gt_anomalies": [],
            "cluster_matches": [], "iou_per_anomaly": [],
        }
        print(f"  [win {w_id:03d}] Finestra nominale — metriche N/A")

    n_anomalies = detailed_info["n_anomalies"]
    n_clusters  = detailed_info.get("n_clusters", 0)
    n_selected  = detailed_info.get("n_selected", 0)
    print()

    if save_plots:
        _plot_cpd_result(
            w_id, W, X, t_stamps, cp_idx, selected_clusters,
            events_in_win, feat_labels, focus_ch,
            prec_classic, rec_classic, f1_classic,
            metrics_T["precision_T"], metrics_T["recall_T"],
            metrics_T["f_beta"], iou_total,
            pen_optimal, D, detailed_info, output_dir, n_freqs,
            has_gt=has_gt, sweep_info=sweep_info,
        )

    gt_label  = "ANOMALY" if has_gt else "NOMINAL"
    detected  = len(cp_idx) > 0
    detection = "CORRECT" if (has_gt == detected) else "WRONG"

    return {
        "window":              w_id,
        "gt_label":            gt_label,
        "detection":           detection,
        "win_start":           str(win_start),
        "win_end":             str(win_end),
        "has_gt_anomalies":    has_gt,
        "n_selected_freqs":    n_freqs,
        "selected_freqs":      selected_freqs_list,
        "n_features":          D,
        "pen_optimal":         pen_optimal,
        "k_knee":              k_natural,
        "n_changepoints":      len(cp_idx),
        "n_anomalies":         n_anomalies,
        "n_clusters":          n_clusters,
        "n_selected_clusters": n_selected,
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


# ===========================
# Test su tutto il test set
# ===========================

def test_cpd_adaptive_realtime(df_test, output_dir=OUTPUT_DIR, save_plots=True):
    """
    CPD adaptive real-time su finestre fisse di WINDOW_DAYS giorni
    su tutto il test set. Penalty sweep ovunque.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    cov = {ch: 0 for ch in present_channels}
    for ch in present_channels:
        lab_col = f"is_anomaly_{ch}"
        if lab_col in df_test.columns:
            cov[ch] = int(df_test[lab_col].isin(target_vals).sum())
    focus_ch = max(cov, key=cov.get)

    windows  = create_fixed_windows(df_test, window_days=WINDOW_DAYS)
    n_windows = len(windows)

    print(f"\n{'='*70}")
    print(f"TEST ADAPTIVE REAL-TIME (PENALTY SWEEP) — finestre {WINDOW_DAYS}gg")
    print(f"{'='*70}")
    print(f"Canale focus:        {focus_ch}")
    print(f"Finestre totali:     {n_windows}")
    print(f"Periodo test:        {df_test.index[0]} → {df_test.index[-1]}")
    print(f"Anomalie GT totali:  {len(lab_sel)}")
    print(f"Sweep steps:         {PENALTY_SWEEP_STEPS}")
    print(f"Jump threshold:      {PENALTY_JUMP_THRESHOLD}")
    print()

    results = []
    glob_prec_classic, glob_rec_classic, glob_f1_classic = [], [], []
    glob_prec_T, glob_rec_T, glob_f1_T = [], [], []
    glob_iou = []

    for w_id, (win_start, win_end) in enumerate(windows, 1):
        if lab_sel.empty:
            events_in_win = pd.DataFrame()
        else:
            events_in_win = lab_sel[
                (lab_sel["StartTime"] <= win_end) &
                (lab_sel["EndTime"]   >= win_start)
            ]
        has_gt = not events_in_win.empty

        gt_tag = "GT:ANOMALY" if has_gt else "GT:NOMINAL"
        print(f"\n{'─'*70}")
        print(f"FINESTRA {w_id:03d}/{n_windows} [{gt_tag}]: {win_start} → {win_end}")

        try:
            result = _cpd_single_window_adaptive(
                df_test, w_id, win_start, win_end,
                present_channels, present_labels,
                events_in_win, has_gt,
                focus_ch, str(plots_dir), save_plots,
            )
            if result is not None:
                results.append(result)
                if has_gt:
                    glob_prec_classic.append(result["precision_classic"])
                    glob_rec_classic.append(result["recall_classic"])
                    glob_f1_classic.append(result["f1_classic"])
                    glob_prec_T.append(result["precision_T"])
                    glob_rec_T.append(result["recall_T"])
                    glob_f1_T.append(result["f_beta"])
                    glob_iou.append(result["iou_total"])
        except Exception as e:
            import traceback
            print(f"  [ERRORE win {w_id}]: {e}")
            traceback.print_exc()
            continue

    # ── Summary ───────────────────────────────────────────────────────────
    n_gt_wins  = sum(1 for r in results if r["has_gt_anomalies"])
    n_nom_wins = sum(1 for r in results if not r["has_gt_anomalies"])

    print(f"\n{'='*70}")
    print(f"✓ TEST COMPLETATO: {len(results)}/{n_windows} finestre processate")
    print(f"  Finestre con anomalie GT: {n_gt_wins}")
    print(f"  Finestre nominali:        {n_nom_wins}")

    if glob_f1_classic:
        print(f"\n{'METRICHE CLASSIC — solo finestre con GT':-^70}")
        print(f"  Precision:  {np.mean(glob_prec_classic):.3f}")
        print(f"  Recall:     {np.mean(glob_rec_classic):.3f}")
        print(f"  F1-Score:   {np.mean(glob_f1_classic):.3f}")

        print(f"\n{'METRICHE RANGE-BASED — solo finestre con GT':-^70}")
        print(f"  Precision_T: {np.mean(glob_prec_T):.3f}")
        print(f"  Recall_T:    {np.mean(glob_rec_T):.3f}")
        print(f"  F-beta_T:    {np.mean(glob_f1_T):.3f}")
        print(f"  IoU Total:   {np.mean(glob_iou):.3f}")

    if results:
        avg_freqs = np.mean([r["n_selected_freqs"] for r in results])
        avg_pen   = np.mean([r["pen_optimal"] for r in results])
        avg_k     = np.mean([r["k_knee"] for r in results])
        print(f"\n  Avg freq per finestra:   {avg_freqs:.1f}")
        print(f"  Avg pen_optimal:         {avg_pen:.2f}")
        print(f"  Avg k al ginocchio:      {avg_k:.1f}")

    print(f"{'='*70}\n")

    import pandas as _pd
    results_df = _pd.DataFrame(results)
    results_csv = Path(output_dir) / "test_results_penalty_sweep.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Risultati: {results_csv}")
    print(f"✓ Plot:      {plots_dir}\n")

    return results


# ===========================
# Plot
# ===========================

def _plot_cpd_result(
    w_id, W, X, t_stamps, cp_idx, selected_clusters,
    events_in_win, feat_labels, focus_ch,
    prec_classic, rec_classic, f1_classic,
    prec_T, rec_T, f_beta_T, iou,
    pen, D, detailed_info, output_dir, n_freqs,
    has_gt: bool = True,
    sweep_info: dict = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(20, 9), sharex=True)

    s_focus = W[focus_ch].astype(float)
    ax0 = axes[0]
    ax0.plot(s_focus.index, s_focus.values, lw=0.9, color="black",
             label=focus_ch, zorder=1)

    gt_anomalies = detailed_info.get("gt_anomalies", [])
    colors = ["tab:green", "tab:blue", "tab:cyan", "tab:olive"]

    for anom in gt_anomalies:
        ts = pd.Timestamp(anom["start_time"])
        te = pd.Timestamp(anom["end_time"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        te = te.tz_localize("UTC") if te.tzinfo is None else te
        mask_anom = (s_focus.index >= ts) & (s_focus.index <= te)
        if mask_anom.any():
            ax0.plot(s_focus.index[mask_anom], s_focus.values[mask_anom],
                     lw=2.0, color="limegreen", zorder=3,
                     label=f"Anomaly {anom['id']} GT")

    for idx, (seg_start, seg_end) in enumerate(selected_clusters):
        t_start = pd.to_datetime(t_stamps[seg_start])
        t_end   = pd.to_datetime(t_stamps[seg_end])
        ax0.axvspan(t_start, t_end, color="#FF6B35", alpha=0.5,
                    label="Selected clusters" if idx == 0 else "", zorder=3)

    rejected_added = False
    for match in detailed_info.get("cluster_matches", []):
        cluster = match["cluster"]
        if match["iou"] < MIN_CLUSTER_IOU or match["matched_anomaly"] is None:
            ax0.axvspan(cluster["start_time"], cluster["end_time"],
                        color="gray", alpha=0.15,
                        label="Rejected clusters" if not rejected_added else "", zorder=2)
            rejected_added = True

    ax0.set_ylabel(focus_ch, fontsize=11)
    ax0.legend(loc="upper right", fontsize=8, ncols=2)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    n_to_plot = min(D, 30)
    for d in range(n_to_plot):
        label = feat_labels[d] if d < len(feat_labels) else f"Feat{d}"
        if "DC" in label:
            ax1.plot(pd.to_datetime(t_stamps), X[:, d], lw=1.5,
                     label=label, alpha=0.9, color="purple")
        else:
            ax1.plot(pd.to_datetime(t_stamps), X[:, d], lw=0.8,
                     label=label if d < 12 else "", alpha=0.7)

    for j, cp in enumerate(cp_idx):
        ax1.axvline(pd.to_datetime(t_stamps[cp]), color="tab:red", ls="--", lw=1.5,
                    label="CP" if j == 0 else "", alpha=0.8, zorder=4)

    for anom in gt_anomalies:
        color = colors[anom["id"] % len(colors)]
        ax1.axvspan(anom["start_time"], anom["end_time"],
                    color=color, alpha=0.12, zorder=1)

    for seg_start, seg_end in selected_clusters:
        ax1.axvspan(pd.to_datetime(t_stamps[seg_start]),
                    pd.to_datetime(t_stamps[seg_end]),
                    color="#FF6B35", alpha=0.3, zorder=2)

    ax1.set_ylabel("Feature z-score", fontsize=11)
    ax1.set_xlabel("Tempo UTC", fontsize=11)
    ax1.legend(loc="upper right", ncols=4, fontsize=6)
    ax1.grid(True, alpha=0.3)

    n_anomalies = detailed_info["n_anomalies"]
    n_clusters  = detailed_info.get("n_clusters", 0)
    n_selected  = detailed_info.get("n_selected", 0)

    k_knee    = sweep_info["k_optimal"] if sweep_info else "?"
    gt_label = "GT: ANOMALY" if has_gt else "GT: NOMINAL"
    gt_color = "darkred" if has_gt else "darkgreen"

    iou_str = ", ".join(
        [f"A{i}={v:.2f}" for i, v in detailed_info.get("iou_per_anomaly", [])]
    )

    if has_gt:
        metrics_line = (
            f"Classic: P={prec_classic:.3f} R={rec_classic:.3f} F1={f1_classic:.3f} | "
            f"Range: P_T={prec_T:.3f} R_T={rec_T:.3f} F_β={f_beta_T:.3f} | "
            f"IoU={iou:.3f}"
        )
        anomaly_line = (
            f"{len(cp_idx)} CP → {n_clusters} clusters ({n_selected} selected) | "
            f"{n_anomalies} anomalies GT: [{iou_str}]"
        )
    else:
        metrics_line = "Metriche: N/A (nessuna anomalia GT)"
        anomaly_line = f"{len(cp_idx)} CP rilevati → {n_clusters} clusters"

    title = (
        f"CPD - Window {w_id:03d} | [{gt_label}] | "
        f"{n_freqs} FREQ PENALTY SWEEP | PELT L2 (pen={pen:.2f}, k_knee={k_knee}, D={D})\n"
        f"{anomaly_line}\n{metrics_line}"
    )
    axes[0].set_title(title, fontsize=9, fontweight="bold", color=gt_color)
    plt.tight_layout()

    gt_prefix = "anomaly" if has_gt else "nominal"
    filepath = Path(output_dir) / f"cpd_{gt_prefix}_win{w_id:03d}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================
# Pipeline completa
# ===========================

def run_complete_pipeline(test_csv=TEST_CSV, output_dir=OUTPUT_DIR, save_plots=True):
    """Pipeline adaptive real-time con penalty sweep su tutto il test set."""

    print(f"\n{'#'*70}")
    print(f"# PIPELINE PENALTY SWEEP — adaptive real-time")
    print(f"# Penalty sweep integrato: selezione frequenze + CPD finale")
    print(f"{'#'*70}\n")

    print(f"FASE 1: Caricamento TEST")
    df_test, _, _ = load_dataset(
        csv_path=test_csv,
        use_channels=CHANNELS_41_46,
        include_labels=True,
        interpolate=True,
        fill_strategy="both",
    )
    print(f"  ✓ {len(df_test)} samples caricati\n")

    results = test_cpd_adaptive_realtime(df_test, output_dir, save_plots)

    print(f"\n{'#'*70}")
    print(f"# PIPELINE COMPLETATA — {len(results)} finestre processate")
    print(f"{'#'*70}\n")

    return results
