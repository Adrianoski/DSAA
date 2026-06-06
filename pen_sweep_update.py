import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import ruptures as rpt

from data_io import load_dataset, CHANNELS_41_46

from config import (
    CHANNELS, LABELS, CATS,
    WINDOW_DAYS,
    CPD_MIN_SIZE, CPD_JUMP,
    ALPHA_RECALL, RECALL_BIAS, PRECISION_BIAS, BETA_SCORE,
    MAX_GAP_SECONDS, MIN_CLUSTER_IOU,
    MAX_FEATURES,
    PENALTY_SWEEP_STEPS, PENALTY_JUMP_THRESHOLD,
    TRAIN_CSV, TEST_CSV,
)
from penalty_sweep import (
    find_optimal_penalty,
    per_dimension_cost_analysis,
    elbow_contributing_dims,
)
from stft_utils import extract_all_frequency_timeseries_window, extract_features_from_selected_freqs
from metrics import compute_timeseries_metrics, compute_classic_pr, compute_iou_corrected
from pipeline import create_fixed_windows, build_lab_table, _plot_cpd_result

# Percentile della distribuzione nominale usato come soglia gain
NOMINAL_GAIN_PERCENTILE = 95

OUTPUT_DIR = "cpd_output_new"


# ===========================
# Selezione frequenze + CPD (approccio CPD completo)
# ===========================

def select_frequencies_and_cps(
    df_window: pd.DataFrame,
    channels: list,
    max_features: int  = MAX_FEATURES,
    n_steps: int       = PENALTY_SWEEP_STEPS,
    jump_threshold: int = PENALTY_JUMP_THRESHOLD,
    gain_threshold: float = 0.0,
    verbose: bool = True,
) -> tuple:
    """
    Approccio CPD completo su una singola finestra.

    Segue esattamente le istruzioni del professore:
      1. Sweep su X_full → pen* (diagnosi granularità)
      2. Diagnostica per-dim a pen* (tabella k_d, C_d per ogni dim)
      3. PELT(X_full, pen*) → CP globali
      4. gain_d ai CP globali per ogni dim individualmente
      5. Elbow su gain_d → dim selezionate
      6. Mappa dim → frequenze

    Returns
    -------
    selected_freqs : list[float]
    cp_idx         : list[int]    CP globali
    info           : dict
    """
    n_samples = len(df_window)
    if verbose:
        print(f"    [CPD] {n_samples} campioni "
              f"[{df_window.index[0]} → {df_window.index[-1]}]")

    # ── Step 1: STFT → X_full ─────────────────────────────────────────────
    freq_series_dict, _ = extract_all_frequency_timeseries_window(df_window, channels)
    if not freq_series_dict:
        return [], [], {"reason": "STFT vuota"}

    feats, dim_labels = [], []
    for freq_hz in sorted(freq_series_dict.keys()):
        for ch in channels:
            if ch in freq_series_dict[freq_hz]:
                feats.append(freq_series_dict[freq_hz][ch])
                dim_labels.append((freq_hz, ch))

    if not feats:
        return [], [], {"reason": "nessuna feature"}

    X_full = np.column_stack(feats)
    nT, D_full = X_full.shape

    if verbose:
        print(f"    [CPD] X_full: {nT}×{D_full}  "
              f"({len(freq_series_dict)} freq × {len(channels)} canali)")

    # ── Step 1b: sweep su X_full → pen* ───────────────────────────────────
    # "Prova diversi valori di beta" — sweep logspace per trovare la granularità
    try:
        pen_optimal, sweep_info = find_optimal_penalty(
            X_full, n_steps=n_steps, jump_threshold=jump_threshold
        )
    except Exception as e:
        return [], [], {"reason": f"sweep fallito: {e}"}

    k_optimal = sweep_info["k_optimal"]
    pen_bic   = sweep_info["pen_bic"]

    if verbose:
        print(f"    [CPD] Sweep → pen*={pen_optimal:.2f}  "
              f"k_knee={k_optimal}  BIC={pen_bic:.2f}")

    if k_optimal == 0:
        if verbose:
            print("    [CPD] k_knee=0: nessuna struttura a questa granularità")
        return [], [], {"pen_optimal": pen_optimal, "k_optimal": 0,
                        "reason": "k_knee=0"}

    # ── Step 3: PELT su TUTTE le dimensioni → CP globali ─────────────────
    # "Metti il valore di PELT, calcoli il CP con tutte le dimensioni"
    algo_global = rpt.Pelt(model="l2", min_size=CPD_MIN_SIZE, jump=CPD_JUMP).fit(X_full)
    bkps_global = algo_global.predict(pen=pen_optimal)
    cp_idx      = [b - 1 for b in bkps_global[:-1]]

    if verbose:
        print(f"    [CPD] CP globali (X_full): {cp_idx}")

    if not cp_idx:
        return [], [], {"pen_optimal": pen_optimal, "reason": "0 CP globali"}

    # ── Step 4: gain per-dim ai CP GLOBALI ───────────────────────────────
    # "Calcoli costi prima e dopo per i CP GLOBALI
    #  ma il costo lo calcoli individualmente"
    gains = per_dimension_cost_analysis(X_full, bkps_global)

    # ── Gate gain nominale (opzione 3) ───────────────────────────────────
    # Se il gain massimo non supera la soglia calibrata sulle finestre
    # nominali del training, i CP trovati sono spuri → li scartiamo.
    if gain_threshold > 0.0:
        max_gain = float(gains.max())
        if max_gain <= gain_threshold:
            if verbose:
                print(f"    [CPD] GATE nominale: max_gain={max_gain:.4f} "
                      f"≤ soglia={gain_threshold:.4f} → CP soppressi")
            return [], [], {
                "pen_optimal": pen_optimal,
                "k_optimal":   k_optimal,
                "gains":       gains,
                "reason":      f"gain_gate: {max_gain:.4f} ≤ {gain_threshold:.4f}",
            }
        elif verbose:
            print(f"    [CPD] GATE nominale: max_gain={max_gain:.4f} "
                  f"> soglia={gain_threshold:.4f} → CP accettati")

    # ── Step 5: elbow → selezione dimensioni ─────────────────────────────
    # "Thresholding non è abbastanza, guarda l'elbow"
    _, threshold_gain, contributing_mask = elbow_contributing_dims(gains, min_contributing=1)
    selected_idx = np.where(contributing_mask)[0]

    if verbose:
        print(f"    [CPD] Gain ai CP globali per-dim:")
        sorted_d = np.argsort(gains)[::-1]
        for rank, d in enumerate(sorted_d, 1):
            fhz, ch = dim_labels[d]
            marker  = "✓" if d in selected_idx else "✗"
            fhz_str = f"{fhz:.6f}" if fhz >= 0.0001 else "DC"
            print(f"      {marker} {rank:2d}. {fhz_str}Hz/{ch}  "
                  f"gain={gains[d]:.3f}")
        print(f"    [CPD] Elbow: {len(selected_idx)}/{D_full} dim selezionate  "
              f"(gain_min={threshold_gain:.3f})")

    if len(selected_idx) == 0:
        return [], [], {"pen_optimal": pen_optimal, "reason": "elbow: 0 dim",
                        "gains": gains}

    # ── Step 6: mappa dim → frequenze uniche ─────────────────────────────
    selected_freqs = []
    for d in selected_idx[np.argsort(gains[selected_idx])[::-1]]:
        freq_hz, _ = dim_labels[d]
        if freq_hz not in selected_freqs:
            selected_freqs.append(freq_hz)
        if len(selected_freqs) >= max_features:
            break

    if verbose:
        print(f"    [CPD] Frequenze selezionate: "
              f"{[f'{f:.6f}Hz' for f in selected_freqs]}")

    info = {
        "pen_optimal":   pen_optimal,
        "pen_bic":       pen_bic,
        "k_optimal":     k_optimal,
        "bkps_global":   bkps_global,
        "gains":         gains,
        "dim_labels":    dim_labels,
        "selected_idx":  selected_idx,
        "threshold_gain": threshold_gain,
        "sweep_info":    sweep_info,
    }

    return selected_freqs, cp_idx, info


# ===========================
# Soglia gain da finestre nominali del training (opzione 3)
# ===========================

def build_nominal_gain_threshold(
    train_csv: str = TRAIN_CSV,
    percentile: float = NOMINAL_GAIN_PERCENTILE,
    verbose: bool = True,
) -> float:
    """
    Calibra una soglia data-driven sul gain massimo per-dim osservato
    nelle finestre NOMINALI del training set.

    Procedura:
    - Carica il training set con labels
    - Crea finestre fisse (stessa WINDOW_DAYS del test)
    - Filtra le finestre senza alcuna anomalia annotata (nominali pure)
    - Per ogni finestra nominale esegue: STFT → sweep → PELT → gain per-dim
    - Raccoglie max(gain) per ogni finestra in cui PELT trova almeno 1 CP
    - Restituisce il percentile richiesto di quella distribuzione

    Questo valore rappresenta il gain massimo atteso in condizioni nominali.
    Un test window con max(gain) inferiore a questa soglia produce CP spuri.

    Returns
    -------
    float: soglia gain. Se non ci sono dati nominali sufficienti, ritorna 0.0
           (nessun filtraggio).
    """
    print(f"\n{'─'*70}")
    print(f"Calibrazione soglia gain nominale (percentile={percentile})")
    print(f"  Carico training: {train_csv}")

    df_train, _, _ = load_dataset(
        csv_path=train_csv, use_channels=CHANNELS_41_46,
        include_labels=True, interpolate=True, fill_strategy="both",
    )
    df_train = df_train.sort_index()
    present_channels = [c for c in CHANNELS if c in df_train.columns]
    present_labels   = [l for l in LABELS   if l in df_train.columns]
    for c in present_channels:
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
    for l in present_labels:
        df_train[l] = pd.to_numeric(df_train[l], errors="coerce").astype("Int64")

    # Tabella anomalie nel training (per filtrare le finestre nominali)
    lab_train = build_lab_table(df_train, present_labels)

    windows_train = create_fixed_windows(df_train, window_days=WINDOW_DAYS)
    print(f"  {len(windows_train)} finestre training totali")

    max_gains = []
    n_nominal = 0
    n_with_cp = 0

    for w_id, (win_start, win_end) in enumerate(windows_train, 1):
        # Salta finestre con anomalie
        events = (
            lab_train[
                (lab_train["StartTime"] <= win_end) &
                (lab_train["EndTime"]   >= win_start)
            ]
            if not lab_train.empty else pd.DataFrame()
        )
        if not events.empty:
            continue
        n_nominal += 1

        W = df_train.loc[win_start:win_end, present_channels]
        if W.dropna(how="all").empty:
            continue

        # STFT → X_full
        freq_series_dict, _ = extract_all_frequency_timeseries_window(W, present_channels)
        if not freq_series_dict:
            continue

        feats, dim_labels = [], []
        for freq_hz in sorted(freq_series_dict.keys()):
            for ch in present_channels:
                if ch in freq_series_dict[freq_hz]:
                    feats.append(freq_series_dict[freq_hz][ch])
                    dim_labels.append((freq_hz, ch))
        if not feats:
            continue

        X_full = np.column_stack(feats)

        try:
            pen_opt, sweep_info = find_optimal_penalty(
                X_full, n_steps=PENALTY_SWEEP_STEPS,
                jump_threshold=PENALTY_JUMP_THRESHOLD,
            )
        except Exception:
            continue

        if sweep_info["k_optimal"] == 0:
            continue

        # CP globali
        try:
            algo = rpt.Pelt(model="l2", min_size=CPD_MIN_SIZE, jump=CPD_JUMP).fit(X_full)
            bkps = algo.predict(pen=pen_opt)
        except Exception:
            continue

        if len(bkps) <= 1:
            continue

        gains = per_dimension_cost_analysis(X_full, bkps)
        max_g = float(gains.max())
        if max_g > 0:
            max_gains.append(max_g)
            n_with_cp += 1

        if verbose and w_id % 20 == 0:
            print(f"  [calibrazione] {n_nominal} nom / {w_id} tot — "
                  f"{n_with_cp} con CP — max_gain corrente: "
                  f"{np.percentile(max_gains, percentile):.3f}" if max_gains else "n/a")

    if not max_gains:
        print("  ⚠ Nessuna finestra nominale con CP trovata — soglia=0 (nessun filtraggio)")
        return 0.0

    threshold = float(np.percentile(max_gains, percentile))
    print(f"  ✓ Finestre nominali: {n_nominal} | con CP spurio: {n_with_cp}")
    print(f"  Distribuzione max_gain nominale → "
          f"μ={np.mean(max_gains):.3f}  σ={np.std(max_gains):.3f}  "
          f"max={np.max(max_gains):.3f}")
    print(f"  Soglia gain ({percentile}° percentile): {threshold:.4f}")
    print(f"{'─'*70}\n")
    return threshold


# ===========================
# Singola finestra
# ===========================

def _cpd_single_window(
    df, w_id, win_start, win_end,
    present_channels, present_labels,
    events_in_win, has_gt,
    focus_ch, output_dir, save_plots,
    gain_threshold: float = 0.0,
):
    W = df.loc[win_start:win_end, present_channels]
    if W.dropna(how="all").empty:
        return None

    selected_freqs, cp_idx, info = select_frequencies_and_cps(
        W, present_channels, gain_threshold=gain_threshold, verbose=True
    )

    if not selected_freqs:
        reason = info.get("reason", "")
        print(f"  [win {w_id:03d}] Skip → {reason}")
        return None

    pen_optimal = info["pen_optimal"]
    k_optimal   = info["k_optimal"]
    n_freqs     = len(selected_freqs)

    print(f"  [win {w_id:03d}] pen*={pen_optimal:.2f}  k_knee={k_optimal}  "
          f"CP={len(cp_idx)}  | {'GT:ANOMALY' if has_gt else 'GT:NOMINAL'}")

    X, t_stamps, feat_labels = extract_features_from_selected_freqs(
        df, win_start, win_end, selected_freqs, present_channels
    )
    if X is None:
        return None

    nT, D = X.shape
    bkps  = info["bkps_global"]

    # ── Maschera GT ───────────────────────────────────────────────────────
    lab_mask = np.zeros(nT, dtype=bool)
    if has_gt:
        for _, r in events_in_win.iterrows():
            s_lab, e_lab = r["StartTime"], r["EndTime"]
            lab_mask |= (
                (t_stamps >= np.datetime64(s_lab)) &
                (t_stamps <= np.datetime64(e_lab))
            )

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

        prec_classic, rec_classic = (
            compute_classic_pr(true_bkps, bkps, true_cps_frames, cp_idx)
            if true_cps_frames else (0.0, 0.0)
        )
        f1_classic = (
            2 * prec_classic * rec_classic / (prec_classic + rec_classic)
            if (prec_classic + rec_classic) > 0 else 0.0
        )
        metrics_T = compute_timeseries_metrics(
            cp_idx, lab_mask.astype(int), nT,
            alpha=ALPHA_RECALL, recall_bias=RECALL_BIAS,
            precision_bias=PRECISION_BIAS, beta=BETA_SCORE,
        )
        iou_total, selected_clusters, _, detailed_info = compute_iou_corrected(
            cp_idx, lab_mask.astype(int), nT, t_stamps,
            max_gap_seconds=MAX_GAP_SECONDS,
            min_cluster_iou=MIN_CLUSTER_IOU, verbose=True,
        )
        print(f"  [win {w_id:03d}] Classic:     P={prec_classic:.3f} "
              f"R={rec_classic:.3f} F1={f1_classic:.3f}")
        print(f"  [win {w_id:03d}] Range-based: "
              f"P_T={metrics_T['precision_T']:.3f} "
              f"R_T={metrics_T['recall_T']:.3f} "
              f"F_β={metrics_T['f_beta']:.3f}")
        print(f"  [win {w_id:03d}] IoU: {iou_total:.3f}")
    else:
        prec_classic = rec_classic = f1_classic = 0.0
        metrics_T    = {"precision_T": 0.0, "recall_T": 0.0, "f_beta": 0.0,
                        "n_real_ranges": 0, "n_pred_ranges": 0}
        iou_total        = 0.0
        selected_clusters = []
        detailed_info    = {
            "n_anomalies": 0, "n_segments": 0, "n_clusters": 0,
            "n_selected": 0, "gt_anomalies": [],
            "cluster_matches": [], "iou_per_anomaly": [],
        }
        print(f"  [win {w_id:03d}] Nominale — metriche N/A")

    print()

    if save_plots:
        _plot_cpd_result(
            w_id, W, X, t_stamps, cp_idx, selected_clusters,
            events_in_win, feat_labels, focus_ch,
            prec_classic, rec_classic, f1_classic,
            metrics_T["precision_T"], metrics_T["recall_T"],
            metrics_T["f_beta"], iou_total,
            pen_optimal, D, detailed_info, output_dir, n_freqs,
            has_gt=has_gt, sweep_info=info["sweep_info"],
        )

    return {
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


# ===========================
# Pipeline completa
# ===========================

def run_complete_pipeline(
    test_csv: str   = TEST_CSV,
    train_csv: str  = TRAIN_CSV,
    output_dir: str = OUTPUT_DIR,
    save_plots: bool = True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "plots").mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# PIPELINE CPD")
    print(f"# Sweep → diagnostica per-dim → CP globali → gain per-dim → elbow")
    print(f"{'#'*70}\n")

    df_test, _, _ = load_dataset(
        csv_path=test_csv, use_channels=CHANNELS_41_46,
        include_labels=True, interpolate=True, fill_strategy="both",
    )
    print(f"  ✓ {len(df_test)} samples caricati\n")

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
    print(f"TEST CPD — {WINDOW_DAYS}gg/finestra  |  {n_windows} finestre totali")
    print(f"  Canale focus: {focus_ch}  |  GT anomalie: {len(lab_sel)}")
    print(f"{'='*70}\n")

    # Calibra soglia gain su finestre nominali del training (opzione 3)
    gain_threshold = build_nominal_gain_threshold(
        train_csv=train_csv,
        percentile=NOMINAL_GAIN_PERCENTILE,
        verbose=True,
    )

    results = []
    g_pc, g_rc, g_f1c = [], [], []
    g_pT, g_rT, g_fbT, g_iou = [], [], [], []

    for w_id, (win_start, win_end) in enumerate(windows, 1):
        events_in_win = (
            lab_sel[(lab_sel["StartTime"] <= win_end) & (lab_sel["EndTime"] >= win_start)]
            if not lab_sel.empty else pd.DataFrame()
        )
        has_gt = not events_in_win.empty

        print(f"\n{'─'*70}")
        print(f"FINESTRA {w_id:03d}/{n_windows} "
              f"[{'GT:ANOMALY' if has_gt else 'GT:NOMINAL'}]: "
              f"{win_start} → {win_end}")

        try:
            r = _cpd_single_window(
                df_test, w_id, win_start, win_end,
                present_channels, present_labels,
                events_in_win, has_gt,
                focus_ch, str(Path(output_dir) / "plots"), save_plots,
                gain_threshold=gain_threshold,
            )
            if r is not None:
                results.append(r)
                if has_gt:
                    g_pc.append(r["precision_classic"])
                    g_rc.append(r["recall_classic"])
                    g_f1c.append(r["f1_classic"])
                    g_pT.append(r["precision_T"])
                    g_rT.append(r["recall_T"])
                    g_fbT.append(r["f_beta"])
                    g_iou.append(r["iou_total"])
        except Exception as e:
            print(f"  [ERRORE win {w_id}]: {e}")
            traceback.print_exc()

    n_gt  = sum(1 for r in results if r["has_gt_anomalies"])
    n_nom = len(results) - n_gt

    print(f"\n{'='*70}")
    print(f"✓ CPD COMPLETATO: {len(results)}/{n_windows} finestre")
    print(f"  Anomalie GT: {n_gt}  |  Nominali: {n_nom}")

    if g_f1c:
        print(f"\n{'CLASSIC':-^70}")
        print(f"  P={np.mean(g_pc):.3f}  R={np.mean(g_rc):.3f}  F1={np.mean(g_f1c):.3f}")
        print(f"\n{'RANGE-BASED':-^70}")
        print(f"  P_T={np.mean(g_pT):.3f}  R_T={np.mean(g_rT):.3f}  "
              f"F-β={np.mean(g_fbT):.3f}  IoU={np.mean(g_iou):.3f}")

    if results:
        correct = sum(1 for r in results if r["detection"] == "CORRECT")
        print(f"\n  Detection: {correct}/{len(results)} "
              f"({100*correct/len(results):.1f}%)")
    print(f"{'='*70}\n")

    csv_path = Path(output_dir) / "test_results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"✓ Risultati: {csv_path}")

    return results


if __name__ == "__main__":
    run_complete_pipeline(
        test_csv=TEST_CSV,
        output_dir=OUTPUT_DIR,
        save_plots=True,
    )
