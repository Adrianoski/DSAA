"""
pen_sweep_paper.py
========================
Mode:
  - 'eval'   (default): legge il test_results.csv esistente e calcola
                        le metriche paper-aligned. Veloce.
  - 'rerun': standardizza i CSV, esegue la pipeline CPD, poi 'eval'.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Riusiamo le costanti / utility già presenti nel progetto
sys.path.insert(0, str(Path(__file__).parent))
from data_io import load_dataset, CHANNELS_41_46  # noqa: E402

from config import (  # noqa: E402
    CHANNELS, LABELS, TRAIN_CSV, TEST_CSV, ESA_ADB_DATA,
)

# Categorie label ESA-ADB
LBL_NOMINAL          = 0
LBL_RARE_NOMINAL     = 1
LBL_ANOMALY          = 2
LBL_COMM_GAP         = 3

DEFAULT_RESULTS_CSV  = "cpd_output_new2/test_results.csv"
PAPER_TRAIN_CSV      = "84_months.train.standardized.csv"
PAPER_TEST_CSV       = "84_months.test.standardized.csv"
PAPER_OUTPUT_DIR     = "cpd_output_paper"   # cartella distinta dal run base

# Annotazioni ufficiali ESA-ADB (fonte di verità del paper)
ESAADB_LABELS_CSV       = os.path.join(ESA_ADB_DATA, "ESA-Mission1/labels.csv")
ESAADB_ANOMTYPES_CSV    = os.path.join(ESA_ADB_DATA, "ESA-Mission1/anomaly_types.csv")
TEST_PERIOD_START       = pd.Timestamp("2007-01-01", tz="UTC")
TEST_PERIOD_END         = pd.Timestamp("2014-01-01", tz="UTC")
TARGET_CATEGORY         = "Anomaly"      # paper: 29 nel test lightweight
RARE_CATEGORY           = "Rare Event"   # paper: 36 nel test lightweight


# ─────────────────────────────────────────────────────────────────────────────
# Standardizzazione (paper: zero-mean / unit-std su punti nominali del training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_nominal_stats(df: pd.DataFrame, channels: List[str],
                          labels: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Calcola mean/std PER CANALE usando solo i samples in cui TUTTI i label
    presenti valgono LBL_NOMINAL (cioè 0). Le righe con label==2 (Anomaly),
    label==1 (Rare nominal) o label==3 (Comm gap) sono escluse.
    """
    if labels:
        nominal_mask = np.ones(len(df), dtype=bool)
        for lab in labels:
            col = pd.to_numeric(df[lab], errors="coerce").fillna(LBL_NOMINAL)
            nominal_mask &= (col == LBL_NOMINAL).values
        df_nom = df.loc[nominal_mask, channels]
    else:
        df_nom = df[channels]

    mean = df_nom.mean()
    std  = df_nom.std().replace(0.0, 1.0)
    return mean, std


def standardize_in_place(df: pd.DataFrame, channels: List[str],
                         mean: pd.Series, std: pd.Series) -> None:
    for c in channels:
        df[c] = (df[c] - mean[c]) / std[c]


def build_standardized_csvs(train_csv: str, test_csv: str,
                            out_train: str, out_test: str) -> None:
    """
    Salva train+test standardizzati con mean/std calcolati su nominal training.
    """
    print(f"[standardize] Carico training: {train_csv}")
    df_train, _, labs = load_dataset(
        csv_path=train_csv, use_channels=CHANNELS_41_46,
        include_labels=True, interpolate=True, fill_strategy="both",
    )
    channels = [c for c in CHANNELS if c in df_train.columns]

    print(f"[standardize] Calcolo mean/std su punti nominal del training "
          f"({len(channels)} canali, {len(labs)} labels)")
    mean, std = compute_nominal_stats(df_train, channels, labs)
    for c in channels:
        print(f"    {c}: μ={mean[c]:.6f}  σ={std[c]:.6f}")

    standardize_in_place(df_train, channels, mean, std)
    df_train.to_csv(out_train, index_label="timestamp")
    print(f"[standardize] ✓ Train salvato: {out_train}")

    print(f"[standardize] Carico test: {test_csv}")
    df_test, _, _ = load_dataset(
        csv_path=test_csv, use_channels=CHANNELS_41_46,
        include_labels=True, interpolate=True, fill_strategy="both",
    )
    standardize_in_place(df_test, channels, mean, std)
    df_test.to_csv(out_test, index_label="timestamp")
    print(f"[standardize] ✓ Test salvato: {out_test}")


# ─────────────────────────────────────────────────────────────────────────────
# Estrazione eventi annotati (paper: 29 anomalie nel test set lightweight)
# ─────────────────────────────────────────────────────────────────────────────

def load_paper_events(category: str = TARGET_CATEGORY,
                      channels: List[str] = None,
                      t_start: pd.Timestamp = TEST_PERIOD_START,
                      t_end:   pd.Timestamp = TEST_PERIOD_END,
                      ) -> pd.DataFrame:
    """
    Carica le annotazioni ufficiali ESA-ADB e ritorna gli eventi UNICI
    per ID, ristretti a `category` (es. "Anomaly" = 29 nel test lightweight),
    ai canali richiesti e al periodo del test set.

    Per ogni ID l'intervallo dell'evento è (min StartTime, max EndTime)
    su tutte le righe (un evento può coinvolgere più canali).

    Returns
    -------
    DataFrame con colonne: ID, StartTime, EndTime, Channels (lista canali).
    """
    if channels is None:
        channels = CHANNELS

    lab = pd.read_csv(ESAADB_LABELS_CSV)
    typ = pd.read_csv(ESAADB_ANOMTYPES_CSV)
    lab["StartTime"] = pd.to_datetime(lab["StartTime"], utc=True)
    lab["EndTime"]   = pd.to_datetime(lab["EndTime"],   utc=True)

    m = lab.merge(typ[["ID", "Category"]], on="ID", how="left")
    sel = m[
        (m["Category"] == category) &
        (m["Channel"].isin(channels)) &
        (m["StartTime"] >= t_start) &
        (m["EndTime"]   <= t_end)
    ]

    events = (
        sel.groupby("ID")
           .agg(StartTime=("StartTime", "min"),
                EndTime  =("EndTime",   "max"),
                Channels =("Channel",   lambda s: sorted(set(s))))
           .reset_index()
           .sort_values("StartTime")
           .reset_index(drop=True)
    )
    return events


def build_gt_mask_from_events(events: pd.DataFrame,
                              index: pd.DatetimeIndex) -> np.ndarray:
    """
    Maschera GT True nei sample che cadono in almeno uno degli eventi
    annotati (sui canali del lightweight subset).
    """
    gt = np.zeros(len(index), dtype=bool)
    for _, e in events.iterrows():
        gt |= (index >= e["StartTime"]) & (index <= e["EndTime"])
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Maschera predetta dalle finestre con CP accettati
# ─────────────────────────────────────────────────────────────────────────────

def build_predicted_mask_from_results(results_df: pd.DataFrame,
                                       test_index: pd.DatetimeIndex
                                       ) -> np.ndarray:
    """
    Costruisce un boolean mask sull'indice del test set:
    True nei sample appartenenti a finestre [win_start, win_end] che hanno
    n_changepoints > 0 (cioè CP che hanno superato il gate gain nominale).

    Nota: questa è l'interpretazione conservativa. Una versione più precisa
    richiederebbe i timestamp dei CP individuali, che il CSV attuale non
    salva. L'effetto è: ogni finestra "triggerata" viene marcata interamente
    come regione di anomalia predetta.
    """
    pred = np.zeros(len(test_index), dtype=bool)
    triggered = results_df[results_df["n_changepoints"] > 0]
    for _, r in triggered.iterrows():
        s = pd.Timestamp(r["win_start"])
        e = pd.Timestamp(r["win_end"])
        if s.tzinfo is None:
            s = s.tz_localize("UTC")
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        in_win = (test_index >= s) & (test_index <= e)
        pred |= in_win
    return pred


def predicted_events_from_mask(pred_mask: np.ndarray,
                                index: pd.DatetimeIndex
                                ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Componenti connesse della pred_mask come (start_ts, end_ts)."""
    events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_evt, start_i = False, -1
    for i, v in enumerate(pred_mask):
        if v and not in_evt:
            in_evt, start_i = True, i
        elif not v and in_evt:
            events.append((index[start_i], index[i - 1]))
            in_evt = False
    if in_evt:
        events.append((index[start_i], index[-1]))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Metrica corretta event-wise F0.5 (paper ESA-ADB)
# ─────────────────────────────────────────────────────────────────────────────

def _interval_overlaps_mask(start: pd.Timestamp, end: pd.Timestamp,
                            mask: np.ndarray, index: pd.DatetimeIndex) -> bool:
    """True se ALMENO UN sample dell'index in [start, end] ha mask=True."""
    in_evt = (index >= start) & (index <= end)
    return bool(np.any(mask & in_evt))


def _to_pairs(events) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Normalizza eventi (DataFrame o lista) in lista di (start, end)."""
    if events is None:
        return []
    if isinstance(events, pd.DataFrame):
        return [(r["StartTime"], r["EndTime"]) for _, r in events.iterrows()]
    return list(events)


def compute_corrected_event_wise_f_beta(
    gt_events:   "pd.DataFrame | List[Tuple[pd.Timestamp, pd.Timestamp]]",
    pred_mask:   np.ndarray,
    gt_mask:     np.ndarray,
    index:       pd.DatetimeIndex,
    beta:        float = 0.5,
    rare_events: "pd.DataFrame | List | None" = None,
) -> dict:
    """
    Formula paper (ESA-ADB, corrected event-wise):

        P_e   = TP_e / (TP_e + FP_e)
        TNR_t = TN_t / N_t       (tempi in nanosecondi)
        P_corr = P_e * TNR_t
        R_e   = TP_e / (TP_e + FN_e)
        F_β   = (1 + β²) · P_corr · R_e / (β² · P_corr + R_e)

    Dove:
      TP_e — evento annotato con ≥1 sample predetto come anomaly
      FN_e — evento annotato senza alcun sample predetto
      FP_e — evento predetto (run di pred_mask) che NON interseca alcun
             evento GT e NON interseca alcun rare nominal event
      N_t  — tempo nominale "puro" (non anomalia, non rare event) in ns
      TN_t — tempo nominale puro correttamente non-predetto in ns

    Esclusione rare events (paper, Suppl. Mat. 3.2.2):
      Nel calcolo "anomalies only" i rare nominal events sono eventi ESCLUSI.
      Le detection che cadono su un rare event vengono IGNORATE — non contano
      come FP — e il loro tempo è escluso da N_t / TN_t.
    """
    if len(index) < 2:
        raise ValueError("Indice test troppo corto")

    gt_pairs   = _to_pairs(gt_events)
    rare_pairs = _to_pairs(rare_events)

    # ── Event-wise TP / FN ───────────────────────────────────────────────────
    tp_e = sum(1 for (s, e) in gt_pairs
               if _interval_overlaps_mask(s, e, pred_mask, index))
    fn_e = len(gt_pairs) - tp_e
    n_gt = len(gt_pairs)

    # ── Event-wise FP (con esclusione rare events) ───────────────────────────
    pred_events = predicted_events_from_mask(pred_mask, index)
    fp_e        = 0
    fp_ignored  = 0   # detection su rare event → ignorate, non penalizzate
    for (s, e) in pred_events:
        overlaps_gt = any((s <= ge) and (e >= gs) for (gs, ge) in gt_pairs)
        if overlaps_gt:
            continue  # detection corretta, non è FP
        overlaps_rare = any((s <= re) and (e >= rs) for (rs, re) in rare_pairs)
        if overlaps_rare:
            fp_ignored += 1   # cade su un rare event → ignorata (paper 3.2.2)
        else:
            fp_e += 1

    # ── Tempi in nanosecondi (paper: dominio temporale, non sample) ──────────
    ts_ns = index.view("int64")
    dt_ns = np.diff(ts_ns)
    if len(dt_ns) == 0:
        return {"error": "indice troppo corto"}
    dt_full = np.append(dt_ns, np.median(dt_ns)).astype(np.int64)

    # Maschera rare events: il loro tempo è escluso dal conteggio nominale
    rare_mask = np.zeros(len(index), dtype=bool)
    for (rs, re) in rare_pairs:
        rare_mask |= (index >= rs) & (index <= re)

    # Tempo nominale "puro": né anomalia, né rare event
    nominal_mask = ~gt_mask & ~rare_mask
    N_t  = int(dt_full[nominal_mask].sum())
    TN_t = int(dt_full[nominal_mask & ~pred_mask].sum())
    TNR_t = TN_t / N_t if N_t > 0 else 0.0

    P_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0.0
    R_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
    P_corr = P_e * TNR_t

    if (beta ** 2 * P_corr + R_e) > 0:
        F_beta = (1 + beta ** 2) * P_corr * R_e / (beta ** 2 * P_corr + R_e)
    else:
        F_beta = 0.0

    return {
        "n_gt_events":   n_gt,
        "n_pred_events": len(pred_events),
        "TP_e":  tp_e,
        "FP_e":  fp_e,
        "FP_ignored_rare": fp_ignored,
        "FN_e":  fn_e,
        "P_e":   P_e,
        "R_e":   R_e,
        "TN_t_ns":  TN_t,
        "N_t_ns":   N_t,
        "TNR_t": TNR_t,
        "P_corr": P_corr,
        "F_beta": F_beta,
        "beta":   beta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(test_csv: str, results_csv: str, beta: float = 0.5,
             include_rare: bool = False) -> dict:
    """
    include_rare:
        False (default) → ANOMALIES-ONLY (Suppl. Table 9 del paper):
                          GT = solo eventi Category=Anomaly (29).
                          I rare events sono ESCLUSI dai FP (le detection che
                          cadono su rare event vengono ignorate).
        True  → ALL EVENTS (Table 2 del paper):
                GT = anomaly + rare events insieme come positivi (65 totali).
                Nessuna esclusione: una detection su un rare event diventa TP.
    """
    mode_label = "ALL EVENTS (Table 2)" if include_rare else "ANOMALIES ONLY (Suppl. Table 9)"
    print(f"\n{'#'*70}")
    print(f"# EVAL PAPER-ALIGNED (β={beta})")
    print(f"# Test CSV:    {test_csv}")
    print(f"# Results CSV: {results_csv}")
    print(f"# Mode:        {mode_label}")
    print(f"{'#'*70}\n")

    print("[eval] Carico test set...")
    df_test, _, _ = load_dataset(
        csv_path=test_csv, use_channels=CHANNELS_41_46,
        include_labels=False, interpolate=True, fill_strategy="both",
    )
    df_test = df_test.sort_index()
    print(f"  {len(df_test)} samples")

    # Verifica spacing temporale ≈ 30 s
    if len(df_test) >= 2:
        dt = (df_test.index[1] - df_test.index[0]).total_seconds()
        print(f"  Spacing campionario: {dt:.3f} s (atteso ~30 s ⇒ 0.033 Hz)")

    print("\n[eval] Carico annotazioni ufficiali ESA-ADB...")
    anom_df = load_paper_events(
        category=TARGET_CATEGORY, channels=CHANNELS,
        t_start=TEST_PERIOD_START, t_end=TEST_PERIOD_END,
    )
    rare_df = load_paper_events(
        category=RARE_CATEGORY, channels=CHANNELS,
        t_start=TEST_PERIOD_START, t_end=TEST_PERIOD_END,
    )
    print(f"  → {len(anom_df)} eventi 'Anomaly' (atteso paper: 29)")
    print(f"  → {len(rare_df)} 'Rare Event' (atteso paper: 36)")

    if include_rare:
        # Table 2: anomalie + rare events come GT positivo unico
        events_for_eval = pd.concat([anom_df, rare_df], ignore_index=True)
        rare_for_exclusion = None
        print(f"  → All-events: {len(events_for_eval)} eventi totali come GT positivo "
              f"(29+36)")
    else:
        # Suppl. Table 9: solo anomalie come GT, rare events esclusi dai FP
        events_for_eval = anom_df
        rare_for_exclusion = rare_df
        print(f"  → Anomalies-only: {len(events_for_eval)} eventi GT, "
              f"{len(rare_df)} rare events ESCLUSI dai FP")

    gt_mask = build_gt_mask_from_events(events_for_eval, df_test.index)
    print(f"  → {gt_mask.sum()} samples in stato 'evento positivo'")

    print("\n[eval] Carico risultati per-finestra...")
    results = pd.read_csv(results_csv)
    triggered = results[results["n_changepoints"] > 0]
    print(f"  {len(results)} finestre nel CSV — {len(triggered)} con CP accettati")

    pred_mask = build_predicted_mask_from_results(results, df_test.index)
    print(f"  pred_mask: {pred_mask.sum()} samples flaggati anomaly su {len(df_test)}")

    print(f"\n[eval] Calcolo metriche corrected event-wise F{beta}...")
    out = compute_corrected_event_wise_f_beta(
        events_for_eval, pred_mask, gt_mask, df_test.index, beta=beta,
        rare_events=rare_for_exclusion,
    )

    gt_label = "ALL EVENTS" if include_rare else "ANOMALY"
    print(f"\n{'='*70}")
    print(f"PAPER-ALIGNED RESULTS — {mode_label}  (β={beta})")
    print(f"{'='*70}")
    print(f"  Eventi GT ({gt_label}):     {out['n_gt_events']}")
    print(f"  Eventi predetti (runs):    {out['n_pred_events']}")
    print(f"  TP_e:                      {out['TP_e']}")
    print(f"  FP_e:                      {out['FP_e']}")
    if not include_rare:
        print(f"  FP ignorati (su rare evt): {out['FP_ignored_rare']}")
    print(f"  FN_e:                      {out['FN_e']}")
    print(f"  P_e (event-wise):          {out['P_e']:.4f}")
    print(f"  R_e (event-wise):          {out['R_e']:.4f}")
    print(f"  TN_t (ns):                 {out['TN_t_ns']:.3e}")
    print(f"  N_t  (ns):                 {out['N_t_ns']:.3e}")
    print(f"  TNR_t:                     {out['TNR_t']:.6f}")
    print(f"  P_corr = P_e * TNR_t:      {out['P_corr']:.4f}")
    print(f"  F_β (β={beta}):              {out['F_beta']:.4f}")
    print(f"{'='*70}\n")

    # Confronto col conteggio per-finestra
    n_win_anom = int(results[results["gt_label"] == "ANOMALY"].shape[0]) \
                 if "gt_label" in results.columns else 0
    if n_win_anom:
        print("[eval] Riconciliazione 'finestre' vs 'eventi':")
        print(f"  - Finestre del CSV con GT:ANOMALY: {n_win_anom}")
        print(f"  - Eventi GT reali (paper):         {out['n_gt_events']}")
        print(f"  → Una singola anomalia può coprire più finestre da 5 giorni.\n")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["eval", "rerun"], default="eval",
                    help="'eval' (default): solo metriche paper sul CSV esistente. "
                         "'rerun': standardizza + esegue la pipeline + eval.")
    ap.add_argument("--test-csv", default=TEST_CSV,
                    help=f"Test CSV (default: {TEST_CSV})")
    ap.add_argument("--train-csv", default=TRAIN_CSV,
                    help=f"Train CSV (default: {TRAIN_CSV})")
    ap.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV,
                    help=f"Per-window results CSV (default: {DEFAULT_RESULTS_CSV})")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="β del F-score (paper: 0.5)")
    ap.add_argument("--standardize", action="store_true",
                    help="Solo con --mode=rerun: standardizza i CSV prima di rieseguire.")
    ap.add_argument("--include-rare", action="store_true",
                    help="Eval in modalità 'all events' (Table 2 del paper): "
                         "anomaly + rare events come GT positivo unico. "
                         "Default (no flag) = 'anomalies only' (Suppl. Table 9).")
    args = ap.parse_args()

    if args.mode == "rerun":
        from pen_sweep_update import run_complete_pipeline

        if args.standardize:
            print("[mode=rerun + standardize] Pre-processing standardizzazione...")
            build_standardized_csvs(args.train_csv, args.test_csv,
                                    PAPER_TRAIN_CSV, PAPER_TEST_CSV)
            train_csv = PAPER_TRAIN_CSV
            test_csv  = PAPER_TEST_CSV
        else:
            train_csv = args.train_csv
            test_csv  = args.test_csv

        print(f"\n[mode=rerun] Output → {PAPER_OUTPUT_DIR}")
        print("[mode=rerun] Eseguo pipeline CPD completa (~ore)...")
        run_complete_pipeline(
            test_csv=test_csv,
            train_csv=train_csv,
            output_dir=PAPER_OUTPUT_DIR,
            save_plots=True,
        )
        evaluate(test_csv=test_csv,
                 results_csv=str(Path(PAPER_OUTPUT_DIR) / "test_results.csv"),
                 beta=args.beta,
                 include_rare=args.include_rare)
    else:
        evaluate(test_csv=args.test_csv,
                 results_csv=args.results_csv,
                 beta=args.beta,
                 include_rare=args.include_rare)


if __name__ == "__main__":
    main()
