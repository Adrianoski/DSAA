#!/bin/bash
#
# run_clf_knee.sh — pipeline completa classifier al threshold KNEE
# (parameter-free, derivato dalla distribuzione max_gain del training).
#
# Pre-requisiti:
#   - calibrate_gain.py già eseguito → file gain_distribution_summary.json
#
# Stadi:
#   0. Legge il knee da gain_distribution_summary.json
#   1. generate-train    → cp_training_dataset_knee.csv
#   2. train-classifier  → cp_classifier_knee.joblib + metrics
#   3. test              → cpd_output_classified_knee/test_results.csv
#   4. eval clf anomalies-only      → log_eval_clf_anomonly_knee.txt
#   5. eval clf all-events          → log_eval_clf_allevents_knee.txt
#   6. crea "unsup-view" del CSV    → test_results_unsup_view.csv
#   7. eval unsup anomalies-only    → log_eval_unsup_anomonly_knee.txt
#   8. eval unsup all-events        → log_eval_unsup_allevents_knee.txt
#

set -e
cd "$(dirname "$0")"

SUMMARY_JSON=${1:-gain_distribution_summary.json}
SUFFIX=knee

if [ ! -f "$SUMMARY_JSON" ]; then
    echo "ERROR: $SUMMARY_JSON non trovato."
    echo "Lancia prima: python3 calibrate_gain.py"
    exit 1
fi

THR=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(d['knee']['knee_gain'])")
KNEE_PERC=$(python3 -c "import json; d=json.load(open('$SUMMARY_JSON')); print(d['knee']['knee_at_percentile'])")

DATASET="cp_training_dataset_${SUFFIX}.csv"
MODEL="cp_classifier_${SUFFIX}.joblib"
METRICS="cp_classifier_${SUFFIX}_metrics.json"
OUT_DIR="cpd_output_classified_${SUFFIX}"

LOG_GEN="log_gen_train_${SUFFIX}.txt"
LOG_TRAIN="log_train_clf_${SUFFIX}.txt"
LOG_TEST="log_test_clf_${SUFFIX}.txt"

LOG_EVAL_CLF_ANOM="log_eval_clf_anomonly_${SUFFIX}.txt"
LOG_EVAL_CLF_ALL="log_eval_clf_allevents_${SUFFIX}.txt"
LOG_EVAL_UNSUP_ANOM="log_eval_unsup_anomonly_${SUFFIX}.txt"
LOG_EVAL_UNSUP_ALL="log_eval_unsup_allevents_${SUFFIX}.txt"

UNSUP_TEST_CSV="84_months.test.unsup_std.csv"
CLF_RESULTS="${OUT_DIR}/test_results.csv"
UNSUP_VIEW_CSV="${OUT_DIR}/test_results_unsup_view.csv"

echo "############################################################"
echo "# KNEE-BASED PIPELINE"
echo "# Knee threshold:   ${THR}"
echo "# At percentile:    ${KNEE_PERC}°"
echo "# Suffix:           ${SUFFIX}"
echo "# Started:          $(date '+%Y-%m-%d %H:%M:%S')"
echo "# PID:              $$"
echo "############################################################"

# ── 1. generate-train ────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] STEP 1/8 — generate-train → ${LOG_GEN}"
python3 -u pen_sweep_classified.py --mode generate-train \
    --gain-threshold "${THR}" \
    --dataset "${DATASET}" \
    > "${LOG_GEN}" 2>&1
echo "[$(date '+%H:%M:%S')] STEP 1 done"

# ── 2. train-classifier ──────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] STEP 2/8 — train-classifier → ${LOG_TRAIN}"
python3 -u pen_sweep_classified.py --mode train-classifier \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --metrics "${METRICS}" \
    > "${LOG_TRAIN}" 2>&1
echo "[$(date '+%H:%M:%S')] STEP 2 done"

# ── 3. test ──────────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] STEP 3/8 — test → ${LOG_TEST}"
python3 -u pen_sweep_classified.py --mode test \
    --gain-threshold "${THR}" \
    --model "${MODEL}" \
    --output-dir "${OUT_DIR}" \
    > "${LOG_TEST}" 2>&1
echo "[$(date '+%H:%M:%S')] STEP 3 done"

# ── 4-5. eval classifier (anomalies-only + all-events) ───────────────────────
echo "[$(date '+%H:%M:%S')] STEP 4/8 — eval CLF anomalies-only → ${LOG_EVAL_CLF_ANOM}"
python3 -u pen_sweep_paper.py --mode eval \
    --test-csv "${UNSUP_TEST_CSV}" \
    --results-csv "${CLF_RESULTS}" \
    > "${LOG_EVAL_CLF_ANOM}" 2>&1

echo "[$(date '+%H:%M:%S')] STEP 5/8 — eval CLF all-events → ${LOG_EVAL_CLF_ALL}"
python3 -u pen_sweep_paper.py --mode eval --include-rare \
    --test-csv "${UNSUP_TEST_CSV}" \
    --results-csv "${CLF_RESULTS}" \
    > "${LOG_EVAL_CLF_ALL}" 2>&1

# ── 6. unsup-view del CSV classified ─────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] STEP 6/8 — produce unsup view del CSV (n_cps_initial)"
python3 eval_unsup_from_clf_csv.py "${CLF_RESULTS}" "${UNSUP_VIEW_CSV}"

# ── 7-8. eval unsup-view (anomalies-only + all-events) ───────────────────────
echo "[$(date '+%H:%M:%S')] STEP 7/8 — eval UNSUP@knee anomalies-only → ${LOG_EVAL_UNSUP_ANOM}"
python3 -u pen_sweep_paper.py --mode eval \
    --test-csv "${UNSUP_TEST_CSV}" \
    --results-csv "${UNSUP_VIEW_CSV}" \
    > "${LOG_EVAL_UNSUP_ANOM}" 2>&1

echo "[$(date '+%H:%M:%S')] STEP 8/8 — eval UNSUP@knee all-events → ${LOG_EVAL_UNSUP_ALL}"
python3 -u pen_sweep_paper.py --mode eval --include-rare \
    --test-csv "${UNSUP_TEST_CSV}" \
    --results-csv "${UNSUP_VIEW_CSV}" \
    > "${LOG_EVAL_UNSUP_ALL}" 2>&1

# ── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "# KNEE PIPELINE DONE @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "# Knee threshold: ${THR}  (at ${KNEE_PERC}° percentile)"
echo "############################################################"

extract_metrics () {
    local label="$1"
    local log="$2"
    echo ""
    echo "── ${label} ──"
    if [ -f "$log" ]; then
        grep -E "Eventi GT|Eventi predetti|TP_e:|FP_e:|FP ignorati|FN_e:|P_e |R_e |TNR_t:|P_corr|F_β" "$log" | head -12
    else
        echo "  (log non trovato: $log)"
    fi
}

extract_metrics "UNSUP @ knee — ANOMALIES ONLY"      "${LOG_EVAL_UNSUP_ANOM}"
extract_metrics "UNSUP @ knee — ALL EVENTS"          "${LOG_EVAL_UNSUP_ALL}"
extract_metrics "CLF   @ knee — ANOMALIES ONLY"      "${LOG_EVAL_CLF_ANOM}"
extract_metrics "CLF   @ knee — ALL EVENTS"          "${LOG_EVAL_CLF_ALL}"
