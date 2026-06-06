import os
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="parsing timezone aware datetimes is deprecated"
)

# ===========================
# Channels & Labels
# ===========================
CHANNELS = [f"channel_{i}" for i in range(41, 47)]
LABELS   = [f"is_anomaly_channel_{i}" for i in range(41, 47)]
CATS     = {"Anomaly", "Rare Event"}

# ===========================
# Windowing
# ===========================
WINDOW_SIZE_HOURS   = 24
WINDOW_OVERLAP_FRAC = 0.5
WINDOW_DAYS         = 5   # finestre fisse sul test set

# ===========================
# STFT
# ===========================
STFT_NPERSEG      = 128
STFT_OVERLAP_FRAC = 0.75
STFT_WINDOW       = "hann"
MIN_BAND_HZ       = 0.0
MAX_BAND_HZ       = 0.02

# ===========================
# Penalty Sweep
# ===========================
PENALTY_SWEEP_STEPS    = 60   # numero di step nel sweep log-spaced
PENALTY_JUMP_THRESHOLD = 2    # salto minimo di CP per identificare l'esplosione

# Soglia BIC per evitare falsi positivi:
# il ginocchio deve trovarsi a pen* >= PENALTY_BIC_MULTIPLIER * log(n) * D
# Se pen* è sotto soglia il segnale è rumore → nessun CP riportato.
# Valori tipici: 1.0 (conservativo) … 0.5 (più sensibile)
PENALTY_BIC_MULTIPLIER = 1.0

# ===========================
# Frequency Selection (Cost-based)
# ===========================
TRAIN_VAL_SPLIT_RATIO = 0.7   # 70% train_sel, 30% val_sel
MIN_RELATIVE_GAIN_VAL = 0.03  # 3% guadagno minimo su validation
MAX_FEATURES          = 3

# ===========================
# CPD
# ===========================
DAYS_PAD     = 5
CPD_MIN_SIZE = 2
CPD_JUMP     = 2

# ===========================
# Per-dimension cost analysis (idea del professore)
# ===========================
# Numero minimo di dimensioni (canale×frequenza) che devono mostrare
# un gain significativo (sopra il gomito) per accettare un CP come reale.
# "Devo essere molto conservativo" → almeno 2 dimensioni devono concordare.
PER_DIM_MIN_CONTRIBUTING = 2

# ===========================
# Metriche Range-based
# ===========================
ALPHA_RECALL   = 0.0
RECALL_BIAS    = 'front'
PRECISION_BIAS = 'flat'
BETA_SCORE     = 1.0

# ===========================
# Metriche Classic P/R
# ===========================
PR_MARGIN_FR = 20

# ===========================
# IoU
# ===========================
MAX_GAP_SECONDS = 86400
MIN_CLUSTER_IOU = 0.1

# ===========================
# Paths
# ===========================
ESA_ADB_DATA = os.environ.get("ESA_ADB_DATA", "data")  # root dati ESA-ADB (override: export ESA_ADB_DATA=/percorso)
TRAIN_CSV  = os.path.join(ESA_ADB_DATA, "preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.train.csv")
TEST_CSV   = os.path.join(ESA_ADB_DATA, "preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv")
OUTPUT_DIR = "cpd_penalty_sweep_output"

FREQ_REPORT_FILE    = "frequency_cost_evaluation.json"
SELECTED_FREQS_FILE = "selected_frequencies_cpd.json"
