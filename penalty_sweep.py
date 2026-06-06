import numpy as np
import ruptures as rpt

from config import PENALTY_SWEEP_STEPS, PENALTY_JUMP_THRESHOLD, CPD_MIN_SIZE, CPD_JUMP, PENALTY_BIC_MULTIPLIER


def find_optimal_penalty(
    X: np.ndarray,
    n_steps: int = PENALTY_SWEEP_STEPS,
    jump_threshold: int = PENALTY_JUMP_THRESHOLD,
    min_size: int = CPD_MIN_SIZE,
    jump: int = CPD_JUMP,
) -> tuple:
    """
    Sweep penalty alto → basso su PELT (L2) e trova il "ginocchio".

    Il ginocchio è il punto poco prima che il numero di CP esploda:
    aggiungere un CP ha ancora un costo significativo → i CP trovati
    corrispondono a cambiamenti strutturali reali, non a rumore.

    Parameters
    ----------
    X : np.ndarray  shape (n,) o (n, D)
    n_steps : int   numero di valori di penalty nel sweep
    jump_threshold : int  salto minimo di CP per identificare l'esplosione

    Returns
    -------
    pen_optimal : float
    sweep_info  : dict
        pen_values, cp_counts, costs, knee_idx, k_optimal,
        C0, C_optimal, delta, score
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    nT, D = X.shape

    # --- Calibra range di penalty su sigma² della serie ---
    sigma2 = max(float(np.mean(np.var(X, axis=0))), 1e-9)
    pen_max = sigma2 * nT * D * 20   # garantisce 0 CP
    pen_min = max(sigma2 * D * 0.05, 1e-6)  # permissivo

    pen_values = np.logspace(np.log10(pen_max), np.log10(pen_min), n_steps)

    algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump).fit(X)

    # C0: costo dell'intera serie senza CP
    C0 = float(algo.cost.sum_of_costs([nT]))

    cp_counts = []
    costs = []
    for pen in pen_values:
        try:
            bkps = algo.predict(pen=float(pen))
            k = len(bkps) - 1
            C = float(algo.cost.sum_of_costs(bkps))
        except Exception:
            k = 0
            C = C0
        cp_counts.append(k)
        costs.append(C)

    cp_counts = np.array(cp_counts, dtype=int)
    costs = np.array(costs, dtype=float)

    # --- Trova il ginocchio ---
    knee_idx = _find_knee(cp_counts, jump_threshold)

    pen_optimal = float(pen_values[knee_idx])
    k_optimal   = int(cp_counts[knee_idx])
    C_optimal   = float(costs[knee_idx])
    delta       = C0 - C_optimal

    # Score: riduzione di costo per CP aggiunto
    # → alto per serie con pochi CP strutturali (buone per anomaly detection)
    # → basso per serie oscillatorie (tanti CP piccoli)
    score = delta / max(k_optimal, 1)

    # Soglia BIC: il ginocchio deve essere sopra log(n)*D per essere significativo.
    # Se pen* < soglia, il cambio è solo noise fitting → non è un'anomalia reale.
    pen_bic        = float(np.log(max(nT, 2)) * D * PENALTY_BIC_MULTIPLIER)
    is_significant = pen_optimal >= pen_bic

    return pen_optimal, {
        "pen_values":     pen_values,
        "cp_counts":      cp_counts,
        "costs":          costs,
        "knee_idx":       knee_idx,
        "k_optimal":      k_optimal,
        "C0":             C0,
        "C_optimal":      C_optimal,
        "delta":          delta,
        "score":          score,
        "pen_max":        pen_max,
        "pen_min":        pen_min,
        "pen_bic":        pen_bic,
        "is_significant": is_significant,
    }


def _find_knee(cp_counts: np.ndarray, jump_threshold: int) -> int:
    """
    Trova l'indice del ginocchio nella curva penalty → n_CP.

    Strategia 1 (priorità): primo salto >= jump_threshold → ci fermiamo
    al punto precedente.
    Se quel punto ha 0 CP (il salto esplode già al primo step), avanziamo
    al primo indice con k >= 1: la penalità più alta che produce ancora CP.
    Strategia 2 (fallback): massima curvatura (metodo della corda).
    """
    n = len(cp_counts)

    # Metodo 1: primo salto brusco
    for i in range(1, n):
        if cp_counts[i] - cp_counts[i - 1] >= jump_threshold:
            knee = i - 1
            if cp_counts[knee] == 0:
                # Il ginocchio è a 0 CP: usa il primo step con k >= 1
                for j in range(i, n):
                    if cp_counts[j] >= 1:
                        return j
                return knee  # nessun CP in tutto lo sweep
            return knee

    # Metodo 2: massima curvatura
    return _max_curvature_idx(cp_counts)


def _max_curvature_idx(cp_counts: np.ndarray) -> int:
    """
    Distanza massima dalla corda tra il primo e l'ultimo punto della curva,
    entrambi normalizzati in [0, 1].
    """
    n = len(cp_counts)
    if n < 3:
        return 0

    x = np.linspace(0.0, 1.0, n)           # asse penalty normalizzato
    y = cp_counts.astype(float)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Vettore corda: da (x[0], y_norm[0]) a (x[-1], y_norm[-1])
    dx = x[-1] - x[0]
    dy = y_norm[-1] - y_norm[0]
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-12

    dists = np.abs(dy * x - dx * y_norm + x[0] * dy - y_norm[0] * dx) / norm

    return int(np.argmax(dists))


# ===========================
# Analisi per-dimensione (approccio del professore)
# ===========================

def _l2_cost_segment(x: np.ndarray) -> float:
    """Costo L2 di un segmento 1D = somma scarti quadratici dalla media."""
    if len(x) < 2:
        return 0.0
    return float(np.sum((x - x.mean()) ** 2))


def per_dimension_cost_analysis(X: np.ndarray, bkps: list) -> np.ndarray:
    """
    Dati i CP globali (trovati su X multivariato), calcola per ogni dimensione:

        gain_d = C0_d - Σ_segmenti L2(X_d in segmento)

    Alto gain_d  → la dimensione d spiega il CP (cambio strutturale reale).
    Basso gain_d → il CP è invisibile in questa dimensione (rumore).

    Il professore: "calcoli costi prima e dopo per i CP globali ma il costo
    lo calcoli individualmente".

    Usa il costo L2 diretto (O(n·D)) senza chiamare PELT per ogni dimensione.

    Parameters
    ----------
    X    : np.ndarray  shape (T, D)
    bkps : list        breakpoints PELT (include T come ultimo elemento)
    """
    nT, D = X.shape
    boundaries = [0] + sorted(bkps)
    gains = np.zeros(D)

    for d in range(D):
        x_d = X[:, d]
        C0_d = _l2_cost_segment(x_d)
        C_split = sum(
            _l2_cost_segment(x_d[boundaries[i]: boundaries[i + 1]])
            for i in range(len(boundaries) - 1)
        )
        gains[d] = max(0.0, C0_d - C_split)

    return gains


def elbow_contributing_dims(
    gains: np.ndarray,
    min_contributing: int = 1,
) -> tuple:
    """
    Trova le dimensioni sopra il ginocchio nella curva dei gain (ordinata
    decrescente) tramite il metodo della corda.

    La curva decresce da (rank=0, gain_max) a (rank=D-1, gain_min).
    Il punto con distanza massima dalla corda diagonale è il "gomito":
    le dimensioni a sinistra del gomito sono quelle che contribuiscono
    davvero al CP; quelle a destra sono rumore.

    Il professore dice: "guarda l'elbow, se hai frequenze che si discostano
    molto" → questa funzione identifica quelle frequenze.

    Parameters
    ----------
    gains           : np.ndarray  gain per dimensione (shape D,)
    min_contributing: int         minimo di dimensioni da selezionare

    Returns
    -------
    n_contributing   : int
    threshold_gain   : float   (gain minimo delle dimensioni selezionate)
    contributing_mask: np.ndarray[bool]  shape (D,)
    """
    D = len(gains)
    sorted_idx = np.argsort(gains)[::-1]   # decrescente per gain
    sorted_g   = gains[sorted_idx]

    pos_count = int((sorted_g > 0).sum())

    if pos_count == 0:
        return 0, 0.0, np.zeros(D, dtype=bool)

    if pos_count == 1:
        mask = np.zeros(D, dtype=bool)
        mask[sorted_idx[0]] = True
        return 1, float(sorted_g[0]), mask

    g = sorted_g[:pos_count]

    # Curva decrescente normalizzata: (x=0..1, y=1..0)
    x_norm = np.linspace(0.0, 1.0, pos_count)
    y_norm = (g - g[-1]) / (g[0] - g[-1] + 1e-12)  # 1 al max, 0 al min

    # Corda da (0,1) a (1,0): retta x + y = 1
    # Distanza punto (x_i, y_i) dalla retta: |x_i + y_i - 1| / sqrt(2)
    dists = np.abs(x_norm + y_norm - 1.0) / np.sqrt(2.0)

    elbow_idx  = int(np.argmax(dists))
    n_selected = max(min_contributing, elbow_idx + 1)
    n_selected = min(n_selected, pos_count)

    threshold_gain = float(g[n_selected - 1])
    mask = np.zeros(D, dtype=bool)
    mask[sorted_idx[:n_selected]] = True

    return n_selected, threshold_gain, mask
