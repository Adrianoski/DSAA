"""
metrics.py
"""

import numpy as np
import pandas as pd

from config import (
    MAX_GAP_SECONDS, MIN_CLUSTER_IOU,
    ALPHA_RECALL, RECALL_BIAS, PRECISION_BIAS, BETA_SCORE,
    PR_MARGIN_FR,
)

try:
    from ruptures.metrics import precision_recall as rpt_precision_recall
    _HAS_RPT = True
except Exception:
    _HAS_RPT = False


# ===========================
# Funzioni di bias posizionale
# ===========================

def positional_bias_flat(i, anomaly_length):
    return 1.0

def positional_bias_front(i, anomaly_length):
    return float(anomaly_length - i)

def positional_bias_back(i, anomaly_length):
    return float(i + 1)

def positional_bias_middle(i, anomaly_length):
    mid = anomaly_length / 2.0
    return float(i + 1) if i <= mid else float(anomaly_length - i)

_BIAS_MAP = {
    "flat":   positional_bias_flat,
    "front":  positional_bias_front,
    "back":   positional_bias_back,
    "middle": positional_bias_middle,
}


# ===========================
# Range-based Precision & Recall
# ===========================

def omega_overlap_size(anomaly_range, overlap_set, delta_func):
    my_value = 0.0
    max_value = 0.0
    anomaly_length = len(anomaly_range)
    for i, point in enumerate(anomaly_range):
        bias = delta_func(i, anomaly_length)
        max_value += bias
        if point in overlap_set:
            my_value += bias
    return my_value / max_value if max_value > 0 else 0.0


def cardinality_factor_reciprocal(num_overlaps):
    return 1.0 if num_overlaps <= 1 else 1.0 / float(num_overlaps)


def existence_reward(real_range_indices, all_predicted_ranges):
    for pred_range in all_predicted_ranges:
        if len(real_range_indices & pred_range) >= 1:
            return 1.0
    return 0.0


def changepoints_to_ranges(cp_indices, n_frames):
    if len(cp_indices) == 0:
        return []
    boundaries = [0] + sorted(cp_indices) + [n_frames]
    ranges = []
    for i in range(len(boundaries) - 1):
        r = set(range(boundaries[i], boundaries[i + 1]))
        if r:
            ranges.append(r)
    return ranges


def gt_labels_to_ranges(gt_mask):
    ranges = []
    in_anomaly = False
    current_range = []
    for i, val in enumerate(gt_mask):
        if val == 1:
            if not in_anomaly:
                in_anomaly = True
                current_range = [i]
            else:
                current_range.append(i)
        else:
            if in_anomaly:
                ranges.append(set(current_range))
                in_anomaly = False
                current_range = []
    if in_anomaly:
        ranges.append(set(current_range))
    return ranges


def ranges_to_mask(ranges, n_frames):
    mask = np.zeros(n_frames, dtype=int)
    for range_set in ranges:
        for idx in range_set:
            if 0 <= idx < n_frames:
                mask[idx] = 1
    return mask


def compute_range_based_recall(
    real_ranges, predicted_ranges,
    alpha=0.0,
    delta_func=positional_bias_flat,
    gamma_func=cardinality_factor_reciprocal,
):
    if len(real_ranges) == 0:
        return 0.0, []
    recall_scores = []
    for real_range in real_ranges:
        exist_reward = existence_reward(real_range, predicted_ranges)
        overlapping_preds = [p for p in predicted_ranges if real_range & p]
        card_factor = gamma_func(len(overlapping_preds))
        overlap_reward = 0.0
        for pred_range in overlapping_preds:
            overlap = real_range & pred_range
            omega = omega_overlap_size(sorted(real_range), set(overlap), delta_func)
            overlap_reward += omega
        overlap_reward *= card_factor
        recall_scores.append(alpha * exist_reward + (1.0 - alpha) * overlap_reward)
    return float(np.mean(recall_scores)), recall_scores


def compute_range_based_precision(
    real_ranges, predicted_ranges,
    delta_func=positional_bias_flat,
    gamma_func=cardinality_factor_reciprocal,
):
    if len(predicted_ranges) == 0:
        return 0.0, []
    precision_scores = []
    for pred_range in predicted_ranges:
        overlapping_reals = [r for r in real_ranges if pred_range & r]
        card_factor = gamma_func(len(overlapping_reals))
        overlap_reward = 0.0
        for real_range in overlapping_reals:
            overlap = pred_range & real_range
            omega = omega_overlap_size(sorted(pred_range), set(overlap), delta_func)
            overlap_reward += omega
        overlap_reward *= card_factor
        precision_scores.append(overlap_reward)
    return float(np.mean(precision_scores)), precision_scores


def compute_f_beta_score(precision, recall, beta=1.0):
    if precision + recall == 0:
        return 0.0
    return (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def compute_timeseries_metrics(
    cp_idx, gt_mask, n_frames,
    alpha=ALPHA_RECALL,
    recall_bias=RECALL_BIAS,
    precision_bias=PRECISION_BIAS,
    beta=BETA_SCORE,
):
    recall_delta    = _BIAS_MAP.get(recall_bias,    positional_bias_flat)
    precision_delta = _BIAS_MAP.get(precision_bias, positional_bias_flat)

    pred_ranges = changepoints_to_ranges(cp_idx, n_frames)
    real_ranges = gt_labels_to_ranges(gt_mask)

    recall_T, recall_per_range = compute_range_based_recall(
        real_ranges, pred_ranges, alpha=alpha, delta_func=recall_delta
    )
    precision_T, precision_per_range = compute_range_based_precision(
        real_ranges, pred_ranges, delta_func=precision_delta
    )
    f_beta = compute_f_beta_score(precision_T, recall_T, beta)

    pred_mask = ranges_to_mask(pred_ranges, n_frames)
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    iou_classic = intersection / union if union > 0 else 0.0

    return {
        "recall_T":           recall_T,
        "precision_T":        precision_T,
        "f_beta":             f_beta,
        "iou_classic":        iou_classic,
        "n_real_ranges":      len(real_ranges),
        "n_pred_ranges":      len(pred_ranges),
        "n_changepoints":     len(cp_idx),
        "recall_per_range":   recall_per_range,
        "precision_per_range":precision_per_range,
        "real_ranges":        real_ranges,
        "pred_ranges":        pred_ranges,
    }


# ===========================
# Classic P/R (fallback senza ruptures)
# ===========================

def pr_fallback(true_cps, pred_cps, margin):
    T = sorted(true_cps)
    P = sorted(pred_cps)
    used, tp = set(), 0
    for p in P:
        best, bestd = None, None
        for i, t in enumerate(T):
            if i in used:
                continue
            d = abs(p - t)
            if d <= margin and (bestd is None or d < bestd):
                best, bestd = i, d
        if best is not None:
            used.add(best)
            tp += 1
    prec = tp / max(1, len(P))
    rec  = tp / max(1, len(T))
    return prec, rec


def compute_classic_pr(true_bkps, bkps, true_cps_frames, cp_idx, margin=PR_MARGIN_FR):
    if _HAS_RPT:
        return rpt_precision_recall(true_bkps=true_bkps, my_bkps=bkps, margin=margin)
    return pr_fallback(true_cps_frames, cp_idx, margin)


# ===========================
# IoU per-anomalia corretta
# ===========================

def extract_gt_anomalies(gt_mask, t_stamps):
    anomalies = []
    in_anomaly = False
    current_start = None
    for i, val in enumerate(gt_mask):
        if val == 1 and not in_anomaly:
            in_anomaly = True
            current_start = i
        elif val == 0 and in_anomaly:
            current_end = i - 1
            anom_mask = np.zeros_like(gt_mask)
            anom_mask[current_start : current_end + 1] = 1
            anomalies.append({
                "id": len(anomalies),
                "start_frame": current_start,
                "end_frame": current_end,
                "start_time": pd.Timestamp(t_stamps[current_start]),
                "end_time": pd.Timestamp(t_stamps[current_end]),
                "duration_frames": current_end - current_start + 1,
                "mask": anom_mask,
            })
            in_anomaly = False
    if in_anomaly:
        current_end = len(gt_mask) - 1
        anom_mask = np.zeros_like(gt_mask)
        anom_mask[current_start : current_end + 1] = 1
        anomalies.append({
            "id": len(anomalies),
            "start_frame": current_start,
            "end_frame": current_end,
            "start_time": pd.Timestamp(t_stamps[current_start]),
            "end_time": pd.Timestamp(t_stamps[current_end]),
            "duration_frames": current_end - current_start + 1,
            "mask": anom_mask,
        })
    return anomalies


def create_segments_from_changepoints(cp_idx, n_frames, t_stamps):
    if len(cp_idx) == 0:
        return [{
            "id": 0, "start_frame": 0, "end_frame": n_frames - 1,
            "start_time": pd.Timestamp(t_stamps[0]),
            "end_time": pd.Timestamp(t_stamps[-1]),
            "duration_frames": n_frames,
            "mask": np.ones(n_frames, dtype=int),
        }]
    segments = []
    boundaries = [0] + sorted(cp_idx) + [n_frames]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1] - 1
        if end < start:
            continue
        seg_mask = np.zeros(n_frames, dtype=int)
        seg_mask[start : end + 1] = 1
        segments.append({
            "id": i, "start_frame": start, "end_frame": end,
            "start_time": pd.Timestamp(t_stamps[start]),
            "end_time": pd.Timestamp(t_stamps[min(end, n_frames - 1)]),
            "duration_frames": end - start + 1,
            "mask": seg_mask,
        })
    return segments


def cluster_segments_by_time(segments, max_gap_seconds=MAX_GAP_SECONDS):
    if not segments:
        return []
    clusters = []
    cur = [segments[0]]
    for seg in segments[1:]:
        gap = (seg["start_time"] - cur[-1]["end_time"]).total_seconds()
        if gap <= max_gap_seconds:
            cur.append(seg)
        else:
            clusters.append(_make_cluster(cur, len(clusters)))
            cur = [seg]
    clusters.append(_make_cluster(cur, len(clusters)))
    return clusters


def _make_cluster(segs, cid):
    mask = np.zeros_like(segs[0]["mask"])
    for s in segs:
        mask = mask | s["mask"]
    return {
        "id": cid,
        "segments": segs.copy(),
        "start_frame": segs[0]["start_frame"],
        "end_frame": segs[-1]["end_frame"],
        "start_time": segs[0]["start_time"],
        "end_time": segs[-1]["end_time"],
        "n_segments": len(segs),
        "mask": mask,
    }


def match_clusters_to_anomalies(clusters, anomalies):
    matches = []
    for cluster in clusters:
        best_match, best_overlap, best_iou = None, 0, 0.0
        for anomaly in anomalies:
            overlap = int(np.sum(cluster["mask"] & anomaly["mask"]))
            if overlap > best_overlap:
                union = int(np.sum(cluster["mask"] | anomaly["mask"]))
                best_match = anomaly
                best_overlap = overlap
                best_iou = overlap / union if union > 0 else 0.0
        matches.append({
            "cluster": cluster,
            "matched_anomaly": best_match,
            "overlap_frames": best_overlap,
            "iou": best_iou,
        })
    return matches


def compute_iou_corrected(
    cp_idx, gt_mask, n_frames, t_stamps,
    max_gap_seconds=MAX_GAP_SECONDS,
    min_cluster_iou=MIN_CLUSTER_IOU,
    verbose=True,
):
    if verbose:
        print("    ┌─ ANALISI IoU CORRETTA ──────────────────────────────────┐")

    gt_anomalies = extract_gt_anomalies(gt_mask, t_stamps)
    n_anomalies = len(gt_anomalies)

    if verbose:
        print(f"    │ ANOMALIE GT: {n_anomalies}")
        for a in gt_anomalies:
            dur = (a["end_time"] - a["start_time"]).total_seconds()
            print(f"    │   Anomalia {a['id']}: [{a['start_frame']:4d}→{a['end_frame']:4d}] "
                  f"({a['duration_frames']} frames, {dur:.0f}s)")

    _empty = {
        "n_anomalies": n_anomalies, "n_segments": 0,
        "n_clusters": 0, "n_selected": 0,
        "gt_anomalies": gt_anomalies,
        "segments": [], "clusters": [],
        "cluster_matches": [], "anomaly_ious": [],
        "anomaly_weights": [], "iou_per_anomaly": [],
    }

    if len(cp_idx) == 0:
        if verbose:
            print("    │ NESSUN CHANGEPOINT rilevato")
            print("    └────────────────────────────────────────────────────────┘")
        return 0.0, [], np.zeros(n_frames, dtype=int), _empty

    segments = create_segments_from_changepoints(cp_idx, n_frames, t_stamps)
    clusters = cluster_segments_by_time(segments, max_gap_seconds)
    n_clusters = len(clusters)

    if verbose:
        print(f"    │ CP: {len(cp_idx)} → {len(segments)} segmenti → {n_clusters} cluster(s)")
        print("    │ MATCHING:")

    cluster_matches = match_clusters_to_anomalies(clusters, gt_anomalies)
    selected_clusters = []
    pred_mask = np.zeros(n_frames, dtype=int)

    for match in cluster_matches:
        cluster = match["cluster"]
        anomaly = match["matched_anomaly"]
        iou     = match["iou"]
        dur = (cluster["end_time"] - cluster["start_time"]).total_seconds()

        if anomaly is not None:
            selected = iou >= min_cluster_iou
            if verbose:
                tag = "✓ SELECTED" if selected else "✗ REJECTED"
                print(f"    │   Cluster {cluster['id']}: [{cluster['start_frame']:4d}→{cluster['end_frame']:4d}]"
                      f" → Anomalia {anomaly['id']} IoU={iou:.3f} {tag}")
            if selected:
                selected_clusters.append((cluster["start_frame"], cluster["end_frame"]))
                pred_mask = pred_mask | cluster["mask"]
        else:
            if verbose:
                print(f"    │   Cluster {cluster['id']}: [{cluster['start_frame']:4d}→{cluster['end_frame']:4d}]"
                      f" → NO MATCH ✗")

    anomaly_ious = []
    anomaly_weights = []
    iou_per_anomaly = []

    for anom in gt_anomalies:
        matched = [
            m["cluster"] for m in cluster_matches
            if m["matched_anomaly"] is not None
               and m["matched_anomaly"]["id"] == anom["id"]
               and m["iou"] >= min_cluster_iou
        ]
        if matched:
            pm = np.zeros(n_frames, dtype=int)
            for c in matched:
                pm = pm | c["mask"]
            inter = int(np.sum(pm & anom["mask"]))
            union = int(np.sum(pm | anom["mask"]))
            iou_a = inter / union if union > 0 else 0.0
        else:
            iou_a = 0.0
        anomaly_ious.append(iou_a)
        anomaly_weights.append(anom["duration_frames"])
        iou_per_anomaly.append((anom["id"], iou_a))
        if verbose:
            print(f"    │   Anomalia {anom['id']}: IoU={iou_a:.3f}")

    total_w = sum(anomaly_weights)
    iou_total = (
        sum(v * w for v, w in zip(anomaly_ious, anomaly_weights)) / total_w
        if total_w > 0 else 0.0
    )

    if verbose:
        print(f"    │ IoU_total={iou_total:.3f} | {n_clusters} clusters → {len(selected_clusters)} selected")
        print("    └────────────────────────────────────────────────────────┘")

    detailed = {
        "n_anomalies": n_anomalies,
        "n_segments": len(segments),
        "n_clusters": n_clusters,
        "n_selected": len(selected_clusters),
        "gt_anomalies": gt_anomalies,
        "segments": segments,
        "clusters": clusters,
        "cluster_matches": cluster_matches,
        "anomaly_ious": anomaly_ious,
        "anomaly_weights": anomaly_weights,
        "iou_per_anomaly": iou_per_anomaly,
    }
    return iou_total, selected_clusters, pred_mask, detailed
