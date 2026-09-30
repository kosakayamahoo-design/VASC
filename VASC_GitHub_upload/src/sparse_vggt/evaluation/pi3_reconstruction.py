"""Pi3-native global point-map reconstruction protocol.

Pi3 predicts one global point map for every input view.  Its official indoor
reconstruction protocol aligns all valid, paired predictions with one
sequence-level Umeyama Sim(3), then refines that alignment with ICP.  This
module intentionally does not reuse VGGT's depth/point criterion.
"""

from __future__ import annotations

import numpy as np


PROTOCOL_NAME = "pi3_global_points_umeyama_sim3_icp_v1"


def _validate_maps(pred_maps, gt_maps, valid_masks):
    if not (len(pred_maps) == len(gt_maps) == len(valid_masks)):
        raise ValueError("Pi3 prediction, ground-truth, and mask counts differ")
    if not pred_maps:
        raise ValueError("Pi3 reconstruction requires at least one point map")

    validated = []
    for pred, gt, mask in zip(pred_maps, gt_maps, valid_masks):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        mask = np.asarray(mask, dtype=bool)
        if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
            raise ValueError("Pi3 point maps must have matching [H,W,3] shapes")
        if mask.shape != pred.shape[:2]:
            raise ValueError("Pi3 valid mask must match point-map spatial shape")
        finite = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
        validated.append((pred, gt, mask & finite))
    return validated


def estimate_umeyama_similarity(pred_maps, gt_maps, valid_masks):
    """Estimate the sequence-level Sim(3) mapping Pi3 predictions to GT.

    Sufficient statistics are accumulated per view so native-length sequences
    do not require concatenating every dense point correspondence.
    """

    views = _validate_maps(pred_maps, gt_maps, valid_masks)
    count = 0
    pred_sum = np.zeros(3, dtype=np.float64)
    gt_sum = np.zeros(3, dtype=np.float64)
    for pred, gt, mask in views:
        pred_valid = pred[mask].astype(np.float64, copy=False)
        gt_valid = gt[mask].astype(np.float64, copy=False)
        count += len(pred_valid)
        pred_sum += pred_valid.sum(axis=0)
        gt_sum += gt_valid.sum(axis=0)
    if count < 3:
        raise ValueError("Pi3 Umeyama alignment requires at least three points")

    pred_mean = pred_sum / count
    gt_mean = gt_sum / count
    covariance = np.zeros((3, 3), dtype=np.float64)
    pred_variance_sum = 0.0
    for pred, gt, mask in views:
        pred_zero = pred[mask].astype(np.float64, copy=False) - pred_mean
        gt_zero = gt[mask].astype(np.float64, copy=False) - gt_mean
        covariance += gt_zero.T @ pred_zero
        pred_variance_sum += float(np.sum(pred_zero * pred_zero))
    covariance /= count
    pred_variance = pred_variance_sum / count
    if pred_variance < 1e-12:
        raise ValueError("Cannot align a degenerate Pi3 point cloud")

    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        reflection[-1, -1] = -1.0
    rotation = u @ reflection @ vt
    scale = float(
        np.sum(singular_values * np.diag(reflection)) / pred_variance
    )
    translation = gt_mean - scale * (rotation @ pred_mean)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "correspondences": int(count),
    }


def apply_similarity(points, similarity):
    points = np.asarray(points)
    return (
        float(similarity["scale"])
        * (points @ np.asarray(similarity["rotation"]).T)
        + np.asarray(similarity["translation"])
    )


def sample_aligned_point_clouds(
    pred_maps,
    gt_maps,
    valid_masks,
    similarity,
    *,
    max_points,
    seed,
):
    """Apply Sim(3) and select one paired, deterministic metric sample."""

    views = _validate_maps(pred_maps, gt_maps, valid_masks)
    valid_indices = [np.flatnonzero(mask.reshape(-1)) for _, _, mask in views]
    counts = np.asarray([len(indices) for indices in valid_indices], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Pi3 reconstruction has no valid point correspondences")
    if max_points is not None and max_points < 0:
        raise ValueError("max_points must be non-negative or None")

    sample_size = total if not max_points else min(total, int(max_points))
    if sample_size == total:
        chosen = np.arange(total, dtype=np.int64)
    else:
        chosen = np.sort(
            np.random.default_rng(seed).choice(
                total, sample_size, replace=False
            )
        )

    pred_chunks = []
    gt_chunks = []
    offset = 0
    for (pred, gt, _), indices, count in zip(views, valid_indices, counts):
        stop = offset + int(count)
        begin_in_sample = np.searchsorted(chosen, offset, side="left")
        end_in_sample = np.searchsorted(chosen, stop, side="left")
        local = chosen[begin_in_sample:end_in_sample] - offset
        if len(local):
            flat_indices = indices[local]
            pred_selected = pred.reshape(-1, 3)[flat_indices]
            gt_selected = gt.reshape(-1, 3)[flat_indices]
            pred_chunks.append(apply_similarity(pred_selected, similarity))
            gt_chunks.append(gt_selected.astype(np.float64, copy=False))
        offset = stop

    return (
        np.concatenate(pred_chunks, axis=0),
        np.concatenate(gt_chunks, axis=0),
        total,
    )
