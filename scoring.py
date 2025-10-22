import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


NUM_KEYPOINTS = 12
KEYPOINT_NAMES = [
    "R_Shoulder",
    "L_Shoulder",
    "R_Elbow",
    "L_Elbow",
    "R_Wrist",
    "L_Wrist",
    "R_Hip",
    "L_Hip",
    "R_Knee",
    "L_Knee",
    "R_Ankle",
    "L_Ankle",
]


def interpolate_sequence_nans(seq_xy: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs along the temporal axis for each joint/dimension."""
    T, K, D = seq_xy.shape
    out = seq_xy.copy()
    for k in range(K):
        for d in range(D):
            track = out[:, k, d]
            mask = ~np.isnan(track)
            if mask.sum() == 0:
                continue
            if mask.sum() == 1:
                out[:, k, d] = track[mask][0]
                continue
            x = np.arange(T)
            out[:, k, d] = np.interp(x, x[mask], track[mask])
    return out


def l2_normalize_per_frame(seq_xy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise each frame vector (K*2) to unit length."""
    T, K, D = seq_xy.shape
    flat = seq_xy.reshape(T, K * D)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + eps
    flat = flat / norms
    return flat.reshape(T, K, D)


def dtw_distance_nd(A: np.ndarray, B: np.ndarray) -> float:
    """Classic DTW distance between two multi-dimensional sequences."""
    T1, D = A.shape
    T2 = B.shape[0]
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    acc = np.full((T1 + 1, T2 + 1), np.inf, dtype=float)
    acc[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return float(acc[T1, T2])


def _dtw_frames_path(A_flat: np.ndarray, B_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """DTW on frame-wise flattened features. Returns path and local costs."""
    T1 = A_flat.shape[0]
    T2 = B_flat.shape[0]
    cost = np.linalg.norm(A_flat[:, None, :] - B_flat[None, :, :], axis=2)
    acc = np.full((T1 + 1, T2 + 1), np.inf, dtype=float)
    acc[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    i, j = T1, T2
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]]
        move = int(np.argmin(choices))
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()
    return np.array(path, dtype=int), cost


@dataclass
class ScoreResult:
    avg_distance: float
    avg_similarity: float
    joint_distances: np.ndarray
    joint_similarities: np.ndarray
    per_frame_similarity: np.ndarray
    per_frame_distance: np.ndarray
    path_ij: np.ndarray


def exp_similarity(dist: float, alpha: float) -> float:
    """Convert DTW distance to similarity percentage via exponential decay."""
    return float(100.0 * math.exp(-alpha * dist))


def compute_similarity(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    alpha: float = 1.0,
    joint_names: Sequence[str] = KEYPOINT_NAMES,
) -> ScoreResult:
    """
    seq_a/seq_b: arrays with shape (T, K, 2) in normalised [0,1] coordinates.
    They will be interpolated, L2-normalised per frame, and compared with DTW.
    """
    if seq_a.ndim != 3 or seq_b.ndim != 3:
        raise ValueError("Sequences must have shape (T, K, 2).")
    if seq_a.shape[1:] != seq_b.shape[1:]:
        raise ValueError("Both sequences must share the same K and dimensionality.")

    seq_a = interpolate_sequence_nans(seq_a)
    seq_b = interpolate_sequence_nans(seq_b)

    seq_a = l2_normalize_per_frame(seq_a)
    seq_b = l2_normalize_per_frame(seq_b)

    T1, K, _ = seq_a.shape
    T2 = seq_b.shape[0]

    joint_dists = np.zeros((K,), dtype=float)
    joint_sims = np.zeros((K,), dtype=float)
    for idx in range(K):
        dist = dtw_distance_nd(seq_a[:, idx, :], seq_b[:, idx, :])
        joint_dists[idx] = dist
        joint_sims[idx] = exp_similarity(dist, alpha)

    avg_dist = float(np.nanmean(joint_dists))
    avg_sim = float(np.nanmean(joint_sims))

    path_ij, local_cost = _dtw_frames_path(
        seq_a.reshape(T1, K * 2), seq_b.reshape(T2, K * 2)
    )

    per_frame_sim = np.full((T1,), np.nan, dtype=float)
    per_frame_dist = np.full((T1,), np.nan, dtype=float)
    total_sim = np.zeros((T1,), dtype=float)
    total_dist = np.zeros((T1,), dtype=float)
    counts = np.zeros((T1,), dtype=int)

    for i, j in path_ij:
        d = local_cost[i, j]
        s = exp_similarity(d, alpha)
        total_dist[i] += d
        total_sim[i] += s
        counts[i] += 1

    valid = counts > 0
    per_frame_dist[valid] = total_dist[valid] / counts[valid]
    per_frame_sim[valid] = total_sim[valid] / counts[valid]

    return ScoreResult(
        avg_distance=avg_dist,
        avg_similarity=avg_sim,
        joint_distances=joint_dists,
        joint_similarities=joint_sims,
        per_frame_similarity=per_frame_sim,
        per_frame_distance=per_frame_dist,
        path_ij=path_ij,
    )


def yolo_txt_to_sequence(folder: str, num_kpts: int = NUM_KEYPOINTS) -> np.ndarray:
    """Load YOLO keypoint txt files into a (T,K,2) array normalised to [0,1]."""
    import glob
    import os
    import re

    def natural_key(name: str) -> List[object]:
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]

    txts = sorted(glob.glob(os.path.join(folder, "*.txt")), key=natural_key)
    if not txts:
        raise FileNotFoundError(f"No .txt files in {folder}")

    frames: List[np.ndarray] = []
    for path in txts:
        with open(path) as fh:
            line = fh.readline().strip()
        if not line:
            frames.append(np.full((num_kpts, 2), np.nan, dtype=float))
            continue
        values = [float(v) for v in line.split()]
        expect = 5 + num_kpts * 3
        if len(values) < expect:
            frames.append(np.full((num_kpts, 2), np.nan, dtype=float))
            continue
        pts = np.array(values[5 : 5 + num_kpts * 3], dtype=float).reshape(num_kpts, 3)
        xy = pts[:, :2]
        vis = pts[:, 2]
        xy[vis <= 0.0] = np.nan
        frames.append(xy)
    return np.stack(frames, axis=0)
