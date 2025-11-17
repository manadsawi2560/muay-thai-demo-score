# ---------------------------------------------------------
# scoring.py
# ---------------------------------------------------------

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

# ส่วนที่เพิ่มเข้ามาสำหรับการ Map ภาษาไทย
KEYPOINT_NAMES_THAI = [
    "ไหล่ขวา", "ไหล่ซ้าย", "ศอกขวา", "ศอกซ้าย", "ข้อมือขวา", "ข้อมือซ้าย",
    "สะโพกขวา", "สะโพกซ้าย", "เข่าขวา", "เข่าซ้าย", "ข้อเท้าขวา", "ข้อเท้าซ้าย",
]
KEYPOINT_MAPPING_THAI = dict(zip(KEYPOINT_NAMES, KEYPOINT_NAMES_THAI))

# dataclass สำหรับเก็บผลลัพธ์การคำนวณ (จำเป็นต้องมีก่อนถูกเรียกใช้)
@dataclass
class ScoreResult:
    avg_distance: float
    avg_similarity: float
    joint_distances: np.ndarray
    joint_similarities: np.ndarray
    per_frame_similarity: np.ndarray
    per_frame_distance: np.ndarray
    path_ij: np.ndarray


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

# ---------------------------------------------------------
# ส่วนที่เพิ่มเข้ามาเพื่อจัดรูปแบบผลลัพธ์สำหรับ Unity (JSON)
# ---------------------------------------------------------

def _get_joint_data(index: int, result: ScoreResult) -> dict:
    """สร้าง dictionary ข้อมูลของข้อต่อเดียว"""
    joint_name_en = KEYPOINT_NAMES[index]
    return {
        "joint_name_en": joint_name_en,
        "joint_name_th": KEYPOINT_MAPPING_THAI.get(joint_name_en, joint_name_en),
        "similarity_pct": float(round(result.joint_similarities[index], 2)),
        "distance": float(round(result.joint_distances[index], 5)),
    }

def format_result_for_unity(result: ScoreResult, reference_name: str) -> dict:
    """
    จัดโครงสร้างผลลัพธ์ทั้งหมดให้เป็น Dictionary ที่พร้อมส่งเป็น JSON
    โดยมีการสรุปผลสูงสุด/ต่ำสุด และ Map ชื่อข้อต่อให้เป็นภาษาไทย
    """
    joint_sims = result.joint_similarities
    valid_mask = np.isfinite(joint_sims)

    highest_data = {}
    lowest_data = {}

    if valid_mask.any():
        # หา index ของ joint ที่คล้ายและต่างกันที่สุด
        highest_idx = int(np.nanargmax(joint_sims))
        lowest_idx = int(np.nanargmin(joint_sims))
        
        highest_data = _get_joint_data(highest_idx, result)
        lowest_data = _get_joint_data(lowest_idx, result)
    
    # รวมผลลัพธ์ทั้งหมด
    all_joints_data = [
        _get_joint_data(i, result)
        for i in range(len(KEYPOINT_NAMES)) if np.isfinite(result.joint_similarities[i])
    ]

    return {
        "summary": {
            "average_similarity_pct": float(round(result.avg_similarity, 2)),
            "average_distance": float(round(result.avg_distance, 5)),
        },
        "highest_similarity": highest_data,
        "lowest_similarity": lowest_data,
        "all_joints": all_joints_data,
        "reference_name": reference_name, # ชื่อท่าอ้างอิง
    }

# ---------------------------------------------------------