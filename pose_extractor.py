from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np


COCO_17_TO_CUSTOM_12: Sequence[int] = (
    6,
    5,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
)


@dataclass
class PoseSequence:
    keypoints: np.ndarray  # shape (T, K, 2) in [0,1]
    confidences: np.ndarray  # shape (T, K)
    frame_indices: List[int]
    fps: float


def extract_pose_sequence(
    video_path: Union[str, Path],
    model_path: Union[str, Path],
    conf: float = 0.25,
    imgsz: int = 640,
    stride: int = 1,
    device: Optional[str] = None,
    max_frames: Optional[int] = None,
    num_keypoints: int = 12,
) -> PoseSequence:
    """
    Run YOLO pose on a video and return the normalised 12-point skeleton sequence.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for pose extraction. Install with `pip install ultralytics`."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_source = str(model_path)
    try:
        model = YOLO(model_source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model weights not found or cannot be loaded: {model_source}"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    def _to_numpy(arr):
        if arr is None:
            return None
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        return np.asarray(arr)

    keypoints = []
    confidences = []
    frame_indices: List[int] = []

    index = 0
    collected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index % stride != 0:
            index += 1
            continue

        results = model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )
        kp = np.full((num_keypoints, 2), np.nan, dtype=float)
        kc = np.zeros((num_keypoints,), dtype=float)

        if results and hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            result = results[0]
            xy = _to_numpy(result.keypoints.xy)
            confs = _to_numpy(result.keypoints.conf)
            if xy is not None and len(xy):
                if xy.ndim == 3:
                    scores = None
                    if getattr(result, "boxes", None) is not None:
                        scores = _to_numpy(result.boxes.conf)
                    if scores is None and confs is not None:
                        scores = confs.mean(axis=1)
                    if scores is None:
                        scores = np.zeros((xy.shape[0],), dtype=float)
                    best_idx = int(np.argmax(scores))
                    points = xy[best_idx]
                    vis = confs[best_idx] if confs is not None else None
                else:
                    points = xy
                    vis = confs
                if points.shape[0] != num_keypoints:
                    if (
                        num_keypoints == len(COCO_17_TO_CUSTOM_12)
                        and points.shape[0] >= len(COCO_17_TO_CUSTOM_12)
                    ):
                        idx = np.array(COCO_17_TO_CUSTOM_12, dtype=int)
                        points = points[idx]
                        if vis is not None:
                            vis = vis[idx]
                    elif points.shape[0] >= num_keypoints:
                        points = points[:num_keypoints]
                        if vis is not None:
                            vis = vis[:num_keypoints]
                    else:
                        padded = np.full((num_keypoints, 2), np.nan, dtype=float)
                        padded[: points.shape[0]] = points
                        points = padded
                        if vis is not None:
                            vis_pad = np.zeros((num_keypoints,), dtype=float)
                        vis_pad[: vis.shape[0]] = vis
                        vis = vis_pad
                kp[:, 0] = points[:, 0] / max(width, 1)
                kp[:, 1] = points[:, 1] / max(height, 1)
                if vis is not None:
                    kc = np.asarray(vis, dtype=float)

        keypoints.append(kp)
        confidences.append(kc)
        frame_indices.append(index)
        index += 1
        collected += 1
        if max_frames is not None and collected >= max_frames:
            break

    cap.release()

    if not keypoints:
        raise RuntimeError("No frames processed from video.")

    return PoseSequence(
        keypoints=np.stack(keypoints, axis=0),
        confidences=np.stack(confidences, axis=0),
        frame_indices=frame_indices,
        fps=fps,
    )
