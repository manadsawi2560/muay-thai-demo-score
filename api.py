import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pose_extractor import extract_pose_sequence
from scoring import KEYPOINT_NAMES, ScoreResult, compute_similarity


app = FastAPI(title="Muay Thai Pose Scoring API")


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, tmp)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


def _summarise_joints(result: ScoreResult):
    joint_sims = result.joint_similarities
    valid_mask = np.isfinite(joint_sims)
    if not valid_mask.any():
        return None, None
    highest_idx = int(np.nanargmax(joint_sims))
    lowest_idx = int(np.nanargmin(joint_sims))
    highest = {
        "joint": KEYPOINT_NAMES[highest_idx],
        "similarity": float(joint_sims[highest_idx]),
        "distance": float(result.joint_distances[highest_idx]),
    }
    lowest = {
        "joint": KEYPOINT_NAMES[lowest_idx],
        "similarity": float(joint_sims[lowest_idx]),
        "distance": float(result.joint_distances[lowest_idx]),
    }
    return highest, lowest


@app.post("/score")
async def score_pose(
    trainee_video: UploadFile = File(...),
    reference_video: UploadFile = File(...),
    model_path: str = Form("yolov8m-pose.pt"),
    confidence: float = Form(0.3),
    stride: int = Form(1),
    max_frames: Optional[int] = Form(None),
    alpha: float = Form(1.0),
):
    user_path = ref_path = None
    try:
        user_path = _save_upload(trainee_video)
        ref_path = _save_upload(reference_video)

        try:
            seq_user = extract_pose_sequence(
                user_path,
                model_path,
                conf=float(confidence),
                stride=int(stride),
                max_frames=int(max_frames) if max_frames else None,
            )
            seq_ref = extract_pose_sequence(
                ref_path,
                model_path,
                conf=float(confidence),
                stride=int(stride),
                max_frames=int(max_frames) if max_frames else None,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pose extraction failed: {exc}")

        if not np.isfinite(seq_user.keypoints).any():
            raise HTTPException(status_code=422, detail="No keypoints detected in trainee video.")
        if not np.isfinite(seq_ref.keypoints).any():
            raise HTTPException(status_code=422, detail="No keypoints detected in reference video.")

        try:
            result = compute_similarity(seq_user.keypoints, seq_ref.keypoints, alpha=float(alpha))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

        highest, lowest = _summarise_joints(result)
        response = {
            "average_similarity": float(result.avg_similarity),
            "average_distance": float(result.avg_distance),
            "highest_joint_similarity": highest,
            "lowest_joint_similarity": lowest,
        }
        return response
    finally:
        for path in (user_path, ref_path):
            try:
                if path and Path(path).exists():
                    Path(path).unlink()
            except Exception:
                pass
