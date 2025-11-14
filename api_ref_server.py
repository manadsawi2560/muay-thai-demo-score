import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pose_extractor import extract_pose_sequence
from scoring import KEYPOINT_NAMES, ScoreResult, compute_similarity


# ---------------------------------------------------------
#  Config & App setup
# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
REFERENCE_DIR = BASE_DIR / "reference_videos"

UPLOAD_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Muay Thai Pose Scoring API (reference on server)")

# เปิด CORS ให้ Unity/WebGL ยิงข้าม origin ได้ตอน dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production ค่อยล็อก origin ทีหลัง
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------
def _save_upload(upload: UploadFile) -> Path:
    """Save uploaded video to a temporary file and return its path."""
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
    """
    สรุป joint ที่ similarity สูงสุด / ต่ำสุด
    ใช้โค้ดเดิมของคุณได้เลย
    """
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


# map ชื่อ reference_name ที่ Unity ส่งมา → path ไฟล์บน server
REFERENCE_VIDEOS: Dict[str, Path] = {
    # ปรับชื่อ key และไฟล์ให้ตรงกับที่คุณมีจริง ๆ
    "R_jab_1": REFERENCE_DIR / "R_jab_ref1.mp4",
    "R_jab_2": REFERENCE_DIR / "R_jab_ref2.mp4",
    # ตัวอย่าง
    # "L_jab_1": REFERENCE_DIR / "L_jab_ref1.mp4",
    # "R_kick_1": REFERENCE_DIR / "R_kick_ref1.mp4",
}


def _get_reference_path(reference_name: str) -> Path:
    """คืน path ของ reference video จากชื่อ ถ้าไม่มีให้ raise HTTPException 400"""
    path = REFERENCE_VIDEOS.get(reference_name)
    if path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown reference_name '{reference_name}'. "
                   f"Available: {list(REFERENCE_VIDEOS.keys())}",
        )
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Reference video not found on server: {path}",
        )
    return path


# ---------------------------------------------------------
#  Main endpoint: ใช้ reference_name ฝั่ง server
# ---------------------------------------------------------
@app.post("/score")
async def score_pose_with_server_reference(
    trainee_video: UploadFile = File(...),
    # ส่งจาก Unity เป็น string เช่น "R_jab_1"
    reference_name: str = Form("R_jab_1"),
    model_path: str = Form("yolov8m-pose.pt"),
    confidence: float = Form(0.3),
    stride: int = Form(1),
    max_frames: Optional[int] = Form(None),
    alpha: float = Form(1.0),
):
    """
    Endpoint เวอร์ชันใหม่:
    - รับเฉพาะ trainee_video จาก Unity
    - reference video อยู่ฝั่ง server เลือกจาก reference_name
    """

    user_path: Optional[Path] = None

    try:
        # 1) เซฟ trainee video ลงไฟล์ชั่วคราว
        user_path = _save_upload(trainee_video)

        # 2) หา reference video จาก reference_name
        ref_path = _get_reference_path(reference_name)

        # 3) Extract pose sequence ของทั้งสอง
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

        # 4) เช็คว่ามี keypoints จริงไหม
        if not np.isfinite(seq_user.keypoints).any():
            raise HTTPException(status_code=422, detail="No keypoints detected in trainee video.")
        if not np.isfinite(seq_ref.keypoints).any():
            raise HTTPException(status_code=422, detail="No keypoints detected in reference video.")

        # 5) คำนวณ similarity (ใช้ compute_similarity เดิม)
        try:
            result = compute_similarity(seq_user.keypoints, seq_ref.keypoints, alpha=float(alpha))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")

        # 6) สรุป joint ที่ดีที่สุด/แย่ที่สุด
        highest, lowest = _summarise_joints(result)

        # 7) สร้าง response
        response = {
            "average_similarity": float(result.avg_similarity),
            "average_distance": float(result.avg_distance),
            "highest_joint_similarity": highest,
            "lowest_joint_similarity": lowest,
            # optional: ถ้าอยากใช้ต่อใน Unity ภายหลัง สามารถส่งเพิ่มได้
            # "per_frame_similarity": result.per_frame_similarity.tolist(),
            # "per_frame_distance": result.per_frame_distance.tolist(),
            "reference_name": reference_name,
        }
        return response

    finally:
        # ลบไฟล์ชั่วคราวของ trainee ออก (reference video ไม่ลบ)
        try:
            if user_path and user_path.exists():
                user_path.unlink()
        except Exception:
            pass
