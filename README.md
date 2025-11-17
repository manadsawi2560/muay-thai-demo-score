# Muay Thai Pose Scoring Demo

This project compares a trainee’s Muay Thai pose video to a reference video using YOLO pose detection and DTW-based scoring. It includes:

- **Streamlit demo** (`streamlit_app.py`) for interactive experimentation.
- **FastAPI service** (`api.py`) for programmatic access.

## Installation

Create a virtual environment and install the dependencies:

```bash
pip install streamlit fastapi uvicorn[standard] ultralytics opencv-python pandas numpy
```

> Place the YOLO pose weights you plan to use (for example `yolov8m-pose.pt`, `yolo11x-pose.pt`) in the project folder or supply an absolute path when running the app/API.

## Streamlit Demo

```bash
streamlit run streamlit_app.py
```

1. Upload the trainee and reference videos.
2. Choose a YOLO pose model in the sidebar (preset or custom path).
3. Click **Score poses** to view:
   - Average similarity (%)
   - Joint-level DTW distances/similarities
   - Per-frame similarity chart

## REST API

Start the API with:
ใช่งานสำหรับ unity 
```bash
uvicorn api_ref_server.py --reload
```

Send a `POST /score` request with two video files:

```bash
curl -X POST http://localhost:8000/score \
  -F "trainee_video=@/path/to/trainee.mp4" \
  -F "reference_video=@/path/to/reference.mp4" \
  -F "model_path=yolov8m-pose.pt" \
  -F "confidence=0.3" \
  -F "stride=1" \
  -F "alpha=1.0"
```
example
```bash
curl -X POST http://localhost:8000/score \
  -F "trainee_video=@/home/manadsawi/project/muay-thai-demo/data/videos/R_jabR1 (3).mp4" \
  -F "reference_video=@/home/manadsawi/project/muay-thai-demo/data/videos/R_jabR1 (4).mp4" \
  -F "model_path=yolov8m-pose.pt" \
  -F "confidence=0.3" \
  -F "stride=1" \
  -F "alpha=1.0"
```

Response example:

```json
{
  "average_similarity": 78.42,
  "average_distance": 0.12345,
  "highest_joint_similarity": {
    "joint": "R_Wrist",
    "similarity": 93.12,
    "distance": 0.045
  },
  "lowest_joint_similarity": {
    "joint": "L_Knee",
    "similarity": 54.87,
    "distance": 0.234
  }
}
```

Response includes average similarity/distance plus the highest and lowest scoring joints.

Optional form fields:

- `max_frames` – limit processed frames (omit or set to empty for all frames).
- `confidence` – YOLO detection confidence (default 0.3).
- `stride` – process every n-th frame (default 1).
- `alpha` – DTW similarity sensitivity (default 1.0).

## Tips

- Larger models usually improve accuracy but require more compute; adjust the sidebar presets accordingly.
- Increase `stride` or set `max_frames` when you need quicker feedback during experiments.
- Ensure videos contain a single athlete centred in frame; multi-person clips can reduce detector reliability.
