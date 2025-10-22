import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pose_extractor import PoseSequence, extract_pose_sequence
from scoring import KEYPOINT_NAMES, ScoreResult, compute_similarity


st.set_page_config(page_title="Muay Thai Pose Scoring", layout="wide")
st.title("Muay Thai Pose Scoring Demo")
st.write(
    "Upload a trainee video and a reference video. The app extracts 12-point "
    "skeletons with YOLO pose, aligns them using DTW, and produces similarity scores."
)


def save_to_temp(upload, suffix: str) -> Path:
    data = upload.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


with st.sidebar:
    st.header("Inference Settings")
    st.markdown("**YOLO pose weights**")
    preset_options = {
        "YOLOv8n Pose (fastest)": "yolov8n-pose.pt",
        "YOLOv8m Pose (balanced)": "yolov8m-pose.pt",
        "YOLOv8x Pose (highest accuracy)": "yolov8x-pose.pt",
        "YOLOv11s Pose (newer balanced)": "yolo11s-pose.pt",
        "YOLOv11x Pose (newer high accuracy)": "yolo11x-pose.pt",
        "Custom path/URL": None,
    }
    preset_choice = st.selectbox("Select weights", list(preset_options.keys()), index=0)
    if preset_choice == "Custom path/URL":
        model_path_input = st.text_input(
            "Custom YOLO pose weights (.pt) or model name",
            help="Enter a local file path, URL, or Ultralytics hub identifier (e.g., 'yolov8n-pose.pt').",
        )
    else:
        model_path_input = preset_options[preset_choice]
        st.caption(f"Using preset weights: `{model_path_input}`")
    confidence = st.slider("Detection confidence", min_value=0.05, max_value=0.9, value=0.3, step=0.05)
    stride = st.number_input("Frame stride", min_value=1, max_value=10, value=1)
    max_frames = st.number_input(
        "Max frames (0 = all)", min_value=0, max_value=2000, value=0, help="Limit to speed up demo."
    )
    alpha = st.slider("Similarity alpha (higher = stricter)", min_value=0.5, max_value=6.0, value=1.0, step=0.1)

col_a, col_b = st.columns(2)
with col_a:
    user_video = st.file_uploader("Trainee video", type=["mp4", "mov", "avi", "mkv"])
with col_b:
    ref_video = st.file_uploader("Reference video", type=["mp4", "mov", "avi", "mkv"])


def ensure_sequence(seq: PoseSequence) -> np.ndarray:
    if not np.isfinite(seq.keypoints).any():
        raise RuntimeError("No pose keypoints detected in the video.")
    return seq.keypoints


score_clicked = st.button("Score poses", type="primary")

if score_clicked:
    if not user_video or not ref_video:
        st.warning("Please upload both videos.")
    elif not model_path_input:
        st.error("Please provide the path to YOLO pose weights.")
    else:
        model_path = Path(model_path_input)
        try:
            user_path = save_to_temp(user_video, ".mp4")
            ref_path = save_to_temp(ref_video, ".mp4")

            seq_user = extract_pose_sequence(
                user_path,
                model_path,
                conf=float(confidence),
                stride=int(stride),
                max_frames=int(max_frames) if max_frames > 0 else None,
            )
            seq_ref = extract_pose_sequence(
                ref_path,
                model_path,
                conf=float(confidence),
                stride=int(stride),
                max_frames=int(max_frames) if max_frames > 0 else None,
            )

            user_seq = ensure_sequence(seq_user)
            ref_seq = ensure_sequence(seq_ref)

            result: ScoreResult = compute_similarity(user_seq, ref_seq, alpha=float(alpha))

            st.success(f"Average similarity: {result.avg_similarity:.2f}% (DTW α={alpha:.2f})")
            st.write(f"Average DTW distance: {result.avg_distance:.5f}")

            st.subheader("Joint-level similarity")
            joint_rows = [
                {
                    "Joint": name,
                    "DTW distance": float(dist),
                    "Similarity (%)": float(sim),
                }
                for name, dist, sim in zip(
                    KEYPOINT_NAMES, result.joint_distances, result.joint_similarities
                )
            ]
            st.table(joint_rows)

            st.subheader("Per-frame similarity (trainee timeline)")
            frame_scores = result.per_frame_similarity
            valid_mask = ~np.isnan(frame_scores)
            if valid_mask.any():
                frame_numbers = np.nonzero(valid_mask)[0]
                df_scores = pd.DataFrame(
                    {"frame": frame_numbers, "similarity": frame_scores[valid_mask]}
                ).set_index("frame")
                st.line_chart(df_scores)
            else:
                st.info("No valid per-frame scores (insufficient alignment).")

        except Exception as exc:
            st.error(f"Error while scoring poses: {exc}")
        finally:
            try:
                if "user_path" in locals() and user_path.exists():
                    user_path.unlink()
                if "ref_path" in locals() and ref_path.exists():
                    ref_path.unlink()
            except Exception:
                pass
