"""Gaze pipeline for autoresearch replay. THIS FILE IS AGENT-EDITABLE.

The agent modifies this file to improve avg_error_px.
It must implement two functions:
  - replay_calibration(frames_dir, calibration_clicks) -> state
  - predict(frame, state) -> (screen_x, screen_y)
"""

import math
import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions


# ============================================================
# Constants
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")

# MediaPipe landmark indices
LEFT_IRIS_CENTER = 473
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_IRIS_CENTER = 468
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

NOSE_TIP = 1

POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
], dtype=np.float64)

# Smoothing parameters
GAZE_SMOOTHING_FACTOR = 0.3
MEDIAN_WINDOW = 7
DETECTOR_EMA_FACTOR = 0.4

MIN_CALIBRATION_POINTS = 6

# Screen bounds (detected dynamically, fallback values)
try:
    import Quartz
    _bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    SCREEN_W = float(_bounds.size.width)
    SCREEN_H = float(_bounds.size.height)
except ImportError:
    SCREEN_W = 3360.0
    SCREEN_H = 2100.0


# ============================================================
# Pipeline state
# ============================================================

@dataclass
class PipelineState:
    landmarker: FaceLandmarker
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    norm_x: tuple = None  # (mean, std) for X features
    norm_y: tuple = None  # (mean, std) for Y features


# ============================================================
# Feature extraction
# ============================================================

def extract_gaze_features(frame: np.ndarray, landmarker: FaceLandmarker) -> tuple[float, ...] | None:
    h, w = frame.shape[:2]
    rgb = frame[:, :, ::-1]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    def eye_ratio(iris_idx, inner_idx, outer_idx, top_idx, bottom_idx):
        iris = landmarks[iris_idx]
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]

        left_x = min(inner.x, outer.x)
        right_x = max(inner.x, outer.x)
        x_range = right_x - left_x
        x_ratio = (iris.x - left_x) / x_range if x_range > 1e-6 else 0.5

        top_y = min(top.y, bottom.y)
        bottom_y = max(top.y, bottom.y)
        y_range = bottom_y - top_y
        y_ratio = (iris.y - top_y) / y_range if y_range > 1e-6 else 0.5

        return x_ratio, y_ratio

    lx, ly = eye_ratio(LEFT_IRIS_CENTER, LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER,
                        LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
    rx, ry = eye_ratio(RIGHT_IRIS_CENTER, RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER,
                        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

    lx = max(0.0, min(1.0, lx))
    ly = max(0.0, min(1.0, ly))
    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))

    # Head pose via solvePnP
    image_points = np.array([
        [landmarks[idx].x * w, landmarks[idx].y * h]
        for idx in POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = w
    cx, cy = w / 2, h / 2
    cam_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        head_yaw, head_pitch = 0.0, 0.0
    else:
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        head_pitch, head_yaw = angles[0], angles[1]

    # Inter-pupillary distance (proxy for distance from camera)
    left_iris = landmarks[LEFT_IRIS_CENTER]
    right_iris = landmarks[RIGHT_IRIS_CENTER]
    ipd = math.sqrt((left_iris.x - right_iris.x)**2 + (left_iris.y - right_iris.y)**2)

    return lx, ly, rx, ry, head_yaw, head_pitch, ipd


# ============================================================
# Calibration
# ============================================================

def build_feature_matrix_x(points: list[tuple[float, ...]]) -> np.ndarray:
    """Features for predicting screen X: horizontal iris ratios + head yaw + IPD."""
    pts = np.array(points)
    n = len(pts)
    lx = pts[:, 0]
    rx = pts[:, 2]
    hx = pts[:, 4]  # head yaw
    ipd = pts[:, 6]
    return np.column_stack([np.ones(n), lx, rx, hx, ipd])


def build_feature_matrix_y(points: list[tuple[float, ...]]) -> np.ndarray:
    """Features for predicting screen Y: vertical iris ratios + head pitch + IPD."""
    pts = np.array(points)
    n = len(pts)
    ly = pts[:, 1]
    ry = pts[:, 3]
    hy = pts[:, 5]  # head pitch
    ipd = pts[:, 6]
    return np.column_stack([np.ones(n), ly, ry, hy, ipd])


RIDGE_ALPHA = 1.3

def _ridge_fit(A, y, alpha):
    ATA = A.T @ A
    reg = alpha * np.eye(ATA.shape[0])
    return np.linalg.solve(ATA + reg, A.T @ y)


def fit_calibration(gaze_points, screen_points):
    Ax = build_feature_matrix_x(gaze_points)
    Ay = build_feature_matrix_y(gaze_points)
    # Z-score normalize each
    mean_x = Ax[:, 1:].mean(axis=0)
    std_x = Ax[:, 1:].std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    Ax_norm = Ax.copy()
    Ax_norm[:, 1:] = (Ax[:, 1:] - mean_x) / std_x

    mean_y = Ay[:, 1:].mean(axis=0)
    std_y = Ay[:, 1:].std(axis=0)
    std_y[std_y < 1e-8] = 1.0
    Ay_norm = Ay.copy()
    Ay_norm[:, 1:] = (Ay[:, 1:] - mean_y) / std_y

    screen = np.array(screen_points)
    coeffs_x = _ridge_fit(Ax_norm, screen[:, 0], RIDGE_ALPHA)
    coeffs_y = _ridge_fit(Ay_norm, screen[:, 1], RIDGE_ALPHA)
    return coeffs_x, coeffs_y, (mean_x, std_x), (mean_y, std_y)


def calibration_predict(gaze_features, coeffs_x, coeffs_y, norm_x, norm_y):
    mean_x, std_x = norm_x
    mean_y, std_y = norm_y
    fx = build_feature_matrix_x([gaze_features])
    fy = build_feature_matrix_y([gaze_features])
    fx[:, 1:] = (fx[:, 1:] - mean_x) / std_x
    fy[:, 1:] = (fy[:, 1:] - mean_y) / std_y
    sx = float((fx @ coeffs_x)[0])
    sy = float((fy @ coeffs_y)[0])
    return sx, sy


# ============================================================
# Smoothing
# ============================================================

def apply_gaze_smoothing(iris_x, iris_y, head_yaw, head_pitch, state):
    if state.prev_gaze_x is None:
        state.prev_gaze_x = iris_x
        state.prev_gaze_y = iris_y
        state.prev_head_yaw = head_yaw
        state.prev_head_pitch = head_pitch
        return iris_x, iris_y, head_yaw, head_pitch

    s = GAZE_SMOOTHING_FACTOR
    sx = s * iris_x + (1 - s) * state.prev_gaze_x
    sy = s * iris_y + (1 - s) * state.prev_gaze_y
    shx = s * head_yaw + (1 - s) * state.prev_head_yaw
    shy = s * head_pitch + (1 - s) * state.prev_head_pitch

    state.prev_gaze_x = sx
    state.prev_gaze_y = sy
    state.prev_head_yaw = shx
    state.prev_head_pitch = shy

    return sx, sy, shx, shy


def apply_screen_smoothing(raw_x, raw_y, state):
    state.history_x.append(raw_x)
    state.history_y.append(raw_y)
    median_x = sorted(state.history_x)[len(state.history_x) // 2]
    median_y = sorted(state.history_y)[len(state.history_y) // 2]

    if state.smoothed_x is None:
        state.smoothed_x = median_x
        state.smoothed_y = median_y
    else:
        s = DETECTOR_EMA_FACTOR
        state.smoothed_x = s * median_x + (1 - s) * state.smoothed_x
        state.smoothed_y = s * median_y + (1 - s) * state.smoothed_y

    return state.smoothed_x, state.smoothed_y


# ============================================================
# Public API (called by prepare.py)
# ============================================================

def replay_calibration(frames_dir: str, calibration_clicks: list[dict]) -> PipelineState:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    gaze_points = []
    screen_points = []

    for click in calibration_clicks:
        frame_path = os.path.join(frames_dir, f"{click['frame_id']}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        features = extract_gaze_features(frame, landmarker)
        if features is None:
            continue

        gaze_points.append(features)
        screen_points.append((click["click_x"], click["click_y"]))

    if len(gaze_points) < MIN_CALIBRATION_POINTS:
        raise ValueError(
            f"Only {len(gaze_points)} usable calibration frames "
            f"(need {MIN_CALIBRATION_POINTS}). Record more data."
        )

    coeffs_x, coeffs_y, norm_x, norm_y = fit_calibration(gaze_points, screen_points)

    return PipelineState(
        landmarker=landmarker,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        norm_x=norm_x,
        norm_y=norm_y,
    )


def predict(frame: np.ndarray, state: PipelineState) -> tuple[float, float]:
    features = extract_gaze_features(frame, state.landmarker)
    if features is None:
        return SCREEN_W / 2, SCREEN_H / 2

    raw_x, raw_y = calibration_predict(
        features, state.coeffs_x, state.coeffs_y,
        state.norm_x, state.norm_y)

    final_x = max(0.0, min(SCREEN_W, raw_x))
    final_y = max(0.0, min(SCREEN_H, raw_y))

    return final_x, final_y
