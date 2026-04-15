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

    # Eye Aspect Ratio (how open the eyes are)
    def ear(top_idx, bottom_idx, inner_idx, outer_idx):
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        v_dist = math.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
        h_dist = math.sqrt((inner.x - outer.x)**2 + (inner.y - outer.y)**2)
        return v_dist / h_dist if h_dist > 1e-6 else 0.3

    left_ear = ear(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER)
    right_ear = ear(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER)

    # Absolute iris position in frame (normalized 0-1)
    avg_iris_abs_x = (left_iris.x + right_iris.x) / 2
    avg_iris_abs_y = (left_iris.y + right_iris.y) / 2

    # Nose-eye offset: nose tip relative to midpoint between eyes
    nose = landmarks[NOSE_TIP]
    eye_mid_x = (landmarks[LEFT_EYE_INNER_CORNER].x + landmarks[RIGHT_EYE_INNER_CORNER].x) / 2
    eye_mid_y = (landmarks[LEFT_EYE_INNER_CORNER].y + landmarks[RIGHT_EYE_INNER_CORNER].y) / 2
    nose_off_x = nose.x - eye_mid_x
    nose_off_y = nose.y - eye_mid_y

    # Iris-to-nose vectors (encodes gaze relative to face structure)
    iris_nose_lx = left_iris.x - nose.x
    iris_nose_ly = left_iris.y - nose.y
    iris_nose_rx = right_iris.x - nose.x
    iris_nose_ry = right_iris.y - nose.y

    # Face tilt: angle of inter-eye line relative to horizontal
    le_inner = landmarks[LEFT_EYE_INNER_CORNER]
    re_inner = landmarks[RIGHT_EYE_INNER_CORNER]
    face_tilt = math.atan2(le_inner.y - re_inner.y, le_inner.x - re_inner.x)

    return lx, ly, rx, ry, head_yaw, head_pitch, ipd, left_ear, right_ear, avg_iris_abs_x, avg_iris_abs_y, nose_off_x, nose_off_y, iris_nose_lx, iris_nose_ly, iris_nose_rx, iris_nose_ry, face_tilt


# ============================================================
# Calibration
# ============================================================

def build_feature_matrix_x(points: list[tuple[float, ...]]) -> np.ndarray:
    """Features for predicting screen X."""
    pts = np.array(points)
    n = len(pts)
    lx = pts[:, 0]
    rx = pts[:, 2]
    avg_x = (lx + rx) / 2
    diff_x = lx - rx
    hx = pts[:, 4]
    ipd = pts[:, 6]
    l_ear = pts[:, 7]
    r_ear = pts[:, 8]
    abs_x = pts[:, 9]  # absolute iris X in frame
    nose_ox = pts[:, 11]
    in_lx = pts[:, 13]  # iris-to-nose left x
    in_rx = pts[:, 15]  # iris-to-nose right x
    face_tilt = pts[:, 17]
    return np.column_stack([np.ones(n), avg_x, diff_x, hx, ipd, l_ear, r_ear, abs_x, nose_ox, in_lx, in_rx, face_tilt])


def build_feature_matrix_y(points: list[tuple[float, ...]]) -> np.ndarray:
    """Features for predicting screen Y."""
    pts = np.array(points)
    n = len(pts)
    ly = pts[:, 1]
    ry = pts[:, 3]
    avg_y = (ly + ry) / 2
    diff_y = ly - ry
    hy = pts[:, 5]
    ipd = pts[:, 6]
    l_ear = pts[:, 7]
    r_ear = pts[:, 8]
    abs_y = pts[:, 10]  # absolute iris Y in frame
    nose_oy = pts[:, 12]
    in_ly = pts[:, 14]  # iris-to-nose left y
    in_ry = pts[:, 16]  # iris-to-nose right y
    face_tilt = pts[:, 17]
    return np.column_stack([np.ones(n), avg_y, diff_y, hy, ipd, l_ear, r_ear, abs_y, nose_oy, in_ly, in_ry, face_tilt])


RIDGE_ALPHA_X = 0.8
RIDGE_ALPHA_Y = 0.5

def _ridge_fit(A, y, alpha):
    ATA = A.T @ A
    reg = alpha * np.eye(ATA.shape[0])
    return np.linalg.solve(ATA + reg, A.T @ y)


OUTLIER_THRESHOLD = 2.0  # remove points with residual > threshold * median residual

def _fit_normalized(gaze_points, screen_points):
    """Single fit with normalization, returns coeffs and norms."""
    Ax = build_feature_matrix_x(gaze_points)
    Ay = build_feature_matrix_y(gaze_points)
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
    coeffs_x = _ridge_fit(Ax_norm, screen[:, 0], RIDGE_ALPHA_X)
    coeffs_y = _ridge_fit(Ay_norm, screen[:, 1], RIDGE_ALPHA_Y)
    return coeffs_x, coeffs_y, (mean_x, std_x), (mean_y, std_y)


def fit_calibration(gaze_points, screen_points):
    cur_gaze = list(gaze_points)
    cur_screen = list(screen_points)

    for _ in range(3):  # iterative outlier rejection
        coeffs_x, coeffs_y, norm_x, norm_y = _fit_normalized(cur_gaze, cur_screen)

        # Compute residuals
        screen_arr = np.array(cur_screen)
        residuals = []
        for i, gp in enumerate(cur_gaze):
            px, py = calibration_predict(gp, coeffs_x, coeffs_y, norm_x, norm_y)
            err = math.sqrt((px - screen_arr[i, 0])**2 + (py - screen_arr[i, 1])**2)
            residuals.append(err)
        residuals = np.array(residuals)
        median_res = np.median(residuals)

        mask = residuals <= OUTLIER_THRESHOLD * median_res
        if mask.sum() >= MIN_CALIBRATION_POINTS:
            cur_gaze = [gp for gp, m in zip(cur_gaze, mask) if m]
            cur_screen = [sp for sp, m in zip(cur_screen, mask) if m]
        else:
            break

    coeffs_x, coeffs_y, norm_x, norm_y = _fit_normalized(cur_gaze, cur_screen)
    return coeffs_x, coeffs_y, norm_x, norm_y


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
