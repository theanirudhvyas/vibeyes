"""Shared test fixtures for VibEyes."""

import os

import numpy as np
import pytest

from vibeyes import EyeData, FaceData, GazeRatio, Point, WindowInfo

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_landmarker.task")


@pytest.fixture
def model_path():
    """Path to the face landmarker model file."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("face_landmarker.task model not found -- run download script first")
    return MODEL_PATH


@pytest.fixture
def blank_frame():
    """A 640x480 blank frame with no face."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_face_data():
    """FaceData with iris centered in both eyes (looking straight ahead)."""
    return FaceData(
        left_eye=EyeData(
            iris_center=Point(0.5, 0.5),
            inner_corner=Point(0.4, 0.5),
            outer_corner=Point(0.6, 0.5),
            top=Point(0.5, 0.45),
            bottom=Point(0.5, 0.55),
        ),
        right_eye=EyeData(
            iris_center=Point(0.3, 0.5),
            inner_corner=Point(0.2, 0.5),
            outer_corner=Point(0.4, 0.5),
            top=Point(0.3, 0.45),
            bottom=Point(0.3, 0.55),
        ),
        nose_tip=Point(0.5, 0.5),
    )


@pytest.fixture
def mock_face_data_looking_left():
    """FaceData with iris shifted to the left (inner corner) side of both eyes."""
    return FaceData(
        left_eye=EyeData(
            iris_center=Point(0.41, 0.5),
            inner_corner=Point(0.4, 0.5),
            outer_corner=Point(0.6, 0.5),
            top=Point(0.5, 0.45),
            bottom=Point(0.5, 0.55),
        ),
        right_eye=EyeData(
            iris_center=Point(0.21, 0.5),
            inner_corner=Point(0.2, 0.5),
            outer_corner=Point(0.4, 0.5),
            top=Point(0.3, 0.45),
            bottom=Point(0.3, 0.55),
        ),
        nose_tip=Point(0.55, 0.5),
    )


@pytest.fixture
def mock_face_data_looking_right():
    """FaceData with iris shifted to the right (outer corner) side of both eyes."""
    return FaceData(
        left_eye=EyeData(
            iris_center=Point(0.59, 0.5),
            inner_corner=Point(0.4, 0.5),
            outer_corner=Point(0.6, 0.5),
            top=Point(0.5, 0.45),
            bottom=Point(0.5, 0.55),
        ),
        right_eye=EyeData(
            iris_center=Point(0.39, 0.5),
            inner_corner=Point(0.2, 0.5),
            outer_corner=Point(0.4, 0.5),
            top=Point(0.3, 0.45),
            bottom=Point(0.3, 0.55),
        ),
        nose_tip=Point(0.45, 0.5),
    )


@pytest.fixture
def sample_windows():
    """List of mock WindowInfo objects for hit testing."""
    return [
        WindowInfo(
            title="main.py - VS Code",
            app_name="Code",
            bounds=(0, 0, 960, 600),
            pid=100,
            layer=0,
        ),
        WindowInfo(
            title="Slack - General",
            app_name="Slack",
            bounds=(960, 0, 960, 600),
            pid=200,
            layer=0,
        ),
        WindowInfo(
            title="Terminal",
            app_name="Terminal",
            bounds=(0, 600, 1920, 480),
            pid=300,
            layer=0,
        ),
    ]
