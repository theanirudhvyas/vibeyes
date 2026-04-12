"""Tests for detector (orchestrator) module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vibeyes import EyeData, FaceData, GazeRatio, Point, WindowInfo
from vibeyes.detector import Detector


def _make_mock_detector(
    face_data=None,
    gaze_ratio=None,
    screen_point=None,
    windows=None,
    is_calibrated=True,
):
    """Create a Detector with mocked dependencies."""
    face_tracker = MagicMock()
    face_tracker.detect.return_value = face_data

    gaze_estimator = MagicMock()
    gaze_estimator.estimate.return_value = gaze_ratio

    calibration = MagicMock()
    calibration.is_calibrated = is_calibrated
    calibration.predict.return_value = screen_point

    window_tracker = MagicMock()
    window_tracker.return_value = windows or []

    return Detector(
        face_tracker=face_tracker,
        gaze_estimator=gaze_estimator,
        calibration=calibration,
        get_windows=window_tracker,
    )


class TestDetector:
    """Tests for gaze-to-window detection orchestrator."""

    def test_detect_returns_window_info(self):
        """Full pipeline should return the window the user is looking at."""
        window = WindowInfo("VS Code", "Code", (0, 0, 960, 600), 100, 0)
        detector = _make_mock_detector(
            face_data=FaceData(
                left_eye=EyeData(
                    Point(0.5, 0.5), Point(0.4, 0.5), Point(0.6, 0.5),
                    Point(0.5, 0.45), Point(0.5, 0.55),
                ),
                right_eye=EyeData(
                    Point(0.3, 0.5), Point(0.2, 0.5), Point(0.4, 0.5),
                    Point(0.3, 0.45), Point(0.3, 0.55),
                ),
            ),
            gaze_ratio=GazeRatio(0.5, 0.5),
            screen_point=Point(480, 300),
            windows=[window],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result is not None
        assert result.app_name == "Code"

    def test_no_face_returns_none(self):
        """When no face is detected, detect() should return None."""
        detector = _make_mock_detector(face_data=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result is None

    def test_gaze_outside_all_windows_returns_none(self):
        """When gaze maps outside all windows, detect() should return None."""
        window = WindowInfo("VS Code", "Code", (0, 0, 960, 600), 100, 0)
        detector = _make_mock_detector(
            face_data=FaceData(
                left_eye=EyeData(
                    Point(0.5, 0.5), Point(0.4, 0.5), Point(0.6, 0.5),
                    Point(0.5, 0.45), Point(0.5, 0.55),
                ),
                right_eye=EyeData(
                    Point(0.3, 0.5), Point(0.2, 0.5), Point(0.4, 0.5),
                    Point(0.3, 0.45), Point(0.3, 0.55),
                ),
            ),
            gaze_ratio=GazeRatio(0.9, 0.9),
            screen_point=Point(5000, 5000),  # way off screen
            windows=[window],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result is None

    def test_not_calibrated_raises(self):
        """detect() should raise RuntimeError when calibration is not done."""
        detector = _make_mock_detector(
            face_data=FaceData(
                left_eye=EyeData(
                    Point(0.5, 0.5), Point(0.4, 0.5), Point(0.6, 0.5),
                    Point(0.5, 0.45), Point(0.5, 0.55),
                ),
                right_eye=EyeData(
                    Point(0.3, 0.5), Point(0.2, 0.5), Point(0.4, 0.5),
                    Point(0.3, 0.45), Point(0.3, 0.55),
                ),
            ),
            is_calibrated=False,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not calibrated"):
            detector.detect(frame)

    def test_pipeline_call_order(self):
        """Detector should call face_tracker -> gaze_estimator -> calibration -> hit_test."""
        face_data = FaceData(
            left_eye=EyeData(
                Point(0.5, 0.5), Point(0.4, 0.5), Point(0.6, 0.5),
                Point(0.5, 0.45), Point(0.5, 0.55),
            ),
            right_eye=EyeData(
                Point(0.3, 0.5), Point(0.2, 0.5), Point(0.4, 0.5),
                Point(0.3, 0.45), Point(0.3, 0.55),
            ),
        )
        window = WindowInfo("VS Code", "Code", (0, 0, 1920, 1080), 100, 0)
        detector = _make_mock_detector(
            face_data=face_data,
            gaze_ratio=GazeRatio(0.5, 0.5),
            screen_point=Point(960, 540),
            windows=[window],
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)

        detector.face_tracker.detect.assert_called_once_with(frame)
        detector.gaze_estimator.estimate.assert_called_once_with(face_data)
        detector.calibration.predict.assert_called_once()
