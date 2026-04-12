"""Tests for face_tracker module."""

import numpy as np
import pytest

from vibeyes import FaceData, EyeData, Point
from vibeyes.face_tracker import FaceTracker


class TestFaceTracker:
    """Tests for FaceTracker landmark detection."""

    def test_no_face_returns_none(self, model_path, blank_frame):
        """Given a frame with no face, detect() should return None."""
        tracker = FaceTracker(model_path)
        result = tracker.detect(blank_frame)
        assert result is None
        tracker.close()

    def test_returns_face_data_type(self, model_path, face_frame):
        """Given a frame with a face, detect() should return a FaceData instance."""
        tracker = FaceTracker(model_path)
        result = tracker.detect(face_frame)
        assert isinstance(result, FaceData)
        tracker.close()

    def test_eye_data_structure(self, model_path, face_frame):
        """FaceData should contain left_eye and right_eye as EyeData instances."""
        tracker = FaceTracker(model_path)
        result = tracker.detect(face_frame)
        assert isinstance(result.left_eye, EyeData)
        assert isinstance(result.right_eye, EyeData)
        tracker.close()

    def test_landmarks_are_points(self, model_path, face_frame):
        """All landmarks in EyeData should be Point instances."""
        tracker = FaceTracker(model_path)
        result = tracker.detect(face_frame)
        for eye in [result.left_eye, result.right_eye]:
            assert isinstance(eye.iris_center, Point)
            assert isinstance(eye.inner_corner, Point)
            assert isinstance(eye.outer_corner, Point)
            assert isinstance(eye.top, Point)
            assert isinstance(eye.bottom, Point)
        tracker.close()

    def test_landmarks_normalized_range(self, model_path, face_frame):
        """All landmark coordinates should be in [0, 1] range (normalized)."""
        tracker = FaceTracker(model_path)
        result = tracker.detect(face_frame)
        for eye in [result.left_eye, result.right_eye]:
            for point in [eye.iris_center, eye.inner_corner, eye.outer_corner, eye.top, eye.bottom]:
                assert 0.0 <= point.x <= 1.0, f"x={point.x} out of range"
                assert 0.0 <= point.y <= 1.0, f"y={point.y} out of range"
        tracker.close()

    def test_iris_between_corners(self, model_path, face_frame):
        """Iris center x should be between inner and outer corner x for each eye."""
        tracker = FaceTracker(model_path)
        result = tracker.detect(face_frame)
        for eye in [result.left_eye, result.right_eye]:
            min_x = min(eye.inner_corner.x, eye.outer_corner.x)
            max_x = max(eye.inner_corner.x, eye.outer_corner.x)
            assert min_x <= eye.iris_center.x <= max_x, (
                f"iris x={eye.iris_center.x} not between corners [{min_x}, {max_x}]"
            )
        tracker.close()

    def test_close_is_idempotent(self, model_path):
        """Calling close() multiple times should not raise."""
        tracker = FaceTracker(model_path)
        tracker.close()
        tracker.close()


@pytest.fixture
def face_frame():
    """Generate a synthetic frame with a simple face-like pattern.

    This creates a basic oval face with dark circles for eyes that MediaPipe
    can detect. If MediaPipe can't detect this synthetic face, tests using
    this fixture will be skipped.
    """
    import cv2

    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # light gray background

    # Draw a skin-colored oval face
    cv2.ellipse(frame, (320, 240), (120, 160), 0, 0, 360, (180, 160, 140), -1)

    # Draw eyes (dark circles)
    cv2.circle(frame, (280, 220), 15, (40, 30, 30), -1)  # left eye
    cv2.circle(frame, (360, 220), 15, (40, 30, 30), -1)  # right eye

    # Draw pupils (darker circles inside eyes)
    cv2.circle(frame, (280, 220), 5, (10, 10, 10), -1)
    cv2.circle(frame, (360, 220), 5, (10, 10, 10), -1)

    # Draw nose
    cv2.line(frame, (320, 230), (315, 265), (150, 130, 120), 2)

    # Draw mouth
    cv2.ellipse(frame, (320, 290), (30, 10), 0, 0, 180, (120, 80, 80), 2)

    return frame
