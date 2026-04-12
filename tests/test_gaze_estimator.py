"""Tests for gaze_estimator module."""

import pytest

from vibeyes import EyeData, FaceData, GazeRatio, Point
from vibeyes.gaze_estimator import GazeEstimator


class TestGazeEstimator:
    """Tests for gaze ratio computation from eye landmarks."""

    def test_center_gaze_returns_midpoint(self, mock_face_data):
        """Iris centered between corners should give gaze ratio near (0.5, 0.5)."""
        estimator = GazeEstimator()
        ratio = estimator.estimate(mock_face_data)
        assert isinstance(ratio, GazeRatio)
        assert abs(ratio.x - 0.5) < 0.05
        assert abs(ratio.y - 0.5) < 0.05

    def test_looking_left_gives_low_x(self, mock_face_data_looking_left):
        """Iris near inner corner should give low x ratio."""
        estimator = GazeEstimator()
        ratio = estimator.estimate(mock_face_data_looking_left)
        assert ratio.x < 0.2

    def test_looking_right_gives_high_x(self, mock_face_data_looking_right):
        """Iris near outer corner should give high x ratio."""
        estimator = GazeEstimator()
        ratio = estimator.estimate(mock_face_data_looking_right)
        assert ratio.x > 0.8

    def test_gaze_ratio_clamped_to_01(self):
        """Gaze ratio should be clamped to [0, 1] even with extreme iris positions."""
        # Iris past the outer corner
        face_data = FaceData(
            left_eye=EyeData(
                iris_center=Point(0.7, 0.5),
                inner_corner=Point(0.4, 0.5),
                outer_corner=Point(0.6, 0.5),
                top=Point(0.5, 0.45),
                bottom=Point(0.5, 0.55),
            ),
            right_eye=EyeData(
                iris_center=Point(0.5, 0.5),
                inner_corner=Point(0.2, 0.5),
                outer_corner=Point(0.4, 0.5),
                top=Point(0.3, 0.45),
                bottom=Point(0.3, 0.55),
            ),
        )
        estimator = GazeEstimator()
        ratio = estimator.estimate(face_data)
        assert 0.0 <= ratio.x <= 1.0
        assert 0.0 <= ratio.y <= 1.0

    def test_averages_both_eyes(self):
        """Gaze should average left and right eye ratios."""
        # Left eye: iris at 25% from inner to outer
        # Right eye: iris at 75% from inner to outer
        # Average should be ~50%
        face_data = FaceData(
            left_eye=EyeData(
                iris_center=Point(0.45, 0.5),
                inner_corner=Point(0.4, 0.5),
                outer_corner=Point(0.6, 0.5),
                top=Point(0.5, 0.45),
                bottom=Point(0.5, 0.55),
            ),
            right_eye=EyeData(
                iris_center=Point(0.35, 0.5),
                inner_corner=Point(0.2, 0.5),
                outer_corner=Point(0.4, 0.5),
                top=Point(0.3, 0.45),
                bottom=Point(0.3, 0.55),
            ),
        )
        estimator = GazeEstimator()
        ratio = estimator.estimate(face_data)
        assert abs(ratio.x - 0.5) < 0.1

    def test_smoothing_reduces_jitter(self):
        """With smoothing enabled, rapid changes in gaze should be dampened."""
        estimator = GazeEstimator(smoothing_factor=0.3)

        # First reading: center
        center_face = FaceData(
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
        )
        r1 = estimator.estimate(center_face)

        # Second reading: suddenly far right
        right_face = FaceData(
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
        )
        r2 = estimator.estimate(right_face)

        # Smoothed result should be between center (0.5) and far right (0.95)
        # but closer to center due to smoothing
        assert r2.x < 0.95, "Smoothing should prevent jumping to extreme"
        assert r2.x > 0.5, "Smoothing should still move toward new value"

    def test_no_smoothing_gives_raw_value(self):
        """With smoothing_factor=1.0, output should equal raw gaze ratio."""
        estimator = GazeEstimator(smoothing_factor=1.0)
        center_face = FaceData(
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
        )
        ratio = estimator.estimate(center_face)
        assert abs(ratio.x - 0.5) < 0.05
        assert abs(ratio.y - 0.5) < 0.05
