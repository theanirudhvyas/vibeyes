"""Tests for calibration module."""

import json
import os
import tempfile

import pytest

from vibeyes import GazeRatio, Point
from vibeyes.calibration import Calibration


def _make_9point_calibration():
    """Create a calibration with 9 points mapping gaze ratios to screen coords.

    Simulates a 1920x1080 screen with a linear relationship for simplicity.
    """
    cal = Calibration()
    # 3x3 grid of calibration points
    for gx, sx in [(0.1, 192), (0.5, 960), (0.9, 1728)]:
        for gy, sy in [(0.1, 108), (0.5, 540), (0.9, 972)]:
            cal.add_point(GazeRatio(gx, gy), Point(sx, sy))
    return cal


class TestCalibration:
    """Tests for gaze-to-screen coordinate calibration."""

    def test_add_point_stores_data(self):
        """Adding calibration points should increase the point count."""
        cal = Calibration()
        assert cal.point_count == 0
        cal.add_point(GazeRatio(0.5, 0.5), Point(960, 540))
        assert cal.point_count == 1

    def test_fit_requires_minimum_points(self):
        """Fitting should raise ValueError with fewer than 6 points."""
        cal = Calibration()
        cal.add_point(GazeRatio(0.5, 0.5), Point(960, 540))
        cal.add_point(GazeRatio(0.1, 0.1), Point(192, 108))
        with pytest.raises(ValueError, match="at least 6"):
            cal.fit()

    def test_predict_before_fit_raises(self):
        """Predicting before fitting should raise RuntimeError."""
        cal = Calibration()
        with pytest.raises(RuntimeError, match="not calibrated"):
            cal.predict(GazeRatio(0.5, 0.5))

    def test_predict_calibration_points(self):
        """Predicting at calibration points should be very close to actual screen coords."""
        cal = _make_9point_calibration()
        cal.fit()
        # Test center point
        predicted = cal.predict(GazeRatio(0.5, 0.5))
        assert isinstance(predicted, Point)
        assert abs(predicted.x - 960) < 50
        assert abs(predicted.y - 540) < 50

    def test_predict_interpolated_point(self):
        """Predicting between calibration points should give reasonable interpolation."""
        cal = _make_9point_calibration()
        cal.fit()
        # Point between top-left and center
        predicted = cal.predict(GazeRatio(0.3, 0.3))
        # Expected: roughly (576, 324) for linear mapping
        assert 400 < predicted.x < 750
        assert 200 < predicted.y < 450

    def test_predict_at_corners(self):
        """Predictions at extreme gaze ratios should map to screen edges."""
        cal = _make_9point_calibration()
        cal.fit()
        # Top-left
        tl = cal.predict(GazeRatio(0.1, 0.1))
        assert tl.x < 400
        assert tl.y < 300
        # Bottom-right
        br = cal.predict(GazeRatio(0.9, 0.9))
        assert br.x > 1500
        assert br.y > 700

    def test_save_and_load(self):
        """Calibration should be saveable and loadable from JSON."""
        cal = _make_9point_calibration()
        cal.fit()
        original_prediction = cal.predict(GazeRatio(0.5, 0.5))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            cal.save(path)
            assert os.path.exists(path)

            loaded = Calibration.load(path)
            loaded_prediction = loaded.predict(GazeRatio(0.5, 0.5))

            assert abs(loaded_prediction.x - original_prediction.x) < 1
            assert abs(loaded_prediction.y - original_prediction.y) < 1
        finally:
            os.unlink(path)

    def test_is_calibrated(self):
        """is_calibrated should reflect whether fit() has been called."""
        cal = Calibration()
        assert not cal.is_calibrated
        cal = _make_9point_calibration()
        assert not cal.is_calibrated
        cal.fit()
        assert cal.is_calibrated

    def test_clear_resets(self):
        """clear() should remove all calibration data."""
        cal = _make_9point_calibration()
        cal.fit()
        cal.clear()
        assert cal.point_count == 0
        assert not cal.is_calibrated
