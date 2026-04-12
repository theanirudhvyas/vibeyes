"""Gaze-to-window detection orchestrator."""

import numpy as np

from vibeyes import Point, WindowInfo
from vibeyes.window_tracker import hit_test


class Detector:
    """Orchestrates the gaze-to-window detection pipeline.

    Pipeline: frame -> face_tracker -> gaze_estimator -> calibration -> window hit_test
    """

    def __init__(self, face_tracker, gaze_estimator, calibration, get_windows):
        self.face_tracker = face_tracker
        self.gaze_estimator = gaze_estimator
        self.calibration = calibration
        self._get_windows = get_windows
        self.last_screen_point: Point | None = None

    def detect(self, frame: np.ndarray) -> WindowInfo | None:
        """Run the full detection pipeline on a single frame.

        Returns the WindowInfo the user is looking at, or None.
        Raises RuntimeError if calibration has not been done.
        """
        # Step 1: Detect face landmarks
        face_data = self.face_tracker.detect(frame)
        if face_data is None:
            return None

        # Step 2: Check calibration
        if not self.calibration.is_calibrated:
            raise RuntimeError("Detector not calibrated -- run calibration first")

        # Step 3: Estimate gaze ratio
        gaze_ratio = self.gaze_estimator.estimate(face_data)

        # Step 4: Map to screen coordinates
        screen_point = self.calibration.predict(gaze_ratio)
        self.last_screen_point = screen_point

        # Step 5: Hit test against visible windows
        windows = self._get_windows()
        return hit_test(screen_point.x, screen_point.y, windows)
