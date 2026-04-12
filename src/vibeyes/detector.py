"""Gaze-to-window detection orchestrator."""

import numpy as np

from vibeyes import GazeRatio, Point, WindowInfo
from vibeyes.window_tracker import hit_test


class Detector:
    """Orchestrates the gaze-to-window detection pipeline.

    Pipeline: frame -> face_tracker -> gaze_estimator -> calibration -> window hit_test
    """

    def __init__(self, face_tracker, gaze_estimator, calibration, get_windows, screen_smoothing: float = 0.3):
        self.face_tracker = face_tracker
        self.gaze_estimator = gaze_estimator
        self.calibration = calibration
        self._get_windows = get_windows
        self.last_screen_point: Point | None = None
        self.last_gaze_ratio: GazeRatio | None = None
        self._screen_smoothing = screen_smoothing
        self._smoothed_screen: Point | None = None

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
        self.last_gaze_ratio = gaze_ratio

        # Step 4: Map to screen coordinates
        raw_screen = self.calibration.predict(gaze_ratio)

        # Step 5: Smooth screen coordinates (EMA)
        if self._smoothed_screen is None:
            self._smoothed_screen = raw_screen
        else:
            s = self._screen_smoothing
            self._smoothed_screen = Point(
                x=s * raw_screen.x + (1 - s) * self._smoothed_screen.x,
                y=s * raw_screen.y + (1 - s) * self._smoothed_screen.y,
            )
        self.last_screen_point = self._smoothed_screen

        # Step 6: Hit test against visible windows
        windows = self._get_windows()
        return hit_test(self._smoothed_screen.x, self._smoothed_screen.y, windows)
