"""Gaze-to-window detection orchestrator."""

from collections import deque

import numpy as np

from vibeyes import GazeRatio, Point, WindowInfo
from vibeyes.window_tracker import hit_test


def _get_screen_bounds() -> tuple[float, float]:
    """Get main screen width and height."""
    try:
        import Quartz
        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return float(bounds.size.width), float(bounds.size.height)
    except ImportError:
        return 1920.0, 1080.0


class Detector:
    """Orchestrates the gaze-to-window detection pipeline.

    Pipeline: frame -> face_tracker -> gaze_estimator -> calibration ->
              median filter -> EMA smooth -> clamp -> window hit_test
    """

    def __init__(self, face_tracker, gaze_estimator, calibration, get_windows,
                 median_window: int = 7, ema_factor: float = 0.4):
        self.face_tracker = face_tracker
        self.gaze_estimator = gaze_estimator
        self.calibration = calibration
        self._get_windows = get_windows
        self.last_screen_point: Point | None = None
        self.last_gaze_ratio: GazeRatio | None = None

        # Smoothing state
        self._history_x: deque[float] = deque(maxlen=median_window)
        self._history_y: deque[float] = deque(maxlen=median_window)
        self._ema_factor = ema_factor
        self._smoothed_x: float | None = None
        self._smoothed_y: float | None = None

        # Screen bounds for clamping
        self._screen_w, self._screen_h = _get_screen_bounds()

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

        # Step 5: Median filter (rejects outlier spikes)
        self._history_x.append(raw_screen.x)
        self._history_y.append(raw_screen.y)
        median_x = sorted(self._history_x)[len(self._history_x) // 2]
        median_y = sorted(self._history_y)[len(self._history_y) // 2]

        # Step 6: Light EMA on top of median (visual smoothness)
        if self._smoothed_x is None:
            self._smoothed_x = median_x
            self._smoothed_y = median_y
        else:
            s = self._ema_factor
            self._smoothed_x = s * median_x + (1 - s) * self._smoothed_x
            self._smoothed_y = s * median_y + (1 - s) * self._smoothed_y

        # Step 7: Clamp to screen bounds
        final_x = max(0.0, min(self._screen_w, self._smoothed_x))
        final_y = max(0.0, min(self._screen_h, self._smoothed_y))

        self.last_screen_point = Point(x=final_x, y=final_y)

        # Step 8: Hit test against visible windows
        windows = self._get_windows()
        return hit_test(final_x, final_y, windows)
