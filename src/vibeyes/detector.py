"""Gaze-to-window detection orchestrator."""

from vibeyes import WindowInfo


class Detector:
    """Orchestrates the gaze-to-window detection pipeline."""

    def __init__(self, face_tracker, gaze_estimator, calibration, get_windows):
        raise NotImplementedError

    def detect(self, frame) -> WindowInfo | None:
        raise NotImplementedError
