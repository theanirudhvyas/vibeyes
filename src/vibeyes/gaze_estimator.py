"""Gaze direction estimation from iris landmarks."""

from vibeyes import FaceData, GazeRatio


class GazeEstimator:
    """Computes normalized gaze ratio from eye landmark positions."""

    def __init__(self, smoothing_factor: float = 0.5):
        raise NotImplementedError

    def estimate(self, face_data: FaceData) -> GazeRatio:
        raise NotImplementedError
