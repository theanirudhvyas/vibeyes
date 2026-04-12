"""Screen coordinate calibration via polynomial regression."""

from vibeyes import GazeRatio, Point


class Calibration:
    """Maps gaze ratios to screen coordinates using polynomial regression."""

    def __init__(self):
        raise NotImplementedError

    @property
    def point_count(self) -> int:
        raise NotImplementedError

    @property
    def is_calibrated(self) -> bool:
        raise NotImplementedError

    def add_point(self, gaze: GazeRatio, screen: Point):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self, gaze: GazeRatio) -> Point:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Calibration":
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
