"""Screen coordinate calibration via polynomial regression."""

import json

import numpy as np

from vibeyes import GazeRatio, Point

MIN_CALIBRATION_POINTS = 6


class Calibration:
    """Maps gaze features (iris + head) to screen coordinates using linear regression.

    Uses features: [1, iris_x, iris_y, head_x, head_y, iris_x*head_x, iris_y*head_y]
    This lets the model learn from both iris position (fine signal) and
    head pose (coarse but strong signal).
    """

    def __init__(self):
        self._gaze_points: list[tuple[float, ...]] = []
        self._screen_points: list[tuple[float, float]] = []
        self._coeffs_x: np.ndarray | None = None
        self._coeffs_y: np.ndarray | None = None

    @property
    def point_count(self) -> int:
        return len(self._gaze_points)

    @property
    def is_calibrated(self) -> bool:
        return self._coeffs_x is not None

    def add_point(self, gaze: GazeRatio, screen: Point):
        """Record a calibration data point."""
        self._gaze_points.append((gaze.x, gaze.y, gaze.head_x, gaze.head_y))
        self._screen_points.append((screen.x, screen.y))

    def fit(self):
        """Fit linear regression from gaze features to screen coordinates."""
        if len(self._gaze_points) < MIN_CALIBRATION_POINTS:
            raise ValueError(
                f"Need at least {MIN_CALIBRATION_POINTS} calibration points, "
                f"got {len(self._gaze_points)}"
            )

        A = self._build_feature_matrix(self._gaze_points)
        screen = np.array(self._screen_points)

        self._coeffs_x, _, _, _ = np.linalg.lstsq(A, screen[:, 0], rcond=None)
        self._coeffs_y, _, _, _ = np.linalg.lstsq(A, screen[:, 1], rcond=None)

    def predict(self, gaze: GazeRatio) -> Point:
        """Map gaze features to screen coordinates using the fitted model."""
        if not self.is_calibrated:
            raise RuntimeError("Calibration not calibrated -- call fit() first")

        features = self._build_feature_matrix([(gaze.x, gaze.y, gaze.head_x, gaze.head_y)])
        sx = float((features @ self._coeffs_x)[0])
        sy = float((features @ self._coeffs_y)[0])
        return Point(x=sx, y=sy)

    def save(self, path: str):
        """Save calibration data and coefficients to a JSON file."""
        data = {
            "gaze_points": self._gaze_points,
            "screen_points": self._screen_points,
            "coeffs_x": self._coeffs_x.tolist() if self._coeffs_x is not None else None,
            "coeffs_y": self._coeffs_y.tolist() if self._coeffs_y is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        """Load a calibration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        cal = cls()
        cal._gaze_points = [tuple(p) for p in data["gaze_points"]]
        cal._screen_points = [tuple(p) for p in data["screen_points"]]
        if data["coeffs_x"] is not None:
            cal._coeffs_x = np.array(data["coeffs_x"])
            cal._coeffs_y = np.array(data["coeffs_y"])
        return cal

    def clear(self):
        """Remove all calibration data and fitted model."""
        self._gaze_points.clear()
        self._screen_points.clear()
        self._coeffs_x = None
        self._coeffs_y = None

    @staticmethod
    def _build_feature_matrix(points: list[tuple[float, ...]]) -> np.ndarray:
        """Build feature matrix from gaze features.

        Input: each point is (iris_x, iris_y, head_x, head_y)
        Features: [1, iris_x, iris_y, head_x, head_y, iris_x*head_x, iris_y*head_y]

        For backwards compatibility with 2-element tuples (old calibration files),
        head_x and head_y default to 0.5.
        """
        pts = np.array(points)
        n = len(pts)

        if pts.shape[1] >= 4:
            ix, iy = pts[:, 0], pts[:, 1]
            hx, hy = pts[:, 2], pts[:, 3]
        else:
            # Old format: only iris_x, iris_y
            ix, iy = pts[:, 0], pts[:, 1]
            hx, hy = np.full(n, 0.5), np.full(n, 0.5)

        return np.column_stack([
            np.ones(n),
            ix, iy,
            hx, hy,
            ix * hx,
            iy * hy,
        ])
