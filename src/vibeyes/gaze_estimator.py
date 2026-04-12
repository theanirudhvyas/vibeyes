"""Gaze direction estimation from iris landmarks and head pose."""

from vibeyes import EyeData, FaceData, GazeRatio


def _eye_gaze_ratio(eye: EyeData) -> tuple[float, float]:
    """Compute horizontal and vertical gaze ratio for one eye.

    Returns (x_ratio, y_ratio) where 0.0 = inner/top and 1.0 = outer/bottom.
    """
    x_range = eye.outer_corner.x - eye.inner_corner.x
    if abs(x_range) < 1e-6:
        x_ratio = 0.5
    else:
        x_ratio = (eye.iris_center.x - eye.inner_corner.x) / x_range

    y_range = eye.bottom.y - eye.top.y
    if abs(y_range) < 1e-6:
        y_ratio = 0.5
    else:
        y_ratio = (eye.iris_center.y - eye.top.y) / y_range

    return x_ratio, y_ratio


class GazeEstimator:
    """Computes gaze features from eye landmarks and head position."""

    def __init__(self, smoothing_factor: float = 0.5):
        self._smoothing = smoothing_factor
        self._prev: GazeRatio | None = None

    def estimate(self, face_data: FaceData) -> GazeRatio:
        """Compute smoothed gaze features from face landmarks."""
        lx, ly = _eye_gaze_ratio(face_data.left_eye)
        rx, ry = _eye_gaze_ratio(face_data.right_eye)

        # Average iris ratios from both eyes
        iris_x = max(0.0, min(1.0, (lx + rx) / 2.0))
        iris_y = max(0.0, min(1.0, (ly + ry) / 2.0))

        # Head position (nose tip in frame, 0-1 normalized)
        head_x = face_data.nose_tip.x
        head_y = face_data.nose_tip.y

        # Combine: use iris ratio as primary signal, head position as secondary
        raw_x = iris_x
        raw_y = iris_y

        # EMA smoothing
        if self._prev is None:
            sx, sy = raw_x, raw_y
            shx, shy = head_x, head_y
        else:
            s = self._smoothing
            sx = s * raw_x + (1 - s) * self._prev.x
            sy = s * raw_y + (1 - s) * self._prev.y
            shx = s * head_x + (1 - s) * self._prev.head_x
            shy = s * head_y + (1 - s) * self._prev.head_y

        result = GazeRatio(x=sx, y=sy, head_x=shx, head_y=shy)
        self._prev = result
        return result
