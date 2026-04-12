"""Gaze direction estimation from iris landmarks."""

from vibeyes import EyeData, FaceData, GazeRatio


def _eye_gaze_ratio(eye: EyeData) -> tuple[float, float]:
    """Compute horizontal and vertical gaze ratio for one eye.

    Returns (x_ratio, y_ratio) where 0.0 = inner/top and 1.0 = outer/bottom.
    """
    # Horizontal: iris position between inner and outer corner
    x_range = eye.outer_corner.x - eye.inner_corner.x
    if abs(x_range) < 1e-6:
        x_ratio = 0.5
    else:
        x_ratio = (eye.iris_center.x - eye.inner_corner.x) / x_range

    # Vertical: iris position between top and bottom of eye
    y_range = eye.bottom.y - eye.top.y
    if abs(y_range) < 1e-6:
        y_ratio = 0.5
    else:
        y_ratio = (eye.iris_center.y - eye.top.y) / y_range

    return x_ratio, y_ratio


class GazeEstimator:
    """Computes normalized gaze ratio from eye landmark positions."""

    def __init__(self, smoothing_factor: float = 0.5):
        """Initialize with smoothing factor (0-1). 1.0 = no smoothing, 0.0 = max smoothing."""
        self._smoothing = smoothing_factor
        self._prev: GazeRatio | None = None

    def estimate(self, face_data: FaceData) -> GazeRatio:
        """Compute smoothed gaze ratio from face landmarks."""
        lx, ly = _eye_gaze_ratio(face_data.left_eye)
        rx, ry = _eye_gaze_ratio(face_data.right_eye)

        # Average both eyes
        raw_x = (lx + rx) / 2.0
        raw_y = (ly + ry) / 2.0

        # Clamp to [0, 1]
        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))

        # Exponential moving average smoothing
        if self._prev is None:
            smoothed_x = raw_x
            smoothed_y = raw_y
        else:
            smoothed_x = self._smoothing * raw_x + (1 - self._smoothing) * self._prev.x
            smoothed_y = self._smoothing * raw_y + (1 - self._smoothing) * self._prev.y

        result = GazeRatio(x=smoothed_x, y=smoothed_y)
        self._prev = result
        return result
