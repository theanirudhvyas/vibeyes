"""Gaze direction estimation from iris landmarks and 3D head pose."""

from vibeyes import EyeData, FaceData, GazeRatio

# Eye Aspect Ratio threshold -- below this the eye is considered closed (blink)
EAR_BLINK_THRESHOLD = 0.15


def _eye_aspect_ratio(eye: EyeData) -> float:
    """Compute Eye Aspect Ratio (EAR) -- low value means eye is closed."""
    top_y = min(eye.top.y, eye.bottom.y)
    bottom_y = max(eye.top.y, eye.bottom.y)
    left_x = min(eye.inner_corner.x, eye.outer_corner.x)
    right_x = max(eye.inner_corner.x, eye.outer_corner.x)
    vertical = bottom_y - top_y
    horizontal = right_x - left_x
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _eye_gaze_ratio(eye: EyeData) -> tuple[float, float]:
    """Compute horizontal and vertical gaze ratio for one eye.

    Uses min/max of corners to be orientation-agnostic.
    Returns (x_ratio, y_ratio) in range [0, 1].
    """
    left_x = min(eye.inner_corner.x, eye.outer_corner.x)
    right_x = max(eye.inner_corner.x, eye.outer_corner.x)
    x_range = right_x - left_x
    if x_range < 1e-6:
        x_ratio = 0.5
    else:
        x_ratio = (eye.iris_center.x - left_x) / x_range

    top_y = min(eye.top.y, eye.bottom.y)
    bottom_y = max(eye.top.y, eye.bottom.y)
    y_range = bottom_y - top_y
    if y_range < 1e-6:
        y_ratio = 0.5
    else:
        y_ratio = (eye.iris_center.y - top_y) / y_range

    return x_ratio, y_ratio


class GazeEstimator:
    """Computes gaze features from iris landmarks and 3D head pose."""

    def __init__(self, smoothing_factor: float = 0.5):
        self._smoothing = smoothing_factor
        self._prev: GazeRatio | None = None

    def estimate(self, face_data: FaceData) -> GazeRatio | None:
        """Compute smoothed gaze features. Returns None during blinks."""
        # Blink detection: if either eye is closed, return previous gaze
        left_ear = _eye_aspect_ratio(face_data.left_eye)
        right_ear = _eye_aspect_ratio(face_data.right_eye)

        if left_ear < EAR_BLINK_THRESHOLD or right_ear < EAR_BLINK_THRESHOLD:
            return self._prev  # freeze during blink

        lx, ly = _eye_gaze_ratio(face_data.left_eye)
        rx, ry = _eye_gaze_ratio(face_data.right_eye)

        iris_x = max(0.0, min(1.0, (lx + rx) / 2.0))
        iris_y = max(0.0, min(1.0, (ly + ry) / 2.0))

        head_yaw = face_data.head_pose.yaw
        head_pitch = face_data.head_pose.pitch

        # EMA smoothing
        if self._prev is None:
            sx, sy = iris_x, iris_y
            shx, shy = head_yaw, head_pitch
        else:
            s = self._smoothing
            sx = s * iris_x + (1 - s) * self._prev.x
            sy = s * iris_y + (1 - s) * self._prev.y
            shx = s * head_yaw + (1 - s) * self._prev.head_x
            shy = s * head_pitch + (1 - s) * self._prev.head_y

        result = GazeRatio(x=sx, y=sy, head_x=shx, head_y=shy)
        self._prev = result
        return result
