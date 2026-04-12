"""Face and iris landmark detection using MediaPipe."""

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

from vibeyes import EyeData, FaceData, Point

# MediaPipe landmark indices for iris and eye corners.
# Left eye (from the subject's perspective, right side in image):
LEFT_IRIS_CENTER = 473
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

# Right eye (from the subject's perspective, left side in image):
RIGHT_IRIS_CENTER = 468
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Nose tip for head pose signal
NOSE_TIP = 1


class FaceTracker:
    """Detects face and iris landmarks from video frames using MediaPipe FaceLandmarker."""

    def __init__(self, model_path: str):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> FaceData | None:
        """Detect face landmarks in a BGR frame. Returns FaceData or None if no face found."""
        rgb = frame[:, :, ::-1]  # BGR to RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        def _point(idx: int) -> Point:
            lm = landmarks[idx]
            return Point(x=lm.x, y=lm.y)

        left_eye = EyeData(
            iris_center=_point(LEFT_IRIS_CENTER),
            inner_corner=_point(LEFT_EYE_INNER_CORNER),
            outer_corner=_point(LEFT_EYE_OUTER_CORNER),
            top=_point(LEFT_EYE_TOP),
            bottom=_point(LEFT_EYE_BOTTOM),
        )
        right_eye = EyeData(
            iris_center=_point(RIGHT_IRIS_CENTER),
            inner_corner=_point(RIGHT_EYE_INNER_CORNER),
            outer_corner=_point(RIGHT_EYE_OUTER_CORNER),
            top=_point(RIGHT_EYE_TOP),
            bottom=_point(RIGHT_EYE_BOTTOM),
        )

        return FaceData(left_eye=left_eye, right_eye=right_eye, nose_tip=_point(NOSE_TIP))

    def close(self):
        """Release the landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
