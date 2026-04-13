"""Face and iris landmark detection using MediaPipe with 3D head pose."""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

from vibeyes import EyeData, FaceData, HeadPose, Point

# MediaPipe landmark indices
LEFT_IRIS_CENTER = 473
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_IRIS_CENTER = 468
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

NOSE_TIP = 1

# Landmarks for solvePnP head pose estimation
# nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]

# Corresponding 3D model points (generic face model, in mm)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye outer corner
    [43.3, 32.7, -26.0],    # right eye outer corner
    [-28.9, -28.9, -24.1],  # left mouth corner
    [28.9, -28.9, -24.1],   # right mouth corner
], dtype=np.float64)


class FaceTracker:
    """Detects face/iris landmarks and computes 3D head pose."""

    def __init__(self, model_path: str):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_shape: tuple[int, int] | None = None
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _get_camera_matrix(self, h: int, w: int) -> np.ndarray:
        """Approximate camera intrinsic matrix from frame dimensions."""
        if self._camera_matrix is None or self._frame_shape != (h, w):
            focal_length = w  # approximate
            cx, cy = w / 2, h / 2
            self._camera_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1],
            ], dtype=np.float64)
            self._frame_shape = (h, w)
        return self._camera_matrix

    def _compute_head_pose(self, landmarks, h: int, w: int) -> HeadPose:
        """Compute head yaw/pitch/roll using solvePnP on 6 face landmarks."""
        image_points = np.array([
            [landmarks[idx].x * w, landmarks[idx].y * h]
            for idx in POSE_LANDMARKS
        ], dtype=np.float64)

        cam_matrix = self._get_camera_matrix(h, w)
        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, image_points, cam_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix to Euler angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)

    def detect(self, frame: np.ndarray) -> FaceData | None:
        """Detect face landmarks in a BGR frame. Returns FaceData or None."""
        h, w = frame.shape[:2]
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

        head_pose = self._compute_head_pose(landmarks, h, w)

        return FaceData(
            left_eye=left_eye,
            right_eye=right_eye,
            nose_tip=_point(NOSE_TIP),
            head_pose=head_pose,
        )

    def close(self):
        """Release the landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
