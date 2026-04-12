"""Face and iris landmark detection using MediaPipe."""

from vibeyes import FaceData


class FaceTracker:
    """Detects face and iris landmarks from video frames using MediaPipe FaceLandmarker."""

    def __init__(self, model_path: str):
        raise NotImplementedError

    def detect(self, frame) -> FaceData | None:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
