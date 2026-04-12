"""Webcam capture module."""

import cv2
import numpy as np


class Camera:
    """Captures frames from a webcam using OpenCV."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device}")

    def read(self) -> np.ndarray | None:
        """Read a single frame. Returns BGR numpy array or None on failure."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
