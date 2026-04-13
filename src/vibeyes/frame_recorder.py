"""Records webcam frames on click events for offline replay."""

import json
import os
import time

import cv2
import numpy as np


class FrameRecorder:
    """Saves webcam frames and click positions to disk for offline analysis."""

    def __init__(self, base_dir: str = os.path.expanduser("~/.cache/vibeyes/recordings")):
        session_id = f"session_{int(time.time())}"
        self._session_dir = os.path.join(base_dir, session_id)
        self._frames_dir = os.path.join(self._session_dir, "frames")
        os.makedirs(self._frames_dir, exist_ok=True)
        self._clicks_path = os.path.join(self._session_dir, "clicks.jsonl")
        self._last_frame: np.ndarray | None = None
        self._frame_count = 0

    def update_frame(self, frame: np.ndarray):
        """Buffer the latest frame (called every frame, only saved on click)."""
        self._last_frame = frame.copy()

    def save_click(self, click_x: float, click_y: float):
        """Save the buffered frame and click position to disk."""
        if self._last_frame is None:
            return

        self._frame_count += 1
        frame_id = f"{self._frame_count:06d}"

        frame_path = os.path.join(self._frames_dir, f"{frame_id}.jpg")
        cv2.imwrite(frame_path, self._last_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        record = {
            "frame_id": frame_id,
            "click_x": click_x,
            "click_y": click_y,
            "timestamp": time.time(),
        }
        with open(self._clicks_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def session_dir(self) -> str:
        return self._session_dir

    @property
    def click_count(self) -> int:
        return self._frame_count
