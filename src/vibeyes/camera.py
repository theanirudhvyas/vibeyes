"""Webcam capture module."""

import sys
import time

import cv2
import numpy as np


def _check_camera_permission() -> bool:
    """Check macOS camera authorization status via AVFoundation.

    Returns True if authorized, False if denied/restricted.
    Returns True on non-macOS platforms (no permission check needed).
    """
    if sys.platform != "darwin":
        return True

    try:
        import AVFoundation

        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
        # 0 = notDetermined, 1 = restricted, 2 = denied, 3 = authorized
        return status == 3
    except ImportError:
        # pyobjc-framework-AVFoundation not installed -- fall through to OpenCV
        return True


def _request_camera_permission():
    """Trigger the macOS camera permission dialog and wait for the user to respond."""
    if sys.platform != "darwin":
        return

    try:
        import AVFoundation
        import objc

        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeVideo
        )

        if status == 0:  # notDetermined
            print("Requesting camera permission... Please allow in the macOS dialog.")
            granted = [None]

            def handler(result):
                granted[0] = result

            AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVFoundation.AVMediaTypeVideo, handler
            )

            # Wait for user to respond to the dialog (up to 60 seconds)
            deadline = time.time() + 60
            while granted[0] is None and time.time() < deadline:
                time.sleep(0.2)

            if not granted[0]:
                raise RuntimeError(
                    "Camera permission denied. Grant access in "
                    "System Settings > Privacy & Security > Camera."
                )
        elif status == 2:  # denied
            raise RuntimeError(
                "Camera permission denied. Grant access in "
                "System Settings > Privacy & Security > Camera."
            )
        elif status == 1:  # restricted
            raise RuntimeError("Camera access is restricted by system policy.")
    except ImportError:
        # No AVFoundation bindings -- OpenCV will trigger the dialog itself
        pass


def list_cameras() -> list[dict]:
    """List available cameras by probing OpenCV indices."""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({"index": i, "name": f"Camera {i}"})
            cap.release()
    return cameras


def select_camera_interactive(cameras: list[dict]) -> int:
    """Cycle through cameras with live preview. User picks visually.

    For each camera, shows a live preview window. User presses:
      SPACE/ENTER = accept this camera
      N = next camera
      ESC = abort
    Returns the selected OpenCV device index.
    """
    for cam in cameras:
        idx = cam["index"]
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue

        title = f"Camera {idx} -- SPACE to use, N for next, ESC to quit"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

        selected = None
        while selected is None:
            ret, frame = cap.read()
            if ret:
                # Add label on frame
                cv2.putText(
                    frame, f"Camera {idx} - Press SPACE to select, N for next",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow(title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), 13):  # SPACE or ENTER
                selected = True
            elif key in (ord("n"), ord("N")):
                selected = False
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cap.release()
                print("Camera selection aborted.")
                sys.exit(1)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cap.release()

        if selected:
            print(f"Selected camera {idx}")
            return idx

    # Shouldn't reach here, but fallback to first camera
    return cameras[0]["index"]


class Camera:
    """Captures frames from a webcam using OpenCV."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480, max_retries: int = 3):
        # Handle macOS camera permissions before opening the device
        _request_camera_permission()

        self._cap = None
        for attempt in range(max_retries):
            cap = cv2.VideoCapture(device)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
            if attempt < max_retries - 1:
                print(f"Camera not ready, retrying ({attempt + 2}/{max_retries})...")
                time.sleep(2)

        if self._cap is None:
            raise RuntimeError(
                f"Cannot open camera device {device}. "
                "Check System Settings > Privacy & Security > Camera."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
