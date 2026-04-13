"""VibEyes: Webcam-based gaze tracking for window focus detection."""

from dataclasses import dataclass


@dataclass
class Point:
    """A 2D point with x, y coordinates (normalized 0-1 for landmarks, pixels for screen)."""
    x: float
    y: float


@dataclass
class EyeData:
    """Landmarks for one eye: iris center + inner/outer corners."""
    iris_center: Point
    inner_corner: Point
    outer_corner: Point
    top: Point
    bottom: Point


@dataclass
class HeadPose:
    """Head orientation from 3D face mesh via solvePnP."""
    yaw: float    # left/right rotation in degrees (negative=left, positive=right)
    pitch: float  # up/down rotation in degrees (negative=up, positive=down)
    roll: float   # tilt in degrees


@dataclass
class FaceData:
    """Extracted face tracking data for gaze estimation."""
    left_eye: EyeData
    right_eye: EyeData
    nose_tip: Point
    head_pose: HeadPose


@dataclass
class GazeRatio:
    """Gaze features for calibration mapping.

    iris_x, iris_y: iris position ratio within eye (0-1)
    head_yaw, head_pitch: head rotation angles from 3D mesh (degrees)
    """
    x: float       # iris x ratio
    y: float       # iris y ratio
    head_x: float = 0.0  # head yaw in degrees
    head_y: float = 0.0  # head pitch in degrees


@dataclass
class WindowInfo:
    """Information about a visible macOS window."""
    title: str
    app_name: str
    bounds: tuple[float, float, float, float]  # (x, y, width, height)
    pid: int
    layer: int
