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
class FaceData:
    """Extracted face tracking data for gaze estimation."""
    left_eye: EyeData
    right_eye: EyeData
    nose_tip: Point  # face center reference for head pose signal


@dataclass
class GazeRatio:
    """Gaze features for calibration mapping.

    iris_x, iris_y: iris position ratio within eye (0-1)
    head_x, head_y: nose tip position in frame (0-1), captures head pose
    """
    x: float  # combined/primary x signal
    y: float  # combined/primary y signal
    head_x: float = 0.5  # nose x position (head pose)
    head_y: float = 0.5  # nose y position (head pose)


@dataclass
class WindowInfo:
    """Information about a visible macOS window."""
    title: str
    app_name: str
    bounds: tuple[float, float, float, float]  # (x, y, width, height)
    pid: int
    layer: int
