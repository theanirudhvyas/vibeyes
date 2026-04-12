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


@dataclass
class GazeRatio:
    """Normalized gaze direction (0-1 range). (0,0) = top-left, (1,1) = bottom-right."""
    x: float
    y: float


@dataclass
class WindowInfo:
    """Information about a visible macOS window."""
    title: str
    app_name: str
    bounds: tuple[float, float, float, float]  # (x, y, width, height)
    pid: int
    layer: int
