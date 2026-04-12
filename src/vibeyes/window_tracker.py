"""macOS window enumeration and hit testing."""

from vibeyes import WindowInfo


def get_visible_windows(exclude_app: str | None = None) -> list[WindowInfo]:
    """Get all visible windows on screen, ordered front-to-back."""
    raise NotImplementedError


def hit_test(x: float, y: float, windows: list[WindowInfo]) -> WindowInfo | None:
    """Return the frontmost window containing the point (x, y), or None."""
    raise NotImplementedError
