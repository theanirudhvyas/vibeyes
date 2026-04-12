"""macOS window enumeration and hit testing."""

import Quartz

from vibeyes import WindowInfo


def get_visible_windows(exclude_app: str | None = None) -> list[WindowInfo]:
    """Get all visible windows on screen, ordered front-to-back.

    Uses CGWindowListCopyWindowInfo to enumerate macOS windows.
    Filters out desktop elements and zero-size windows.
    """
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

    if window_list is None:
        return []

    windows = []
    for w in window_list:
        app_name = w.get(Quartz.kCGWindowOwnerName, "")
        if not app_name:
            continue
        if exclude_app and app_name == exclude_app:
            continue

        bounds = w.get(Quartz.kCGWindowBounds, {})
        x = bounds.get("X", 0)
        y = bounds.get("Y", 0)
        width = bounds.get("Width", 0)
        height = bounds.get("Height", 0)

        # Skip zero-size windows (menu bar items, etc.)
        if width <= 0 or height <= 0:
            continue

        title = w.get(Quartz.kCGWindowName, "") or ""
        pid = w.get(Quartz.kCGWindowOwnerPID, 0)
        layer = w.get(Quartz.kCGWindowLayer, 0)

        windows.append(WindowInfo(
            title=title,
            app_name=app_name,
            bounds=(x, y, width, height),
            pid=pid,
            layer=layer,
        ))

    return windows


def hit_test(x: float, y: float, windows: list[WindowInfo]) -> WindowInfo | None:
    """Return the frontmost window containing the point (x, y), or None.

    Windows are expected to be ordered front-to-back (first = frontmost).
    """
    for w in windows:
        wx, wy, ww, wh = w.bounds
        if wx <= x <= wx + ww and wy <= y <= wy + wh:
            return w
    return None
