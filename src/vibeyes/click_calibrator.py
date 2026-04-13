"""Implicit calibration from mouse clicks.

Monitors global mouse clicks and uses them as calibration points:
when you click somewhere, you're presumably looking there.
"""

import threading
import time

import Quartz

from vibeyes import GazeRatio, Point
from vibeyes.calibration import Calibration


class ClickCalibrator:
    """Collects mouse clicks and uses them to refine calibration in real-time."""

    def __init__(self, calibration: Calibration, min_points_to_refit: int = 5, max_points: int = 100):
        self._calibration = calibration
        self._min_points = min_points_to_refit
        self._max_points = max_points
        self._pending_clicks: list[tuple[float, float, float]] = []  # (x, y, timestamp)
        self._click_count = 0
        self._last_gaze: GazeRatio | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start monitoring mouse clicks in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_tap, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._running = False

    def update_gaze(self, gaze: GazeRatio):
        """Update the current gaze estimate (called each frame from tracking loop)."""
        self._last_gaze = gaze

    def check_refit(self) -> bool:
        """Check if we have enough new click-based points to refit. Returns True if refitted."""
        with self._lock:
            if self._click_count >= self._min_points:
                try:
                    self._calibration.fit()
                    count = self._click_count
                    self._click_count = 0
                    print(f"  [click-cal] Refitted with {count} new click points "
                          f"({self._calibration.point_count} total)")
                    return True
                except ValueError:
                    pass
        return False

    def _run_tap(self):
        """Run a Quartz event tap to monitor mouse clicks."""
        def callback(proxy, event_type, event, refcon):
            if not self._running:
                return event

            if event_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown):
                loc = Quartz.CGEventGetLocation(event)
                click_x, click_y = loc.x, loc.y

                gaze = self._last_gaze
                if gaze is not None:
                    with self._lock:
                        self._calibration.add_point(gaze, Point(click_x, click_y))
                        self._click_count += 1

                        # Trim old points if over max (keep recent ones)
                        if self._calibration.point_count > self._max_points:
                            # Keep the last max_points entries
                            self._calibration._gaze_points = self._calibration._gaze_points[-self._max_points:]
                            self._calibration._screen_points = self._calibration._screen_points[-self._max_points:]

            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,  # passive, don't modify events
            Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown) | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown),
            callback,
            None,
        )

        if tap is None:
            print("  [click-cal] Could not create event tap. Grant Accessibility permission.")
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(loop, source, Quartz.kCFRunLoopDefaultMode)
        Quartz.CGEventTapEnable(tap, True)

        while self._running:
            Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.5, False)
