"""Implicit calibration from mouse clicks.

Monitors global mouse clicks via NSEvent and uses them as calibration
points: when you click somewhere, you're presumably looking there.
"""

import threading

import Quartz
from Cocoa import NSApplication, NSEvent

from vibeyes import GazeRatio, Point
from vibeyes.calibration import Calibration
from vibeyes.metrics import MetricsTracker


class ClickCalibrator:
    """Collects mouse clicks and uses them to refine calibration in real-time."""

    def __init__(self, calibration: Calibration, min_points_to_refit: int = 5, max_points: int = 100):
        self._calibration = calibration
        self._min_points = min_points_to_refit
        self._max_points = max_points
        self._click_count = 0
        self._last_gaze: GazeRatio | None = None
        self._last_screen_point: Point | None = None
        self._lock = threading.Lock()
        self._monitor = None
        self._metrics = MetricsTracker()

    def start(self):
        """Start monitoring mouse clicks via NSEvent global monitor."""
        # Ensure NSApplication exists (needed for NSEvent monitors)
        NSApplication.sharedApplication()

        mask = (1 << 1) | (1 << 3)  # NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown

        def handler(event):
            gaze = self._last_gaze
            predicted = self._last_screen_point
            if gaze is None:
                return

            # Get click position in screen coords (top-left origin via Quartz)
            loc = Quartz.NSEvent.mouseLocation()
            # Convert from Cocoa bottom-left to top-left origin
            screen_h = Quartz.CGDisplayPixelsHigh(Quartz.CGMainDisplayID())
            click_x = loc.x
            click_y = screen_h - loc.y

            with self._lock:
                # Record error metric
                if predicted is not None:
                    self._metrics.record_click(
                        predicted.x, predicted.y,
                        click_x, click_y,
                        self._calibration.point_count,
                    )

                self._calibration.add_point(gaze, Point(click_x, click_y))
                self._click_count += 1

                # Trim old points if over max
                if self._calibration.point_count > self._max_points:
                    self._calibration._gaze_points = self._calibration._gaze_points[-self._max_points:]
                    self._calibration._screen_points = self._calibration._screen_points[-self._max_points:]

        self._monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, handler)
        if self._monitor is None:
            print("  [click-cal] Could not create event monitor.")

    def stop(self):
        """Stop monitoring and print final stats."""
        if self._monitor is not None:
            NSEvent.removeMonitor_(self._monitor)
            self._monitor = None
        stats = self._metrics.get_stats()
        if stats["total_clicks"] > 0:
            print(f"\n  Accuracy stats: {stats['total_clicks']} clicks tracked, "
                  f"avg error={stats['avg_error_px']}px, "
                  f"recent avg={stats['recent_avg_error_px']}px")
        self._metrics.close()

    def update_gaze(self, gaze: GazeRatio, screen_point: Point | None = None):
        """Update the current gaze estimate (called each frame from tracking loop)."""
        self._last_gaze = gaze
        self._last_screen_point = screen_point

    def check_refit(self) -> bool:
        """Check if we have enough new click-based points to refit."""
        with self._lock:
            if self._click_count >= self._min_points:
                try:
                    self._calibration.fit()
                    count = self._click_count
                    self._click_count = 0
                    recent_err = self._metrics.get_recent_avg_error(20)
                    err_str = f", recent avg error={recent_err:.0f}px" if recent_err else ""
                    print(f"  [click-cal] Refitted with {count} new clicks "
                          f"({self._calibration.point_count} total){err_str}")
                    return True
                except ValueError:
                    pass
        return False
