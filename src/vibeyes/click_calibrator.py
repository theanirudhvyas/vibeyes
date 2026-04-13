"""Implicit calibration from mouse clicks.

Uses a CGEventTap on the main thread to capture mouse clicks globally.
Requires Accessibility permission (System Settings > Privacy & Security > Accessibility).
"""

import sys
import time

import Quartz

from vibeyes import GazeRatio, Point
from vibeyes.calibration import Calibration
from vibeyes.metrics import MetricsTracker


def _ensure_accessibility_permission():
    """Check for Accessibility permission and prompt the user if needed."""
    if sys.platform != "darwin":
        return True

    from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions

    if AXIsProcessTrusted():
        return True

    # Prompt the user -- macOS will show a system dialog
    print("  Requesting Accessibility permission for click tracking...")
    AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})

    print("  Please grant permission in the system dialog, then wait...")
    for _ in range(30):
        time.sleep(1)
        if AXIsProcessTrusted():
            print("  Accessibility permission granted!")
            return True

    print("  Permission not granted. Click calibration will be disabled.")
    print("  You can grant it later in System Settings > Privacy & Security > Accessibility")
    return False


class ClickCalibrator:
    """Collects mouse clicks via CGEventTap and uses them to refine calibration.

    The event tap and its run loop source live on the main thread. Call pump()
    each frame from the tracking loop to process pending events.
    """

    def __init__(self, calibration: Calibration, min_points_to_refit: int = 5, max_points: int = 100):
        self._calibration = calibration
        self._min_points = min_points_to_refit
        self._max_points = max_points
        self._click_count = 0
        self._last_gaze: GazeRatio | None = None
        self._last_screen_point: Point | None = None
        self._metrics = MetricsTracker()
        self._tap = None
        self._source = None

    def start(self):
        """Create the CGEventTap on the current (main) thread."""
        if not _ensure_accessibility_permission():
            print("  [click-cal] Skipping click calibration (no Accessibility permission)")
            return

        def callback(proxy, event_type, event, refcon):
            if event_type in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown):
                self._on_click(event)
            return event

        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            (Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown) |
             Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)),
            callback,
            None,
        )

        if self._tap is None:
            print("  [click-cal] Cannot create event tap.")
            print("  Grant Accessibility permission to your terminal app:")
            print("  System Settings > Privacy & Security > Accessibility")
            return

        self._source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetMain(),
            self._source,
            Quartz.kCFRunLoopCommonModes,
        )
        Quartz.CGEventTapEnable(self._tap, True)

    def pump(self):
        """Process pending event tap callbacks. Call once per frame from main thread."""
        if self._tap is None:
            return
        # Run the main run loop briefly to process any queued events
        Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.001, True)

    def stop(self):
        """Disable the event tap and print final stats."""
        if self._tap is not None:
            Quartz.CGEventTapEnable(self._tap, False)
            if self._source is not None:
                Quartz.CFRunLoopRemoveSource(
                    Quartz.CFRunLoopGetMain(),
                    self._source,
                    Quartz.kCFRunLoopCommonModes,
                )
            self._tap = None
            self._source = None

        stats = self._metrics.get_stats()
        if stats["total_clicks"] > 0:
            print(f"\n  Accuracy stats: {stats['total_clicks']} clicks tracked, "
                  f"avg error={stats['avg_error_px']}px, "
                  f"recent avg={stats['recent_avg_error_px']}px")
        self._metrics.close()

    def update_gaze(self, gaze: GazeRatio, screen_point: Point | None = None):
        """Update the current gaze estimate (called each frame)."""
        self._last_gaze = gaze
        self._last_screen_point = screen_point

    def check_refit(self) -> bool:
        """Check if we have enough new click-based points to refit."""
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

    def _on_click(self, event):
        """Handle a captured mouse click event."""
        gaze = self._last_gaze
        predicted = self._last_screen_point
        if gaze is None:
            return

        loc = Quartz.CGEventGetLocation(event)
        click_x, click_y = loc.x, loc.y

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
