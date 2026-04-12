"""Transparent gaze overlay using a Cocoa NSWindow (macOS)."""

import sys
import threading

if sys.platform != "darwin":
    raise ImportError("Overlay only supported on macOS")

import Quartz
from Cocoa import (
    NSApplication,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSMakeRect,
    NSScreen,
    NSString,
    NSView,
    NSWindow,
    NSWindowStyleMaskBorderless,
)
from Cocoa import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary
import objc


OVERLAY_SIZE = 30  # diameter of gaze dot
TRAIL_SIZE = 10     # smaller trail dots


class GazeDotView(NSView):
    """Custom NSView that draws a gaze indicator dot."""

    def initWithFrame_(self, frame):
        self = objc.super(GazeDotView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._label = ""
        return self

    def drawRect_(self, rect):
        # Clear background (transparent)
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(rect)

        # Draw outer ring (semi-transparent red)
        ring = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(2, 2, rect.size.width - 4, rect.size.height - 4)
        )
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.2, 0.2, 0.6).set()
        ring.setLineWidth_(3.0)
        ring.stroke()

        # Draw center dot (solid red)
        center_size = 8
        cx = (rect.size.width - center_size) / 2
        cy = (rect.size.height - center_size) / 2
        dot = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx, cy, center_size, center_size)
        )
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.1, 0.1, 0.8).set()
        dot.fill()

    def setLabel_(self, label):
        self._label = label
        self.setNeedsDisplay_(True)


class GazeOverlay:
    """Transparent always-on-top overlay showing where VibEyes estimates gaze position."""

    def __init__(self):
        self._window = None
        self._view = None
        self._running = False

    def start(self):
        """Create the overlay window. Must be called from main thread or after NSApp init."""
        # Ensure NSApplication exists
        NSApplication.sharedApplication()

        screen = NSScreen.mainScreen()
        screen_frame = screen.frame()

        # Create transparent, borderless, always-on-top, click-through window
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, OVERLAY_SIZE, OVERLAY_SIZE),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setLevel_(Quartz.kCGMaximumWindowLevelKey)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
        )

        self._view = GazeDotView.alloc().initWithFrame_(
            NSMakeRect(0, 0, OVERLAY_SIZE, OVERLAY_SIZE)
        )
        self._window.setContentView_(self._view)
        self._window.orderFrontRegardless()
        self._running = True

    def update(self, screen_x: float, screen_y: float):
        """Move the overlay to the given screen coordinates.

        screen_x, screen_y are in macOS screen coordinates (origin at bottom-left
        of main display for Cocoa, but our gaze uses top-left origin).
        """
        if not self._running or self._window is None:
            return

        # Convert from top-left origin (CGWindowList style) to bottom-left (Cocoa style)
        screen = NSScreen.mainScreen()
        screen_height = screen.frame().size.height
        cocoa_x = screen_x - OVERLAY_SIZE / 2
        cocoa_y = screen_height - screen_y - OVERLAY_SIZE / 2

        self._window.setFrameOrigin_((cocoa_x, cocoa_y))

    def stop(self):
        """Remove the overlay window."""
        if self._window is not None:
            self._window.orderOut_(None)
            self._window = None
        self._running = False
