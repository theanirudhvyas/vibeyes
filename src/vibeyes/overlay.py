"""Transparent gaze overlay using a Cocoa NSWindow (macOS)."""

import sys

if sys.platform != "darwin":
    raise ImportError("Overlay only supported on macOS")

import Quartz
from Cocoa import (
    NSApplication,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSDate,
    NSMakeRect,
    NSRunLoop,
    NSScreen,
    NSView,
    NSWindow,
    NSWindowStyleMaskBorderless,
)
from Cocoa import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary
import objc


OVERLAY_SIZE = 30


class GazeDotView(NSView):
    """Custom NSView that draws a gaze indicator dot."""

    def initWithFrame_(self, frame):
        self = objc.super(GazeDotView, self).initWithFrame_(frame)
        return self

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(rect)

        # Outer ring
        ring = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(2, 2, rect.size.width - 4, rect.size.height - 4)
        )
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.2, 0.2, 0.7).set()
        ring.setLineWidth_(3.0)
        ring.stroke()

        # Center dot
        cs = 8
        cx = (rect.size.width - cs) / 2
        cy = (rect.size.height - cs) / 2
        dot = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx, cy, cs, cs))
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.1, 0.1, 0.9).set()
        dot.fill()


class GazeOverlay:
    """Transparent always-on-top overlay showing estimated gaze position."""

    def __init__(self):
        self._window = None
        self._app = None
        self._screen_height = 0

    def start(self):
        """Create the overlay window."""
        self._app = NSApplication.sharedApplication()
        # Activate as accessory (no dock icon, no menu bar)
        self._app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory

        screen = NSScreen.mainScreen()
        self._screen_height = screen.frame().size.height

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100, 100, OVERLAY_SIZE, OVERLAY_SIZE),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setLevel_(Quartz.kCGScreenSaverWindowLevel + 1)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setAlphaValue_(1.0)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
        )

        view = GazeDotView.alloc().initWithFrame_(
            NSMakeRect(0, 0, OVERLAY_SIZE, OVERLAY_SIZE)
        )
        self._window.setContentView_(view)
        self._window.makeKeyAndOrderFront_(None)
        self._window.orderFrontRegardless()

        # Pump the event loop once to actually show the window
        self._pump()

    def update(self, screen_x: float, screen_y: float):
        """Move the overlay dot to the given screen coordinates (top-left origin)."""
        if self._window is None:
            return

        # Convert from top-left origin to Cocoa bottom-left origin
        cocoa_x = screen_x - OVERLAY_SIZE / 2
        cocoa_y = self._screen_height - screen_y - OVERLAY_SIZE / 2

        self._window.setFrameOrigin_((cocoa_x, cocoa_y))
        self._window.display()
        self._pump()

    def stop(self):
        """Remove the overlay."""
        if self._window is not None:
            self._window.orderOut_(None)
            self._pump()
            self._window = None

    def _pump(self):
        """Process pending Cocoa events so the window actually renders."""
        run_loop = NSRunLoop.currentRunLoop()
        until = NSDate.dateWithTimeIntervalSinceNow_(0.001)
        run_loop.runUntilDate_(until)
