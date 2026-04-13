"""Entry point for VibEyes prototype."""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from vibeyes import GazeRatio, Point
from vibeyes.calibration import Calibration
from vibeyes.camera import Camera, list_cameras, select_camera_interactive
from vibeyes.click_calibrator import ClickCalibrator
from vibeyes.detector import Detector
from vibeyes.face_tracker import FaceTracker
from vibeyes.gaze_estimator import GazeEstimator
from vibeyes.pane_tracker import get_zellij_layout, parse_active_tab_panes, hit_test_pane
from vibeyes.window_tracker import get_visible_windows

# Terminal apps where we should try zellij pane detection
TERMINAL_APPS = {"Alacritty", "alacritty", "kitty", "WezTerm", "iTerm2", "Terminal"}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "calibration.json")


def get_screen_size() -> tuple[int, int]:
    """Get the main screen resolution."""
    try:
        import Quartz
        main = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(main.size.width), int(main.size.height)
    except ImportError:
        return 1920, 1080


def _generate_calibration_points(screen_w: int, screen_h: int, n_points: int = 16) -> list[tuple[int, int]]:
    """Generate randomized calibration points covering the full screen.

    Starts with a 4x4 grid, then shuffles for randomized order.
    Extra points are added randomly within the margins.
    """
    import random

    margin_x = int(screen_w * 0.05)
    margin_y = int(screen_h * 0.05)
    usable_w = screen_w - 2 * margin_x
    usable_h = screen_h - 2 * margin_y

    # Base grid: 4x4 = 16 points
    grid_cols = 4
    grid_rows = 4
    points = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = margin_x + int(col * usable_w / (grid_cols - 1))
            y = margin_y + int(row * usable_h / (grid_rows - 1))
            points.append((x, y))

    # If more points requested, add random ones
    while len(points) < n_points:
        x = random.randint(margin_x, screen_w - margin_x)
        y = random.randint(margin_y, screen_h - margin_y)
        points.append((x, y))

    # Shuffle so the order is random (prevents systematic drift bias)
    random.shuffle(points)
    return points[:n_points]


def run_calibration(face_tracker: FaceTracker, gaze_estimator: GazeEstimator, camera_device: int = 0, n_points: int = 16) -> Calibration:
    """Run calibration using OpenCV windows."""
    screen_w, screen_h = get_screen_size()
    calibration = Calibration()
    camera = Camera(device=camera_device)

    # Warm up camera
    print("Warming up camera...")
    for _ in range(30):
        camera.read()
        time.sleep(0.03)

    points = _generate_calibration_points(screen_w, screen_h, n_points)
    print(f"Calibrating with {len(points)} randomized points...")

    cv2.namedWindow("VibEyes Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("VibEyes Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i, (sx, sy) in enumerate(points):
        samples = []
        face_detected = False

        while True:
            # Read camera frame
            frame = camera.read()
            if frame is not None:
                face_data = face_tracker.detect(frame)
                if face_data is not None:
                    face_detected = True
                    ratio = gaze_estimator.estimate(face_data)
                    samples.append(ratio)
                else:
                    face_detected = False

            # Draw calibration UI with live status
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            # Draw the target dot
            cv2.circle(canvas, (sx, sy), 20, (0, 255, 0), -1)
            cv2.circle(canvas, (sx, sy), 5, (255, 255, 255), -1)

            # Show instructions
            cv2.putText(
                canvas,
                f"Look at the green dot ({i + 1}/{len(points)}) then press SPACE",
                (screen_w // 2 - 300, screen_h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
            )

            # Show face detection status
            if face_detected:
                status_color = (0, 255, 0)
                status_text = f"Face detected - {len(samples)} samples"
            else:
                status_color = (0, 0, 255)
                status_text = "No face detected - make sure camera can see your face"
            cv2.putText(
                canvas, status_text,
                (screen_w // 2 - 300, screen_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
            )

            # Show small camera preview in corner
            if frame is not None:
                preview = cv2.resize(frame, (160, 120))
                canvas[10:130, screen_w - 170:screen_w - 10] = preview

            cv2.imshow("VibEyes Calibration", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and len(samples) >= 3:
                break
            if key == ord(" ") and len(samples) < 3:
                # Don't accept until we have enough samples
                pass
            if key == 27:  # ESC to abort
                cv2.destroyAllWindows()
                camera.release()
                print("Calibration aborted.")
                sys.exit(1)

        # Use median of last N samples for stability
        recent = samples[-15:]
        median_x = sorted(s.x for s in recent)[len(recent) // 2]
        median_y = sorted(s.y for s in recent)[len(recent) // 2]
        calibration.add_point(GazeRatio(median_x, median_y), Point(sx, sy))
        print(f"  Point {i + 1}: gaze=({median_x:.3f}, {median_y:.3f}) -> screen=({sx}, {sy})")

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    camera.release()

    if calibration.point_count < 6:
        print(f"Only got {calibration.point_count} points -- need at least 6. Try again.")
        sys.exit(1)

    calibration.fit()
    calibration.save(CALIBRATION_PATH)
    print(f"Calibration saved to {CALIBRATION_PATH}")
    return calibration


def run_tracking(detector: Detector, camera_device: int = 0, dwell_time: float = 0.5,
                 show_overlay: bool = False, debug: bool = False, click_cal: bool = True):
    """Run continuous gaze tracking, printing detected window to terminal."""
    camera = Camera(device=camera_device)
    confirmed_label = None
    candidate_label = None
    candidate_since = 0.0

    fps_counter = 0
    fps_start = time.time()

    # Click-based implicit calibration
    click_calibrator = None
    if click_cal:
        try:
            click_calibrator = ClickCalibrator(detector.calibration)
            click_calibrator.start()
            print("Click calibration enabled (clicks refine accuracy over time)")
        except Exception as e:
            print(f"Could not start click calibrator: {e}")

    # Gaze overlay
    overlay = None
    if show_overlay:
        try:
            from vibeyes.overlay import GazeOverlay
            overlay = GazeOverlay()
            overlay.start()
            print("Gaze overlay enabled (red dot on screen)")
        except Exception as e:
            print(f"Could not start overlay: {e}")

    # Cache zellij layout (refresh periodically)
    zellij_panes = None
    zellij_layout_time = 0

    print(f"\nTracking started (dwell={dwell_time}s). Press Ctrl+C to stop.\n")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            try:
                window = detector.detect(frame)
            except RuntimeError as e:
                print(f"Error: {e}")
                break

            # Feed current gaze to click calibrator and process click events
            if click_calibrator and detector.last_gaze_ratio:
                click_calibrator.update_gaze(detector.last_gaze_ratio, detector.last_screen_point)
                click_calibrator.pump()
                click_calibrator.check_refit()

            # Update overlay with raw gaze position (before hysteresis)
            if overlay and detector.last_screen_point:
                overlay.update(detector.last_screen_point.x, detector.last_screen_point.y)

            fps_counter += 1
            now = time.time()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = now

                # Debug output: print raw values once per second
                if debug and detector.last_gaze_ratio and detector.last_screen_point:
                    gr = detector.last_gaze_ratio
                    sp = detector.last_screen_point
                    print(f"  [debug] gaze=({gr.x:.3f}, {gr.y:.3f}) screen=({sp.x:.0f}, {sp.y:.0f}) fps={fps:.0f}")
            else:
                fps = 0

            if window is None:
                # Require dwell to switch to "nothing"
                raw_label = None
            else:
                label = f"{window.app_name}: {window.title}" if window.title else window.app_name
                pane_label = ""

                # If the window is a terminal, try to detect zellij pane
                if window.app_name in TERMINAL_APPS:
                    # Refresh zellij layout every 2 seconds
                    if now - zellij_layout_time > 2.0:
                        layout = get_zellij_layout()
                        if layout:
                            zellij_panes = parse_active_tab_panes(layout)
                        else:
                            zellij_panes = None
                        zellij_layout_time = now

                    if zellij_panes:
                        screen_point = detector.last_screen_point
                        if screen_point:
                            wx, wy, ww, wh = window.bounds
                            pane = hit_test_pane(
                                screen_point.x, screen_point.y,
                                zellij_panes,
                                window_x=wx, window_y=wy,
                                window_w=ww, window_h=wh,
                            )
                            if pane:
                                pane_cmd = pane.command.split("/")[-1]  # basename
                                pane_label = f" > pane: {pane_cmd}"
                                if pane.cwd:
                                    pane_cwd = pane.cwd.split("/")[-1]
                                    pane_label += f" ({pane_cwd})"

                raw_label = label + pane_label

            # Dwell-time hysteresis: only switch if new target is stable
            if raw_label == confirmed_label:
                # Still on same target -- reset candidate
                candidate_label = None
            elif raw_label == candidate_label:
                # Candidate is consistent -- check dwell time
                if now - candidate_since >= dwell_time:
                    confirmed_label = candidate_label
                    candidate_label = None
                    if confirmed_label:
                        print(f"  Looking at: {confirmed_label}" + (f"  ({fps:.0f} fps)" if fps > 0 else ""))
                    else:
                        print("  Looking at: (nothing detected)")
            else:
                # New candidate -- start timing
                candidate_label = raw_label
                candidate_since = now

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if click_calibrator:
            click_calibrator.stop()
            # Save updated calibration with click data
            if detector.calibration.is_calibrated:
                detector.calibration.save(CALIBRATION_PATH)
                print(f"  Calibration saved ({detector.calibration.point_count} points)")
        if overlay:
            overlay.stop()
        camera.release()


def main():
    parser = argparse.ArgumentParser(description="VibEyes: Webcam gaze-to-window tracker")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to face_landmarker.task model")
    parser.add_argument("--smoothing", type=float, default=0.3, help="Gaze smoothing factor (0-1, lower=smoother, default 0.3)")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (use --list-cameras to see available)")
    parser.add_argument("--dwell", type=float, default=0.5, help="Seconds gaze must stay on new target before switching (default 0.5)")
    parser.add_argument("--overlay", action="store_true", help="Show a red dot overlay where gaze is estimated")
    parser.add_argument("--debug", action="store_true", help="Print raw gaze/screen values every second")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_cameras()
        if not cameras:
            print("No cameras found.")
        else:
            print("Available cameras:")
            for cam in cameras:
                print(f"  [{cam['index']}] {cam['name']}")
        sys.exit(0)

    # Camera selection
    if args.camera is not None:
        camera_device = args.camera
    else:
        cameras = list_cameras()
        if len(cameras) == 0:
            print("Error: No cameras found.")
            sys.exit(1)
        elif len(cameras) == 1:
            camera_device = cameras[0]["index"]
            print(f"Using camera {camera_device}")
        else:
            print(f"Found {len(cameras)} cameras. Showing preview of each...")
            camera_device = select_camera_interactive(cameras)

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Download it: curl -L -o models/face_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        sys.exit(1)

    print("Initializing face tracker...")
    face_tracker = FaceTracker(model_path)
    gaze_estimator = GazeEstimator(smoothing_factor=args.smoothing)

    if args.calibrate:
        print("Starting calibration...")
        calibration = run_calibration(face_tracker, gaze_estimator, camera_device=camera_device)
    else:
        if os.path.exists(CALIBRATION_PATH):
            print(f"Loading calibration from {CALIBRATION_PATH}")
            calibration = Calibration.load(CALIBRATION_PATH)
        else:
            print("No calibration found. Run with --calibrate first.")
            face_tracker.close()
            sys.exit(1)

    detector = Detector(
        face_tracker=face_tracker,
        gaze_estimator=gaze_estimator,
        calibration=calibration,
        get_windows=lambda: get_visible_windows(exclude_app="Python"),
    )

    run_tracking(detector, camera_device=camera_device, dwell_time=args.dwell, show_overlay=args.overlay, debug=args.debug)
    face_tracker.close()


if __name__ == "__main__":
    main()
