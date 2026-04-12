"""Entry point for VibEyes prototype."""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from vibeyes import GazeRatio, Point
from vibeyes.calibration import Calibration
from vibeyes.camera import Camera
from vibeyes.detector import Detector
from vibeyes.face_tracker import FaceTracker
from vibeyes.gaze_estimator import GazeEstimator
from vibeyes.window_tracker import get_visible_windows

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


def run_calibration(face_tracker: FaceTracker, gaze_estimator: GazeEstimator) -> Calibration:
    """Run 9-point calibration using OpenCV windows."""
    screen_w, screen_h = get_screen_size()
    calibration = Calibration()
    camera = Camera()

    # 3x3 grid of calibration points with 10% margin
    margin_x = int(screen_w * 0.1)
    margin_y = int(screen_h * 0.1)
    points = []
    for row in range(3):
        for col in range(3):
            x = margin_x + col * (screen_w - 2 * margin_x) // 2
            y = margin_y + row * (screen_h - 2 * margin_y) // 2
            points.append((x, y))

    cv2.namedWindow("VibEyes Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("VibEyes Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i, (sx, sy) in enumerate(points):
        # Draw calibration dot
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(canvas, (sx, sy), 20, (0, 255, 0), -1)
        cv2.circle(canvas, (sx, sy), 5, (255, 255, 255), -1)
        cv2.putText(
            canvas,
            f"Look at the dot ({i + 1}/{len(points)}) - Press SPACE",
            (screen_w // 2 - 250, screen_h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )
        cv2.imshow("VibEyes Calibration", canvas)

        # Wait for spacebar, collecting gaze samples
        samples = []
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == 27:  # ESC to abort
                cv2.destroyAllWindows()
                camera.release()
                print("Calibration aborted.")
                sys.exit(1)

            frame = camera.read()
            if frame is None:
                continue
            face_data = face_tracker.detect(frame)
            if face_data is not None:
                ratio = gaze_estimator.estimate(face_data)
                samples.append(ratio)

        # Use median of last N samples for stability
        if len(samples) >= 3:
            recent = samples[-10:]
            median_x = sorted(s.x for s in recent)[len(recent) // 2]
            median_y = sorted(s.y for s in recent)[len(recent) // 2]
            calibration.add_point(GazeRatio(median_x, median_y), Point(sx, sy))
            print(f"  Point {i + 1}: gaze=({median_x:.3f}, {median_y:.3f}) -> screen=({sx}, {sy})")
        else:
            print(f"  Point {i + 1}: not enough samples, skipping")

    cv2.destroyAllWindows()
    camera.release()

    if calibration.point_count < 6:
        print(f"Only got {calibration.point_count} points -- need at least 6. Try again.")
        sys.exit(1)

    calibration.fit()
    calibration.save(CALIBRATION_PATH)
    print(f"Calibration saved to {CALIBRATION_PATH}")
    return calibration


def run_tracking(detector: Detector):
    """Run continuous gaze tracking, printing detected window to terminal."""
    camera = Camera()
    last_window = None
    fps_counter = 0
    fps_start = time.time()

    print("\nTracking started. Press Ctrl+C to stop.\n")

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

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()
            else:
                fps = 0

            if window is not None:
                label = f"{window.app_name}: {window.title}" if window.title else window.app_name
                if label != last_window:
                    print(f"  Looking at: {label}" + (f"  ({fps:.0f} fps)" if fps > 0 else ""))
                    last_window = label
            else:
                if last_window is not None:
                    print("  Looking at: (nothing detected)")
                    last_window = None

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        camera.release()


def main():
    parser = argparse.ArgumentParser(description="VibEyes: Webcam gaze-to-window tracker")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to face_landmarker.task model")
    parser.add_argument("--smoothing", type=float, default=0.4, help="Gaze smoothing factor (0-1)")
    args = parser.parse_args()

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
        calibration = run_calibration(face_tracker, gaze_estimator)
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

    run_tracking(detector)
    face_tracker.close()


if __name__ == "__main__":
    main()
