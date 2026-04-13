# VibEyes

Webcam-based gaze tracking that detects which macOS window (and zellij pane) you're looking at.

## Setup

```bash
make setup          # Create venv, install deps, download model
make calibrate      # 16-point calibration (look at dots, press SPACE)
make run-overlay    # Track with gaze dot overlay
```

Multiple cameras? Use `make calibrate CAMERA=1` (shows preview to pick the right one).

## Developer Setup

```bash
make setup          # venv + deps + model
make test           # 52 tests, all should pass
make run-debug      # overlay + debug output (gaze ratios, screen coords, fps)
make metrics        # show click accuracy stats from SQLite
```

To collect data for the autoresearch accuracy optimizer:
```bash
make run-overlay                 # use normally, frames auto-record on every click
make autoresearch-recordings     # check click counts (need 30+)
make autoresearch-baseline       # run baseline evaluation
cd autoresearch && claude        # start autonomous experiment loop
```

See `autoresearch/program.md` for the full experiment loop spec.

## Reference

### Pipeline

```
Webcam (640x480) -> MediaPipe Face Landmarker (478 landmarks + iris)
  -> Iris ratio + 3D head pose (solvePnP yaw/pitch)
  -> Polynomial calibration -> Median filter + EMA smoothing
  -> CGWindowListCopyWindowInfo hit-test -> Zellij pane detection
```

### Calibration

- **Initial**: 16 randomized dots on screen, look + press SPACE for each
- **Implicit**: every mouse click silently refines the model (you look where you click)
- **Persistent**: saved across sessions to `calibration.json`
- **Blink-safe**: gaze freezes during blinks (Eye Aspect Ratio detection)

### Permissions

Both prompted automatically on first run:
- **Camera** -- core functionality
- **Accessibility** -- click tracking for implicit calibration

### All Make Targets

Run `make` to see the full list. Key ones:

| Target | Purpose |
|--------|---------|
| `make setup` | Install everything |
| `make calibrate` | 16-point calibration |
| `make run` | Track, terminal output only |
| `make run-overlay` | Track with gaze dot |
| `make run-debug` | Track + overlay + debug values |
| `make test` | Run pytest suite |
| `make metrics` | Click accuracy stats |
| `make autoresearch-baseline` | Evaluate pipeline on recorded data |

### Project Structure

```
src/vibeyes/
  main.py              Entry point + CLI
  camera.py            Webcam capture + camera selection
  face_tracker.py      MediaPipe landmarks + 3D head pose (solvePnP)
  gaze_estimator.py    Iris ratio + head pose -> gaze features + blink detection
  calibration.py       Polynomial regression (gaze -> screen coords)
  detector.py          Pipeline orchestrator (median + EMA smoothing)
  window_tracker.py    macOS window enumeration (Quartz) + hit testing
  pane_tracker.py      Zellij pane detection (parses dump-layout)
  overlay.py           Transparent Cocoa NSWindow gaze dot
  click_calibrator.py  CGEventTap click capture + implicit recalibration
  frame_recorder.py    Save frames on click for offline replay
  metrics.py           SQLite accuracy tracking (predicted vs actual)

autoresearch/
  prepare.py           Read-only eval harness (loads recordings, measures error)
  pipeline.py          Agent-editable gaze pipeline (modify to improve accuracy)
  program.md           Autonomous experiment loop instructions

tests/                 52 tests (pytest)
models/                face_landmarker.task (gitignored, downloaded by make setup)
```
