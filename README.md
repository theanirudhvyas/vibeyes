# VibEyes

Webcam-based gaze tracking that detects which macOS window (and zellij pane) you're looking at.

## Quick Start

```bash
make setup          # Create venv, install deps, download model
make calibrate      # 16-point calibration (look at dots, press SPACE)
make run-overlay    # Track with gaze dot overlay
```

## Commands

| Command | What it does |
|---------|-------------|
| `make setup` | Install Python 3.12 venv + dependencies + MediaPipe model |
| `make calibrate` | Run 16-point calibration with overlay |
| `make run` | Track gaze, print detected window to terminal |
| `make run-overlay` | Track with translucent gaze dot on screen |
| `make run-debug` | Track with overlay + debug output (gaze ratios, screen coords) |
| `make record` | Track + record frames for autoresearch (same as run-debug) |
| `make list-cameras` | List available cameras |
| `make test` | Run all tests |
| `make metrics` | Show accuracy stats from click tracking |
| `make autoresearch-recordings` | List recorded sessions and click counts |
| `make autoresearch-baseline` | Run autoresearch baseline evaluation |

### Camera Selection

If you have multiple cameras, specify which one:

```bash
make calibrate CAMERA=1
make run-overlay CAMERA=1
```

## How It Works

```
Webcam (640x480, 30fps)
  -> MediaPipe Face Landmarker (478 landmarks + iris)
  -> Iris ratio (position within eye bounds)
  -> 3D head pose via solvePnP (yaw, pitch)
  -> Polynomial calibration (gaze features -> screen coords)
  -> Median filter + EMA smoothing
  -> CGWindowListCopyWindowInfo -> window hit-test
  -> Zellij pane detection (if terminal window)
```

### Calibration

1. **Initial**: 16 randomized dots, look at each + press SPACE
2. **Implicit**: Every mouse click refines the model (you look where you click)
3. **Persistent**: Calibration + click data saved across sessions

### Permissions Required

- **Camera**: prompted automatically on first run
- **Accessibility**: prompted automatically for click tracking (System Settings > Privacy & Security > Accessibility)

## Autoresearch

An autonomous experimentation system that replays recorded webcam frames through a modifiable gaze pipeline to optimize accuracy.

```bash
# 1. Record data (click around normally for 30+ minutes)
make record

# 2. Check recordings
make autoresearch-recordings

# 3. Run baseline
make autoresearch-baseline

# 4. Start the autonomous loop (in a separate Claude Code session)
cd autoresearch && read program.md
```

See `autoresearch/program.md` for full details.

## Project Structure

```
src/vibeyes/
  main.py              # Entry point + CLI
  camera.py            # Webcam capture + camera selection
  face_tracker.py      # MediaPipe face/iris landmark detection + 3D head pose
  gaze_estimator.py    # Iris ratio + head pose -> gaze features
  calibration.py       # Polynomial regression gaze -> screen mapping
  detector.py          # Pipeline orchestrator (median + EMA smoothing)
  window_tracker.py    # macOS window enumeration + hit testing
  pane_tracker.py      # Zellij pane detection from layout dump
  overlay.py           # Transparent Cocoa NSWindow gaze dot
  click_calibrator.py  # CGEventTap click monitoring + implicit recalibration
  frame_recorder.py    # Save webcam frames on click for offline replay
  metrics.py           # SQLite accuracy tracking (predicted vs actual)

autoresearch/
  prepare.py           # Read-only evaluation harness
  pipeline.py          # Agent-editable gaze pipeline
  program.md           # Autonomous experiment loop instructions

tests/                 # 52 tests (pytest)
models/                # MediaPipe face_landmarker.task (gitignored)
PRD.md                 # Product requirements document
```
