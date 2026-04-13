# VibEyes

Webcam-based gaze tracking that detects which macOS window (and zellij pane) you're looking at. No extra hardware -- just your laptop's built-in camera.

VibEyes uses MediaPipe's 478-point face mesh to track your iris position and head pose, maps them to screen coordinates via a calibrated regression model, and hit-tests against macOS windows to determine where your visual attention is. For terminal users, it parses zellij's layout to identify individual panes within a split.

The system self-improves: every mouse click is treated as ground truth ("you look where you click") and fed back into the calibration model. An autoresearch subsystem can replay recorded sessions through a modifiable pipeline, letting an AI agent autonomously experiment with different feature extraction, calibration, and smoothing approaches to minimize prediction error.

**Status:** Working prototype (Python). Gaze-to-window detection works, accuracy is being improved via autoresearch. See [PRD.md](PRD.md) for the full product vision including gesture interaction and cross-platform plans.

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

### Roadmap

#### Phase 1: Gaze-Aware Window Detection (MVP) -- in progress

| Feature | Status |
|---------|--------|
| Webcam capture + camera selection with preview | Done |
| MediaPipe face/iris landmark detection | Done |
| 3D head pose estimation (solvePnP) | Done |
| Gaze estimation (iris ratio + head pose) | Done |
| Blink detection (freeze gaze during blinks) | Done |
| 16-point randomized calibration flow | Done |
| Implicit click-based recalibration | Done |
| Polynomial regression (gaze -> screen coords) | Done |
| Median filter + EMA smoothing pipeline | Done |
| Window hit-testing (CGWindowListCopyWindowInfo) | Done |
| Zellij pane detection within terminals | Done |
| Transparent overlay gaze dot | Done |
| Dwell-time hysteresis (stable window switching) | Done |
| SQLite accuracy metrics tracking | Done |
| Frame recording for offline replay | Done |
| Autoresearch autonomous accuracy optimizer | Done |
| macOS permission handling (Camera, Accessibility) | Done |
| Improve gaze accuracy (currently 750-2200px error) | Pending |
| L2CS-Net or similar dedicated gaze estimation model (ONNX) | Pending |
| Posture-invariant tracking (lap vs desk vs standing) | Pending |
| Auto-detect posture change and trigger recalibration | Pending |
| Camera angle normalization (perspective correction) | Pending |
| External webcam support (different FOVs and positions) | Pending |
| Menu bar app with camera status indicator | Pending |
| Auto-raise gazed window after sustained attention | Pending |
| Attention analytics (time per app) | Pending |
| Text cursor tracking as calibration signal | Pending |
| >80% window detection accuracy target | Pending |

#### Phase 2: Gesture Interaction

| Feature | Status |
|---------|--------|
| Facial gesture detection (blink, nod, shake, wink) | Pending |
| Hand gesture detection (MediaPipe Hand Landmarker) | Pending |
| Customizable gesture-to-action mapping | Pending |
| Per-application gesture profiles | Pending |
| Gesture calibration (personal thresholds) | Pending |
| Compound gestures for destructive actions | Pending |

#### Phase 3: Intelligence & Integrations

| Feature | Status |
|---------|--------|
| Attention analytics dashboard (daily/weekly reports) | Pending |
| Focus mode (detect scattered gaze, suggest focus) | Pending |
| API/SDK for other apps to query gaze position | Pending |
| Browser extension (gaze within browser tabs) | Pending |
| Multi-monitor support | Pending |
| Shortcuts/Raycast integration | Pending |

#### Phase 4: Cross-Platform

| Feature | Status |
|---------|--------|
| Port to Tauri v2 + Rust (production app) | Pending |
| Linux desktop app (X11, then Wayland) | Pending |
| Browser-based version (onnxruntime-web) | Pending |
