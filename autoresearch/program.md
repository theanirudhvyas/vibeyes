# vibeyes-autoresearch

Autonomous experimentation to improve VibEyes gaze tracking accuracy.

## Setup

1. Agree on a run tag (e.g. `apr13`). Branch: `autoresearch/<tag>`.
2. Create the branch from current master.
3. Read all files: this file, `prepare.py` (read-only), `pipeline.py` (your file).
4. Verify recordings exist: check `~/.cache/vibeyes/recordings/` has at least one
   session directory with 30+ clicks.
5. Initialize `results.tsv` with the header row.
6. Run baseline: `python prepare.py > run.log 2>&1`
7. Record baseline avg_error_px in results.tsv.

## Experimentation

Each experiment replays recorded webcam frames through your modified pipeline.

**What you CAN modify:**
- `pipeline.py` — this is the only file you edit. Everything is fair game:
  - Which MediaPipe landmarks to use for iris/eye tracking
  - How iris ratios are computed (different eye landmarks, weighted averages, etc.)
  - Head pose estimation (different solvePnP landmarks, 3D model dimensions, solver flags)
  - Feature engineering (quadratic terms, interaction terms, new features from the
    478 available landmarks)
  - Calibration method (ridge regression, SVR, polynomial regression, neural net, etc.)
  - Smoothing strategy (EMA factors, median window, Kalman filter, etc.)
  - Any preprocessing (frame resolution changes, contrast, cropping, etc.)

**What you CANNOT modify:**
- `prepare.py` — the evaluation harness is read-only
- The recorded data (frames and click positions are ground truth)
- You cannot install new packages beyond what's in the vibeyes pyproject.toml
  (mediapipe, opencv-python, numpy are available; also scipy if added)

**The goal: minimize avg_error_px.** Lower = better.

## Output format

The script prints metrics like:
```
---
avg_error_px:    234.5
median_error_px: 198.3
p90_error_px:    412.7
min_error_px:    12.1
max_error_px:    891.2
n_test_clicks:   45
---
```

Extract the key metric: `grep "^avg_error_px:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Header and columns:

```
commit	avg_error_px	median_error_px	status	description
```

1. git commit hash (short, 7 chars)
2. avg_error_px (e.g. 234.5) — use 0.0 for crashes
3. median_error_px (e.g. 198.3) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what the experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at git state and past results in results.tsv
2. Modify `pipeline.py` with an experimental idea
3. git commit
4. Run: `python prepare.py > run.log 2>&1`
5. Read results: `grep "^avg_error_px:\|^median_error_px:" run.log`
6. If grep empty → crash. Check `tail -n 50 run.log` for the error.
7. Record results in results.tsv (do NOT commit results.tsv)
8. If avg_error_px improved (lower): keep the commit
9. If avg_error_px equal or worse: `git reset --hard` to discard

**NEVER STOP.** The human may be asleep. Each experiment takes ~30-60 seconds,
so you can run ~60-120 experiments per hour.

## Ideas to try (not exhaustive)

Feature extraction:
- Use more landmark indices (MediaPipe provides 478 total)
- Weight left/right eye differently based on head yaw
- Use eye aspect ratio (openness) as an additional feature
- Use nose tip position relative to face center
- Compute gaze direction vectors instead of simple ratios

Calibration:
- Quadratic polynomial (add iris_x^2, iris_y^2, etc.)
- Ridge regression (regularization to prevent overfitting)
- Separate models for horizontal and vertical prediction
- Weighted least squares (recent clicks weighted more)

Smoothing:
- Kalman filter instead of median + EMA
- Adaptive smoothing (more smoothing when head is still, less when moving)
- Different median window sizes
- One-euro filter (jitter-adaptive low-pass filter)

Head pose:
- Different solvePnP solver flags (EPNP, AP3P, SQPNP)
- More landmarks for solvePnP (use 8+ instead of 6)
- Different 3D face model dimensions
- Use head roll as an additional feature

Preprocessing:
- Histogram equalization on frames before landmark detection
- Different frame resolutions
