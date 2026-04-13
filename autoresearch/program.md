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

**The goal: minimize median_error_px.** Lower = better. Median is used instead
of mean because it's robust to outlier clicks where the user wasn't looking at
the click target.

## Output format

The script prints metrics like:
```
---
median_error_px: 198.3
avg_error_px:    234.5
p90_error_px:    412.7
min_error_px:    12.1
max_error_px:    891.2
n_test_clicks:   45

Per-region median error (3x3 grid):
  TL:   312.4  TC:   198.3  TR:   445.1
  ML:   156.2  MC:   123.4  MR:   234.5
  BL:   567.8  BC:   345.6  BR:   456.7
---
```

Extract the key metric: `grep "^median_error_px:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Header and columns:

```
commit	median_error_px	avg_error_px	status	description
```

1. git commit hash (short, 7 chars)
2. median_error_px (e.g. 198.3) — THE PRIMARY METRIC — use 0.0 for crashes
3. avg_error_px (e.g. 234.5) — use 0.0 for crashes
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
8. If median_error_px improved (lower): keep the commit
9. If median_error_px equal or worse: `git reset --hard` to discard

**NEVER STOP.** The human may be asleep. Each experiment takes ~30-60 seconds,
so you can run ~60-120 experiments per hour.

## Ideas to try (ordered by expected impact)

### HIGH PRIORITY — likely to give biggest accuracy gains

Geometric normalization (THE #1 problem per SOTA research):
- The current iris ratio is computed on raw 2D landmark projections which are
  confounded by head pose. When the head turns, the 2D projection distorts the
  iris-to-corner distance even if gaze direction didn't change.
- Use the rotation matrix from solvePnP to warp/rotate the eye landmarks to a
  canonical frontal view BEFORE computing iris ratios.
- This is what GeoGaze (2026) and 3DPE-Gaze (2025) both do.
- Implementation: apply inverse rotation to the 2D eye landmarks, then compute ratios.

Ridge regression (prevent overfitting):
- Replace np.linalg.lstsq with Ridge regression (L2 regularization).
- Current lstsq with 7 features and ~20-100 noisy points overfits badly.
- Try: `from sklearn.linear_model import Ridge` or implement manually with
  `(A^T A + alpha*I)^-1 A^T y` using numpy.
- Try alpha values: 0.1, 1.0, 10.0, 100.0.

Feature normalization:
- Z-score normalize all features before regression (subtract mean, divide by std).
- Iris ratios are 0-1, head yaw is -15 to +15 degrees — mixing these scales
  without normalization biases the regression toward larger-magnitude features.

3D gaze vector instead of 2D ratios:
- Estimate 3D eyeball center from the face mesh (midpoint of eye corners + offset).
- Compute gaze direction vector from eyeball center through iris center.
- Intersect this ray with the screen plane to get screen coordinates.
- This is geometrically more correct than a 2D ratio mapped through polynomial regression.

### MEDIUM PRIORITY — good incremental improvements

Feature extraction:
- Use more landmark indices (MediaPipe provides 478 total)
- Weight left/right eye differently based on head yaw (the eye closer to the
  camera has better iris visibility)
- Use eye aspect ratio (openness) as an additional feature
- Add face width/height ratio as a distance proxy
- Inter-pupillary distance (changes with viewing distance)
- Nose tip position relative to face center

Calibration:
- Quadratic polynomial (add iris_x^2, iris_y^2, etc.)
- Separate models for horizontal (screen_x) and vertical (screen_y)
- Weighted least squares (weight recent clicks more than old ones)
- Try SVR (Support Vector Regression) instead of linear regression

Head pose:
- Add head roll as a feature (currently only yaw + pitch)
- Use more landmarks for solvePnP (8+ instead of 6 — more stable)
- Different solvePnP solver flags (EPNP, AP3P, SQPNP)
- Different 3D face model dimensions
- Use translation vector (encodes distance from camera)

Smoothing:
- Kalman filter instead of median + EMA
- One-euro filter (jitter-adaptive low-pass filter)
- Adaptive smoothing (more when head is still, less when moving)
- Different median window sizes

### LOWER PRIORITY — bigger changes, try after basics work

Preprocessing:
- Histogram equalization on frames before landmark detection
- CLAHE (contrast-limited adaptive histogram equalization) on eye region
- Different frame resolutions

Appearance-based features:
- Compute histogram of oriented gradients (HOG) on cropped eye region
- Use mean pixel intensity of iris region as additional feature
- Edge detection on eye region for pupil localization
