# vibeyes-autoresearch

Autonomous experimentation to improve VibEyes gaze tracking for window focus
detection. The end goal is not pixel-perfect gaze — it's correctly identifying
which window the user is looking at.

## Setup

1. Agree on a run tag (e.g. `apr13`). Branch: `autoresearch/<tag>`.
2. Create the branch from current master.
3. Read all files: this file, `prepare.py` (read-only), `pipeline.py` (your file).
4. Verify recordings exist: check `~/.cache/vibeyes/recordings/` has at least one
   session directory with 30+ clicks.
5. Initialize `results.tsv` with the header row.
6. Run baseline 3 times: `python prepare.py > run.log 2>&1` to measure variance.
7. Record baseline metrics in results.tsv. Note the variance between runs —
   only keep changes that improve by MORE than the baseline noise.
8. Read `EXPERIMENTS.md` for the prioritized backlog of ideas already tried.

## What we're optimizing

VibEyes detects which **window** the user is looking at, not which pixel.
The evaluation harness reports multiple metrics — use them all to make decisions:

### Primary metric: `hit_rate_quadrant`
The percentage of test clicks where the predicted gaze falls in the same
**screen quadrant** (2x2 grid) as the actual click. This directly measures
the system's ability to distinguish which area of the screen the user is
looking at.

### Secondary metrics (also important):
- `hit_rate_2col` — correct half of screen (left vs right). Simulates
  2 side-by-side windows.
- `hit_rate_3col` — correct third of screen. Simulates 3 column layout.
- `median_error_px` — pixel distance error. Still useful for measuring
  overall accuracy, but has diminishing returns for window detection.
  (Going from 270px to 200px only improves quadrant hit rate by ~2%.)
- `median_angle_deg` — angular error in degrees. Industry standard metric.
  For context: WebGazer.js ~4°, SeeSo (commercial) ~1.7°, Tobii (IR) ~0.5°.
- Per-region breakdown — shows where the model is weakest.

### How to decide keep vs discard
Keep an experiment if ANY of these are true:
- `hit_rate_quadrant` improved (even by 0.5%)
- `hit_rate_3col` improved AND `median_error_px` didn't get worse
- `median_error_px` improved by more than 5px

Discard if hit rates stayed the same AND median_error_px didn't improve.

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
median_angle_deg: 1.76
avg_angle_deg:    2.08
hit_rate_2col:   95.0%
hit_rate_3col:   82.0%
hit_rate_quadrant: 80.0%

Per-region median error (3x3 grid):
  TL:   312.4  TC:   198.3  TR:   445.1
  ML:   156.2  MC:   123.4  MR:   234.5
  BL:   567.8  BC:   345.6  BR:   456.7
---
```

Extract metrics: `grep "^median_error_px:\|^hit_rate" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Header and columns:

```
commit	median_error_px	hit_rate_quadrant	status	description
```

1. git commit hash (short, 7 chars)
2. median_error_px (e.g. 198.3) — use 0.0 for crashes
3. hit_rate_quadrant (e.g. 0.80) — THE PRIMARY METRIC — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what the experiment tried

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
- `prepare.py` — the evaluation harness
- The recorded data (frames and click positions are ground truth)
- You cannot install new packages beyond what's in the vibeyes pyproject.toml
  (mediapipe, opencv-python, numpy are available; also scipy if added)

## The experiment loop

LOOP FOREVER:

1. Look at git state and past results in results.tsv
2. **Research** (see Research-Driven Strategy below) — search for relevant
   techniques before coding. Form a hypothesis grounded in published methods.
3. Modify `pipeline.py` with the experimental idea
4. git commit
5. Run: `python prepare.py > run.log 2>&1`
6. Read results: `grep "^median_error_px:\|^hit_rate" run.log`
7. If grep empty → crash. Check `tail -n 50 run.log` for the error.
8. Record results in results.tsv (do NOT commit results.tsv)
9. If improved (see "How to decide keep vs discard" above): keep the commit
10. If not improved: `git reset --hard` to discard

**NEVER STOP.** The human may be asleep. Each experiment takes ~30-60 seconds,
so you can run ~60-120 experiments per hour.

## Understanding the problem

Before grinding on feature tweaks, understand what actually limits accuracy:

### The data distribution problem
Calibration data is heavily left-biased. Many sessions have zero clicks on the
right side of the screen. The model can't predict regions it's never seen.
Check the per-region breakdown — any region with n/a or >500px error is a
data gap, not a model problem. No feature engineering can fix missing data.

### Diminishing returns of pixel accuracy
At ~270px median error (~2.4°), going from 270px to 200px only improves
window hit rate by ~2%. The marginal value of pixel improvements is tiny.
Focus on the catastrophic failures (>1000px errors in right/bottom regions)
and the hit rate metrics, not small pixel improvements.

### What actually moves the needle
From 80+ experiments across two runs, the biggest improvements came from:
1. **Fixing fundamental bugs** (removing smoothing in replay: -428px)
2. **Adding genuinely informative features** (EAR: -97px combined)
3. **Simplifying noisy features** (avg EAR beats separate L/R EAR)
4. **Cross-axis features** (abs_y in X model: -10px)
5. **More calibration data** (baseline improved as user recorded more clicks)

What DIDN'T help: polynomial features, nonlinear models (neural net, SVR),
CLAHE preprocessing, 3D landmarks, frontalization, upscaling frames.

### The real ceiling
The linear ridge model with MediaPipe landmarks has been squeezed nearly dry.
To break through, you need fundamentally different approaches (see below).

## Research-Driven Strategy

**Do not just try random feature tweaks.** Before each batch of experiments, use
Perplexity (web search) to find state-of-the-art techniques. The goal is to bring
published research into this pipeline, not to guess blindly.

### When to research

- **At the start of each run** — before your first experiment, do 3-5 searches
  to understand the current landscape. What are the best webcam-only gaze tracking
  systems achieving? What techniques do they use?
- **When you hit a plateau** — if 5+ consecutive experiments fail to improve,
  stop and research. Search for the specific problem you're stuck on.
- **Before trying a new category of change** — before attempting a new approach,
  search for how others have done it in gaze tracking specifically.

### What to search for

Use Perplexity MCP (`mcp__perplexity-mcp__perplexity_search_web`) for all
web searches. Specific topics to research:

1. **State of the art in webcam gaze tracking** — Search for:
   - "state of the art webcam eye gaze tracking 2025 2026 accuracy"
   - "best open source gaze estimation single camera no special hardware"
   - Look for: GeoGaze, L2CS-Net, WebEyeTrack, BlazeGaze, ETH-XGaze, MPIIGaze

2. **Geometric gaze estimation** — The most promising direction. Search for:
   - "3D eyeball center estimation from face landmarks gaze ray screen intersection"
   - "iris ratio normalization head pose compensation gaze tracking"
   - "GeoGaze MediaPipe iris ratio frontalization technique"

3. **Calibration and regression methods for gaze** — Search for:
   - "personal calibration gaze estimation few-shot ridge SVR random forest"
   - "gaze calibration polynomial regression vs neural network vs SVR accuracy"

4. **Window/AOI classification vs regression** — Search for:
   - "gaze tracking optimize for area of interest hit rate classification"
   - "gaze contingent interaction window detection accuracy needed"
   - Consider: should the pipeline predict screen regions directly instead
     of pixel coordinates?

5. **Smoothing and temporal filtering** — Search for:
   - "one euro filter eye tracking implementation parameters"
   - "Kalman filter gaze estimation prediction correction"

6. **Specific problems you observe** — When per-region errors suggest a
   pattern, search for that specific issue:
   - "gaze estimation accuracy worse at screen edges vs center"
   - "head pose confounding iris position gaze tracking fix"
   - "cross-session gaze calibration generalization"

### How to use research findings

1. **Identify the technique** — What specific algorithm or approach?
2. **Map to our pipeline** — How does this translate to `pipeline.py` changes?
3. **Estimate feasibility** — Can we implement with numpy/scipy/opencv only?
4. **Design the experiment** — One variable at a time.
5. **Record the source** — In commit message, note what inspired it.

### Research budget

Spend ~5 minutes researching per ~30 minutes of experimentation. Research when:
- Starting a new direction
- Stuck after 5+ failures
- Moving to a fundamentally different approach

## Strategy

### Priority 1: Fix catastrophic failures
Regions with >500px error (typically right side, bottom) are where window
classification actually fails. A prediction that's 1200px off always hits
the wrong window. Focus here first — the hit rate will jump.

### Priority 2: Try fundamentally different approaches
The linear ridge model is at its ceiling. Directions that could break through:

- **3D gaze ray geometry** — Estimate a 3D gaze vector from eyeball center
  through iris, intersect with screen plane. Used by WebEyeTrack (2025) and
  geometric VOG systems. Research implementation details first.

- **Iris ratio frontalization** — GeoGaze (2026) normalizes iris ratios by
  removing head-pose distortion via the solvePnP rotation matrix. Research
  "GeoGaze iris ratio normalization" for specifics.

- **Hybrid calibration** — WebEyeTrack uses a 3D geometric model for coarse
  gaze + a lightweight learned model for refinement. Search for details.

- **Ellipse fitting for iris** — Fit an ellipse to iris boundary landmarks.
  The ellipse orientation encodes gaze direction independent of head pose.

- **Region classification** — Instead of predicting pixel coordinates and
  hoping they land in the right window, train a classifier that directly
  predicts which screen region the user is looking at.

### Priority 3: Incremental improvements (diminishing returns)
Feature engineering, alpha tuning, outlier threshold tweaks. These gave
~1-5px improvements in previous runs. Still worth trying but don't expect
breakthroughs.

### Tactics
- **Ablation first** — before adding complexity, check if removing things helps.
- **Analyze errors** — per-region breakdown shows where the model fails.
- **Small changes** — one variable per experiment.
- **Research before building** — search for how others solved the problem.
- **Watch the hit rates** — a 1% improvement in hit_rate_quadrant is more
  valuable than a 10px improvement in median_error_px.
