# vibeyes-autoresearch

Autonomous experimentation to improve VibEyes gaze tracking accuracy.

## Setup

1. Agree on a run tag (e.g. `apr13`). Branch: `autoresearch/<tag>`.
2. Create the branch from current master.
3. Read all files: this file, `prepare.py` (read-only), `pipeline.py` (your file).
4. Verify recordings exist: check `~/.cache/vibeyes/recordings/` has at least one
   session directory with 30+ clicks.
5. Initialize `results.tsv` with the header row.
6. Run baseline 3 times: `python prepare.py > run.log 2>&1` to measure variance.
7. Record baseline median_error_px in results.tsv. Note the variance between runs --
   only keep changes that improve by MORE than the baseline noise.
8. Read `EXPERIMENTS.md` for the prioritized backlog of ideas already tried.

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
2. **Research** (see Research-Driven Strategy below) — search for relevant
   techniques before coding. Form a hypothesis grounded in published methods.
3. Modify `pipeline.py` with the experimental idea
4. git commit
5. Run: `python prepare.py > run.log 2>&1`
6. Read results: `grep "^avg_error_px:\|^median_error_px:" run.log`
7. If grep empty → crash. Check `tail -n 50 run.log` for the error.
8. Record results in results.tsv (do NOT commit results.tsv)
9. If median_error_px improved (lower): keep the commit
10. If median_error_px equal or worse: `git reset --hard` to discard

**NEVER STOP.** The human may be asleep. Each experiment takes ~30-60 seconds,
so you can run ~60-120 experiments per hour.

## Research-Driven Strategy

**Do not just try random feature tweaks.** Before each batch of experiments, use
Perplexity (web search) to find state-of-the-art techniques. The goal is to bring
published research into this pipeline, not to guess blindly.

### When to research

- **At the start of each run** — before your first experiment, do 3-5 searches
  to understand the current landscape. What are the best webcam-only gaze tracking
  systems achieving? What techniques do they use?
- **When you hit a plateau** — if 5+ consecutive experiments fail to improve,
  stop and research. Search for the specific problem you're stuck on (e.g.,
  "how to reduce gaze estimation error in the center-screen region").
- **Before trying a new category of change** — before attempting a new approach
  (e.g., switching from ridge to SVR, adding a Kalman filter), search for how
  others have done it in gaze tracking specifically.

### What to search for

Use Perplexity MCP (`mcp__perplexity-mcp__perplexity_search_web`) for all
web searches. Specific topics to research:

1. **State of the art in webcam gaze tracking** — What accuracy do the best
   systems achieve? What are the key techniques? Search for:
   - "state of the art webcam eye gaze tracking 2025 2026 accuracy"
   - "best open source gaze estimation single camera no special hardware"
   - Look for: GeoGaze, L2CS-Net, WebEyeTrack, BlazeGaze, ETH-XGaze, MPIIGaze

2. **Geometric gaze estimation** — The most promising direction for our
   MediaPipe-based approach. Search for:
   - "3D eyeball center estimation from face landmarks gaze ray screen intersection"
   - "iris ratio normalization head pose compensation gaze tracking"
   - "GeoGaze MediaPipe iris ratio frontalization technique"
   - The core idea: estimate a 3D gaze ray from eyeball center through iris,
     intersect with screen plane. This is geometrically correct vs our current
     2D ratio → regression approach.

3. **Calibration and regression methods for gaze** — Search for:
   - "personal calibration gaze estimation few-shot ridge SVR random forest"
   - "gaze calibration polynomial regression vs neural network vs SVR accuracy"
   - Our calibration has ~80-200 points per session — what model fits best?

4. **Smoothing and temporal filtering** — Search for:
   - "one euro filter eye tracking implementation parameters"
   - "Kalman filter gaze estimation prediction correction"
   - "adaptive smoothing saccade detection gaze tracking"
   - Note: smoothing is for the LIVE system, not replay. But understanding
     temporal dynamics helps with understanding noise patterns.

5. **Feature engineering for landmark-based gaze** — Search for:
   - "MediaPipe face mesh gaze features which landmarks carry gaze signal"
   - "eye aspect ratio pupil position head pose features gaze estimation"
   - "iris-to-nose vector face geometry gaze direction"

6. **Specific problems you observe** — When per-region errors suggest a
   pattern, search for that specific issue:
   - "gaze estimation accuracy worse at screen edges vs center"
   - "head pose confounding iris position gaze tracking fix"
   - "cross-session gaze calibration generalization"

### How to use research findings

After searching, synthesize what you learned into a concrete experiment:

1. **Identify the technique** — What specific algorithm or approach did the
   paper/project use?
2. **Map to our pipeline** — How does this translate to changes in `pipeline.py`?
   What features to add/remove? What model to use?
3. **Estimate feasibility** — Can we implement this with numpy/scipy/opencv only?
   Does it need the full 478 landmarks? Is it too complex for a single experiment?
4. **Design the experiment** — One variable at a time. If the technique involves
   3 changes, try them separately.
5. **Record the source** — In the git commit message and results.tsv description,
   note what research inspired the experiment (e.g., "per GeoGaze frontalization
   approach" or "WebEyeTrack 3D gaze ray method").

### Research budget

Spend ~5 minutes researching per ~30 minutes of experimentation. Don't research
every single experiment — batch related experiments together. Research when:
- Starting a new direction
- Stuck after 5+ failures
- Moving to a fundamentally different approach

## Strategy

Read `EXPERIMENTS.md` for a backlog of ideas already tried, but don't treat it
as a checklist. Use your own judgment: read the current pipeline code, analyze
the per-region error breakdown, and form hypotheses about what's limiting accuracy.
The best experiments come from understanding *why* the current approach fails,
not from trying ideas in order.

Useful tactics:
- **Ablation first** — before adding complexity, check if removing things helps.
  Drop features, simplify the model, see what actually matters.
- **Analyze errors** — the per-region breakdown tells you where the model is
  weakest. Focus experiments on the worst regions.
- **Small changes** — one variable per experiment. If you change three things and
  the metric improves, you don't know which one helped.
- **Research before building** — search Perplexity for how others solved the
  specific problem you're attacking. Don't reinvent solutions that exist.

### Key directions from published research

These are high-level directions grounded in the current SOTA. Research the
specifics before implementing:

- **3D gaze ray geometry** — Instead of mapping 2D iris ratios through regression,
  estimate a 3D gaze vector (from eyeball center through iris) and intersect with
  the screen plane. This is the approach used by WebEyeTrack (2025) and geometric
  VOG systems. Search for implementation details before attempting.

- **Iris ratio frontalization** — GeoGaze (2026) normalizes iris ratios by
  removing head-pose distortion before feeding to the model. Uses the solvePnP
  rotation matrix to warp eye landmarks to a canonical frontal view. Search for
  "GeoGaze iris ratio normalization" for specifics.

- **Hybrid calibration** — WebEyeTrack uses a 3D geometric model for coarse
  gaze + a lightweight learned model (few-shot calibrated) for refinement.
  Search for "hybrid geometric appearance gaze estimation calibration."

- **EdgeGauss / One-Euro smoothing** — Temporal filtering that preserves saccades
  while removing jitter. One-Euro filter adapts its cutoff frequency based on
  signal velocity. Search for implementation and parameter tuning for gaze data.

- **Ellipse fitting for iris** — Instead of using MediaPipe's single iris center
  point, fit an ellipse to the iris boundary landmarks and use ellipse parameters
  (center, axes, orientation) as features. The ellipse orientation encodes gaze
  direction independent of head pose.

- **Pupil-cornea offset** — The offset between the pupil center and corneal
  reflection encodes gaze direction. MediaPipe doesn't give corneal reflections,
  but the z-coordinate of iris landmarks relative to eye surface landmarks
  captures a similar signal.
