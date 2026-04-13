# Autoresearch Experiment Backlog

Running list of ideas for the autonomous agent to try. Ordered by expected impact. Agent should work top-down but can skip if an idea seems unpromising after reading the code.

## Status Key
- **untried** -- not yet attempted
- **tried:keep** -- tried, improved metric, kept
- **tried:discard** -- tried, no improvement, discarded
- **tried:crash** -- tried, crashed, needs debugging
- **blocked** -- needs something else first (noted in comments)

---

## Tier 1: High Impact (try first)

### Geometric Normalization
- [ ] **Frontalize eye landmarks before iris ratio** -- Use solvePnP rotation matrix to rotate the 2D eye landmarks to a canonical frontal view, THEN compute iris ratios. This removes head-pose confounding from the gaze signal. This is what GeoGaze (2026) does. Status: untried
- [ ] **Frontalize using all 478 landmarks** -- Instead of just eye landmarks, normalize all landmark positions by the head rotation. Gaze features become pose-invariant. Status: untried

### Calibration Method
- [ ] **Ridge regression (L2 regularization)** -- Replace `np.linalg.lstsq` with Ridge regression. Prevents overfitting on noisy features. Try alpha=0.1, 1.0, 10.0, 100.0. Can implement without sklearn: `(A^T A + alpha*I)^-1 A^T y`. Status: untried
- [ ] **Feature normalization (z-score)** -- Subtract mean, divide by std for each feature column before regression. Iris ratios are 0-1, head angles are -15 to +15 -- mixing these scales biases the fit. Status: untried
- [ ] **Separate X and Y models** -- Fit two independent models: one for screen_x, one for screen_y. Different features may matter for horizontal vs vertical prediction. Status: untried

### 3D Gaze Geometry
- [ ] **3D gaze vector intersection** -- Estimate 3D eyeball center (midpoint of eye corners + depth offset from face mesh). Compute gaze ray from eyeball center through iris center. Intersect with screen plane. Geometrically correct approach. Status: untried
- [ ] **Use 3D landmark coordinates** -- MediaPipe returns `.z` for each landmark. Use the 3D (x,y,z) coordinates instead of just 2D (x,y) for eye geometry. Status: untried

---

## Tier 2: Medium Impact (try after Tier 1)

### Feature Engineering
- [ ] **Add head roll** -- Currently only yaw and pitch. Roll captures head tilt. Status: untried
- [ ] **Add solvePnP translation vector** -- tvec encodes (x,y,z) position of face relative to camera. The z component is distance from camera, which affects iris ratio scale. Status: untried
- [ ] **Weight eyes by head yaw** -- When head turns right, left eye is more visible and has better iris data. Weight accordingly: `w_left = 0.5 + yaw/60, w_right = 1 - w_left`. Status: untried
- [ ] **Eye aspect ratio as feature** -- How open the eye is affects iris visibility. Add EAR as a calibration feature. Status: untried
- [ ] **Face width/height ratio** -- Proxy for face distance and angle. Status: untried
- [ ] **Inter-pupillary distance** -- Distance between iris centers. Changes with viewing distance. Status: untried
- [ ] **Nose tip relative to face center** -- The offset of nose tip from the midpoint of eyes encodes head rotation differently than solvePnP. Status: untried
- [ ] **More landmarks for solvePnP** -- Use 8-10 instead of 6 for more stable pose estimation. Status: untried
- [ ] **Different solvePnP solver** -- Try SOLVEPNP_EPNP, SOLVEPNP_AP3P, SOLVEPNP_SQPNP. Status: untried
- [ ] **Quadratic features** -- Add iris_x^2, iris_y^2, head_yaw^2, head_pitch^2 to feature matrix. Status: untried

### Smoothing
- [ ] **One-euro filter** -- Jitter-adaptive low-pass filter. Less lag than EMA for slow movement, more smoothing for jitter. Status: untried
- [ ] **Kalman filter** -- Proper state estimation with prediction + correction. Handles velocity. Status: untried
- [ ] **Tune median window** -- Try sizes 3, 5, 9, 11 instead of 7. Status: untried
- [ ] **Adaptive smoothing** -- More smoothing when head is still (low pose variance), less when moving (user is looking somewhere new). Status: untried
- [ ] **Remove smoothing entirely in replay** -- Smoothing is for real-time display. In replay, each frame is independent. Removing smoothing may give more honest calibration data. Status: untried

### Calibration Variants
- [ ] **Weighted least squares** -- Weight recent calibration points more than old ones. Exponential decay weights. Status: untried
- [ ] **SVR (Support Vector Regression)** -- Non-linear regression that's more robust to outliers than polynomial. Status: untried
- [ ] **Per-region calibration** -- Split screen into quadrants, fit separate models. Status: untried

---

## Tier 3: Lower Impact / Bigger Effort

### Preprocessing
- [ ] **CLAHE on eye region** -- Contrast-limited adaptive histogram equalization may improve iris detection in poor lighting. Status: untried
- [ ] **Histogram equalization on full frame** -- Global contrast normalization. Status: untried
- [ ] **Process at higher resolution** -- Try 1280x720 instead of 640x480. More pixels on eye = better iris detection, but slower. Status: untried

### Appearance Features
- [ ] **HOG on eye crop** -- Histogram of oriented gradients captures eye texture patterns. Status: untried
- [ ] **Mean iris region intensity** -- Bright spot on iris from screen reflection could indicate gaze direction. Status: untried
- [ ] **Eye crop pixel features** -- Feed raw cropped eye pixels (resized to 32x32) as additional features. Requires PCA or autoencoder to reduce dimensionality. Status: untried

### Alternative Models
- [ ] **Per-user fine-tuned gaze CNN** -- Train a small CNN on the user's own calibration data. Needs enough data (200+ clicks). Status: untried
- [ ] **Random forest regression** -- Ensemble method, handles non-linear relationships. Status: untried
- [ ] **KNN regression** -- Simple non-parametric method. May work well with enough calibration data. Status: untried

---

## Ideas from External Research

- **3DPE-Gaze (NeurIPS 2025)**: Uses 3D facial landmarks as strong geometric priors. ~1-2 degree angular error.
- **GazeSymCAT (2025)**: Symmetric cross-attention transformer over face + eye crops. SOTA on ETH-XGaze.
- **Gaze-LLE (CVPR 2025)**: Frozen DINOv2/ViT scene encoder + lightweight decoder. SOTA gaze target estimation.
- **GeoGaze (2026)**: MediaPipe 478-point mesh + normalized iris ratios, 66fps CPU, no training.
- **pperle/gaze-tracking**: Lightweight CNN, ~2.4 degree with 128 calibration samples.
