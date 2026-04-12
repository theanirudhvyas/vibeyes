# VibEyes - Product Requirements Document

**Version:** 0.1 (Draft)
**Date:** 2026-04-13
**Status:** Research & Discovery

---

## Table of Contents

1. [Vision & Problem Statement](#1-vision--problem-statement)
2. [Competitive Landscape](#2-competitive-landscape)
3. [Technical Feasibility Analysis](#3-technical-feasibility-analysis)
4. [Product Phases & Feature Scope](#4-product-phases--feature-scope)
5. [Architecture & Technology Stack](#5-architecture--technology-stack)
6. [Key Technical Risks & Mitigations](#6-key-technical-risks--mitigations)
7. [Privacy & Permissions](#7-privacy--permissions)
8. [Performance Budget](#8-performance-budget)
9. [UX Design Principles](#9-ux-design-principles)
10. [Go-to-Market Considerations](#10-go-to-market-considerations)
11. [Open Questions](#11-open-questions)

---

## 1. Vision & Problem Statement

### The Problem

Modern knowledge workers operate across dozens of windows, tabs, and applications simultaneously. Switching context -- clicking to activate a window, alt-tabbing, using trackpad gestures -- creates constant micro-friction. More importantly, there is no passive way to know which application is actually receiving a user's attention vs. which one happens to be "in focus."

### The Vision

**VibEyes** is a macOS-first desktop application that uses the built-in webcam to track where the user is looking on screen, determining which window has their visual attention. In later phases, it enables hands-free interaction through facial gestures -- allowing users to scroll, switch tabs, or trigger actions without touching their keyboard or mouse.

### Core Value Propositions

1. **Attention awareness** -- Know which window/app you're actually looking at, enabling smarter window management, focus tracking, and productivity analytics
2. **Hands-free interaction** -- Use facial and hand gestures to interact with the attended window (scroll, switch tabs, click) without touching peripherals
3. **Accessibility** -- Enable people with motor impairments to navigate between applications using eye gaze, facial gestures, and hand gestures
4. **Privacy-first** -- All processing happens locally on-device; no camera data ever leaves the machine

### Target Users

| Segment | Use Case |
|---------|----------|
| **Power users / developers** | Gaze-aware window management across multi-monitor setups |
| **Productivity enthusiasts** | Attention analytics ("I spent 3 hours looking at Slack today") |
| **Accessibility users** | Hands-free navigation between windows and applications |
| **Streamers / content creators** | Show audience where they're looking in real-time |

---

## 2. Competitive Landscape

### Direct Competitors (Gaze-to-Window Focus)

**This is an unserved market.** No shipping product on macOS specifically determines which window a user is looking at using webcam-based gaze tracking. The closest products are:

| Product | What It Does | Platform | Limitation |
|---------|-------------|----------|------------|
| **Tobii Aware** | Detects which display you're viewing in multi-monitor setups, auto-dims/locks | Windows (OEM only) | Requires Tobii IR hardware, OEM-bundled only, Windows-only, display-level not window-level |
| **Apple Eye Tracking** (macOS Sequoia) | OS-level eye tracking for accessibility -- dwell-to-click, gaze navigation | macOS 15+ | Accessibility-only, no developer API for gaze coordinates, not designed for window focus detection |
| **Apple Head Pointer** | Moves cursor via head movement tracked by FaceTime camera | macOS | Head tracking only (not eye), deliberate head movement required |

### Adjacent Competitors (Eye/Face Tracking Software)

| Product | Approach | Platform | Pricing | Relevance |
|---------|----------|----------|---------|-----------|
| **Eyeware Beam** | iPhone TrueDepth camera streams gaze data to desktop | iOS -> Win/Mac | Free / $5/mo | Closest UX concept but requires iPhone as input device |
| **Talon Voice** | Hands-free coding via voice + Tobii eye tracker | Win/Mac/Linux | Free | Best-in-class accessibility but requires $230 Tobii hardware |
| **Cephable** | Facial expressions + head movements mapped to keyboard/mouse | Win/Mac | Free/Paid | Gesture control but no gaze estimation, no hand gestures |
| **Project Gameface** (Google) | Facial gestures for mouse/keyboard control | Android | Free/OSS | Similar gesture approach but mobile-only, no gaze, no hand gestures |
| **GazePointer** | Webcam-based gaze moves mouse cursor | Windows | Free | Webcam gaze but Windows-only, outdated |
| **Smyle Mouse** | Head movement + smile-to-click | Windows | ~$100 | Gesture control, Windows-only |

### Open Source Projects

| Project | What It Does | Last Active | Usefulness |
|---------|-------------|-------------|------------|
| **MediaPipe Face Mesh** (Google) | 478 face/iris landmarks, real-time, cross-platform | Active | **Core dependency** -- best starting point for face/iris detection |
| **OpenFace 2.0** (CMU) | Gaze estimation, head pose, facial action units | Active | Excellent reference for gaze estimation pipeline |
| **WebGazer.js** (Brown University) | Browser-based webcam eye tracking | Active | Reference for browser target; implicit calibration approach |
| **L2CS-Net** | Lightweight gaze estimation CNN | Active | Strong candidate for gaze estimation model |
| **ETH-XGaze** (ETH Zurich) | Large-scale gaze estimation dataset + model | Active | Training data / model for gaze estimation |
| **GazeTracking** (Python) | Basic gaze direction detection via OpenCV/dlib | Low maintenance | Reference only; too basic for production |
| **OpenTrack** | Head tracking for flight sims | Active | Head pose tracking reference only |
| **PyGaze** | Python wrapper for various eye trackers | Active | Research tool, not directly usable |

### Key Insight: Market Gap

No product exists that:
- Uses a standard webcam (no extra hardware)
- Runs natively on macOS
- Determines which **window** (not just which monitor) you're looking at
- Offers facial gesture interaction with the attended window
- Processes everything locally (privacy-first)

This is VibEyes' opportunity.

---

## 3. Technical Feasibility Analysis

### 3.1 Gaze Estimation: What's Realistic with a Webcam

The fundamental question: **Can a standard webcam determine which window you're looking at?**

**Yes, with caveats.**

| Detection Granularity | Expected Accuracy | Feasibility |
|----------------------|-------------------|-------------|
| Which **monitor** (in multi-monitor setup) | >95% | Highly feasible |
| Which **screen half** (left/right) | >90% | Highly feasible |
| Which **screen quadrant** | 80-90% | Feasible |
| Which **window** (typical size) | 70-85% | Feasible for reasonably-sized windows |
| Which **UI element** within a window | Depends on element size | Only for large, well-separated elements |
| **Pixel-precise** gaze point | Not realistic | Not feasible with webcam only |

**Angular accuracy achievable:**
- Without calibration: 4-8 degrees (~200-400px on a 24" 1080p monitor at 60cm)
- With 9-point calibration: 2-4 degrees (~100-200px)
- With calibration + temporal smoothing: 1.5-3 degrees (~75-150px)

**For window-level detection, this is sufficient.** Most windows on a typical desktop are 300-1000+ pixels wide. Even at 3-degree accuracy (~150px error), the system can reliably determine which window the user is looking at, especially with temporal smoothing (requiring sustained gaze rather than instantaneous detection).

### 3.2 How Gaze Estimation Works (Webcam-Based)

**Recommended pipeline:**

```
Webcam Frame (640x480, 30fps)
    |
    v
[Face Detection + 478 Landmarks] -- MediaPipe Face Mesh
    |
    v
[Iris Center + Eye Corner Landmarks] -- 5 iris points + eye contour per eye
    |
    v
[Head Pose Estimation] -- solvePnP from 3D face model
    |
    v
[Gaze Vector Estimation] -- Iris position relative to eye corners,
    |                        corrected for head pose
    v
[Screen Coordinate Mapping] -- Calibrated polynomial regression
    |
    v
[Window Hit Testing] -- CGWindowListCopyWindowInfo bounds matching
    |
    v
"User is looking at: VS Code"
```

**Two approaches for gaze estimation:**

| Approach | How It Works | Accuracy | Speed |
|----------|-------------|----------|-------|
| **Geometric** | Compute 3D gaze vector from iris position + eye model + head pose, intersect with screen plane | 2-4 degrees (calibrated) | ~5ms |
| **Appearance-based (deep learning)** | Feed face/eye crops to a CNN (L2CS-Net, ETH-XGaze) that directly predicts gaze angles | 3-5 degrees (cross-person) | ~10-15ms on CPU, ~5ms on GPU |
| **Hybrid (recommended)** | Use geometric approach for coarse estimation, refine with lightweight learned model | 1.5-3 degrees | ~10ms |

### 3.3 Calibration

Calibration maps raw gaze features to screen coordinates. This is essential for window-level accuracy.

**Recommended: 9-point calibration**
- User looks at 9 dots shown sequentially on screen
- System records iris positions / gaze features at each point
- Polynomial regression fits the mapping:
  ```
  Screen_X = a0 + a1*gx + a2*gy + a3*gx^2 + a4*gy^2 + a5*gx*gy
  Screen_Y = b0 + b1*gx + b2*gy + b3*gx^2 + b4*gy^2 + b5*gx*gy
  ```
- Takes ~15-20 seconds

**Implicit recalibration:** Mouse clicks can serve as implicit calibration points (user generally looks where they click). This allows the system to drift-correct over time without explicit recalibration sessions. WebGazer.js demonstrates this approach works.

**Recalibration frequency:** Every 10-20 minutes of active use, or when the system detects accuracy degradation (e.g., gaze consistently offset from clicked locations).

### 3.4 Gesture Recognition Feasibility

VibEyes supports two gesture categories: **facial gestures** (detected via the same face mesh used for gaze tracking) and **hand gestures** (detected via MediaPipe Hand Landmarker). Both use the same webcam feed, so hand gestures come "for free" once the camera is active.

#### Facial Gestures

**High-confidence (ship in v1):**

| Gesture | Detection Method | Reliability | False Positive Rate |
|---------|-----------------|-------------|-------------------|
| **Blink (both eyes)** | Eye Aspect Ratio < 0.2 for 100-400ms | >98% | <1% |
| **Deliberate long blink** | Eyes closed 500-1500ms | >95% | <2% |
| **Head nod** | Head pitch change >10 degrees down then up | >90% | 2-5% |
| **Head shake** | Head yaw oscillation >15 degrees | >90% | 2-5% |
| **Mouth open** | Lip distance ratio > threshold for 300ms+ | >95% | 1-3% |

**Medium-confidence (v2, with user calibration):**

| Gesture | Detection Method | Reliability | False Positive Rate |
|---------|-----------------|-------------|-------------------|
| **Wink (single eye)** | Asymmetric EAR, one eye < 0.2 while other > 0.3 | 85-95% | 3-8% |
| **Eyebrow raise** | Eyebrow-to-eye distance increase | 85-93% | 2-5% |
| **Smile** | Mouth corner distance / blendshape coefficient | 85-95% | 3-5% |

#### Hand Gestures

MediaPipe Hand Landmarker detects 21 landmarks per hand (up to 2 hands), enabling rich gesture recognition from a standard webcam. The hand just needs to be visible to the camera (raised near the face/chest area works well at a desk).

**High-confidence (ship in v1-v2):**

| Gesture | Detection Method | Reliability | False Positive Rate |
|---------|-----------------|-------------|-------------------|
| **Open palm (stop)** | All fingers extended, palm facing camera | >95% | <2% |
| **Fist (grab/select)** | All fingers curled, no fingers extended | >93% | <3% |
| **Thumbs up** | Thumb extended, other fingers curled | >90% | 2-4% |
| **Thumbs down** | Thumb down, other fingers curled | >88% | 3-5% |
| **Victory / peace sign** | Index + middle extended, others curled | >92% | 2-4% |
| **Point up/down** | Index finger extended + direction | >90% | 3-5% |
| **Pinch (zoom)** | Thumb-index distance < threshold | >88% | 3-6% |

**Medium-confidence (with calibration):**

| Gesture | Detection Method | Reliability | False Positive Rate |
|---------|-----------------|-------------|-------------------|
| **Swipe left/right** | Hand movement direction + velocity tracking | 80-90% | 5-8% |
| **Swipe up/down** | Hand movement direction + velocity tracking | 80-90% | 5-8% |
| **Pinch-and-drag** | Pinch detected + hand translation | 75-85% | 5-10% |
| **Finger count (1-5)** | Count extended fingers | 85-92% | 3-7% |
| **Wave** | Oscillating hand movement pattern | 80-88% | 5-8% |

#### Hand Gesture Technical Considerations

- **Camera FOV:** Built-in MacBook cameras have a ~65-degree FOV. Hands are only detectable when raised into the camera's view (roughly chest-to-head height at a desk). Users may need to adjust to keeping hands slightly raised.
- **Simultaneous tracking:** Running both face mesh and hand landmark detection increases CPU load. MediaPipe can run both models but budget ~15-20ms total on CPU (vs ~10ms for face only).
- **Disambiguation:** The system must distinguish intentional gestures from resting hand positions. Require gestures to be held for 300ms+ or involve movement to avoid false triggers.
- **One-hand vs two-hand:** Start with single-hand gestures. Two-hand gestures (e.g., pinch-to-zoom with both hands) are possible but add complexity and are harder to perform at a desk.

#### Combining Facial + Hand Gestures

The richest interaction model combines both: use gaze to select the target, facial gestures for quick/subtle actions (blink to confirm, nod to scroll), and hand gestures for more expressive commands (swipe to switch apps, pinch to zoom, open palm to pause). This gives users a natural vocabulary where:
- **Facial gestures** = low-effort, subtle, fast (good for frequent actions)
- **Hand gestures** = higher-effort, more expressive, more precise (good for occasional/complex actions)

**Key design principle:** Use **compound gestures** (across modalities) for destructive/important actions (e.g., point + nod to close a window) to minimize accidental triggers. Simple gestures for safe, reversible actions (e.g., long blink to scroll).

### 3.5 Challenges & Constraints

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Head movement** | Primary source of gaze error | Head pose compensation, frequent implicit recalibration |
| **Glasses** | 1-2 degree additional error, reflections | Train on glasses-wearing data, IR not an option |
| **Lighting variation** | Affects iris detection accuracy | Adaptive processing, user guidance for setup |
| **Camera quality** | Lower resolution = fewer eye pixels | Minimum 720p required, 1080p recommended |
| **Viewing distance** | >80cm significantly degrades accuracy | Guide user to optimal 40-70cm range |
| **Calibration drift** | Posture changes invalidate calibration | Implicit recalibration via mouse clicks |
| **Natural face movements** | Talking, yawning can trigger facial gestures | Temporal filtering, cooldown periods, compound gestures |
| **Hand visibility** | Hands must be in camera FOV for hand gestures | Guide users on optimal hand position; facial gestures as fallback |
| **CPU load with hand tracking** | Running face + hand models together increases load | Lazy-load hand model only when hand gesture mode is enabled |

---

## 4. Product Phases & Feature Scope

### Phase 1: Gaze-Aware Window Detection (MVP)

**Goal:** Determine which window the user is looking at and surface this information.

**Features:**
- Menu bar app with camera status indicator
- 9-point calibration flow (first launch + on-demand)
- Real-time gaze region estimation (which area of screen)
- Window hit-testing: map gaze to window using `CGWindowListCopyWindowInfo`
- Visual indicator showing which window VibEyes thinks you're looking at (subtle highlight/border)
- "Gaze focus" mode: optionally auto-raise the gazed window after sustained attention (e.g., 1.5 seconds)
- Basic attention analytics: time spent looking at each application
- Settings: sensitivity, auto-raise delay, enable/disable

**Success criteria:**
- >80% accuracy on window detection for windows >400px wide
- <100ms end-to-end latency (gaze to window highlight)
- <5% CPU usage during steady-state tracking on Apple Silicon

### Phase 2: Gesture Interaction

**Goal:** Enable hands-free interaction with the attended window through facial and hand gestures.

**Features:**
- Dual gesture detection engine: facial gestures (face mesh) + hand gestures (hand landmark model)
- Customizable gesture-to-action mapping:
  - *Facial:* Long blink -> scroll down, Double blink -> scroll up, Head nod -> activate/click, Head shake -> go back, Wink (left/right) -> switch tabs, Mouth open -> open command palette
  - *Hand:* Open palm -> pause/stop, Thumbs up -> confirm, Swipe left/right -> switch apps/tabs, Point up/down -> scroll, Pinch -> zoom, Fist -> grab/select
- Per-application gesture profiles (different gestures for different apps)
- Gesture calibration: user trains their own gesture thresholds for both face and hand
- Visual/audio feedback when gesture is recognized
- Gesture cooldown periods to prevent accidental triggers
- Toggle between face-only, hand-only, or combined gesture modes

**Success criteria:**
- >90% gesture recognition accuracy for high-confidence facial gestures
- >85% gesture recognition accuracy for high-confidence hand gestures
- <5% false positive rate for any gesture
- Gestures feel responsive (<200ms from gesture completion to action)

### Phase 3: Intelligence & Integrations

**Goal:** Make VibEyes contextually smart and integrate with the ecosystem.

**Features:**
- Attention analytics dashboard: daily/weekly reports on where visual attention goes
- "Focus mode": detect when user's gaze is scattered (frequent window switching) and suggest focus
- API/SDK for other apps to query "where is the user looking?"
- Browser extension: gaze tracking within browser tabs (which tab has attention)
- Multi-monitor support: track gaze across monitors
- Shortcuts/Raycast integration
- Customizable automation triggers (e.g., "if I look at Slack for >5 seconds, mark as read")

### Phase 4: Cross-Platform

**Goal:** Bring VibEyes to Linux and the browser.

**Features:**
- Linux desktop app (X11 first, Wayland where possible)
- Browser-based version (subset of features, no window detection)
- Shared gaze estimation models across all platforms

---

## 5. Architecture & Technology Stack

### 5.1 Recommended Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **App framework** | **Tauri v2** (Rust backend + web frontend) | Small footprint (~15MB), Rust for CV performance, web UI for settings/dashboard |
| **Frontend** | **SvelteKit** | Lightweight, fast rendering, excellent DX |
| **CV/ML runtime** | **ONNX Runtime** via `ort` crate | Cross-platform inference, CoreML execution provider on macOS for Neural Engine |
| **Face/iris detection** | **MediaPipe Face Mesh** exported to ONNX | 478 landmarks including iris, well-documented, proven |
| **Hand detection** | **MediaPipe Hand Landmarker** exported to ONNX | 21 landmarks per hand, up to 2 hands, cross-platform |
| **Gaze estimation** | **L2CS-Net** or custom model (ONNX) | Lightweight, good accuracy, exportable |
| **Camera capture** | **`nokhwa`** crate (Rust) | Cross-platform: AVFoundation (macOS), V4L2 (Linux) |
| **Window tracking** | **`CGWindowListCopyWindowInfo`** (macOS) | Native API, returns all window bounds/names/owners |
| **Overlay** | **Transparent Tauri window** | Always-on-top, click-through, for gaze indicators |
| **Browser target** | **onnxruntime-web** + MediaPipe JS | Same models, WASM/WebGL runtime |

### 5.2 Architecture Diagram

```
+------------------------------------------------------------------+
|                         VibEyes (Tauri v2)                        |
|                                                                    |
|  +-------------------------+    +------------------------------+  |
|  |     Web Frontend        |    |       Rust Backend           |  |
|  |     (SvelteKit)         |    |                              |  |
|  |                         |    |  +------------------------+  |  |
|  |  - Calibration UI       |    |  |   Camera Thread        |  |  |
|  |  - Settings panel       |    |  |   (nokhwa)             |  |  |
|  |  - Analytics dashboard  |    |  |   640x480 @ 30fps      |  |  |
|  |  - Gesture config       |    |  +----------+-------------+  |  |
|  |                         |    |             | frames          |  |
|  +----------+--------------+    |             v                 |  |
|             ^                   |  +------------------------+  |  |
|             |  Tauri IPC        |  |   CV Pipeline Thread   |  |  |
|             | (events/commands) |  |                        |  |  |
|             +-------------------+  |  Face Mesh (ONNX)      |  |  |
|                                 |  |  Hand Landmarks (ONNX) |  |  |
|                                 |  |      |                 |  |  |
|                                 |  |  Gaze Estimation       |  |  |
|                                 |  |      |                 |  |  |
|                                 |  |  Gesture Detection     |  |  |
|                                 |  |  (face + hand)         |  |  |
|                                 |  |      |                 |  |  |
|                                 |  |  Screen Coord Mapping  |  |  |
|                                 |  +----------+-------------+  |  |
|                                 |             |                 |  |
|                                 |             v                 |  |
|                                 |  +------------------------+  |  |
|                                 |  |   Window Tracker       |  |  |
|                                 |  |   (CGWindowList)       |  |  |
|                                 |  |   -> Hit test gaze     |  |  |
|                                 |  |   -> Emit events       |  |  |
|                                 |  +------------------------+  |  |
|                                 |                              |  |
|                                 |  +------------------------+  |  |
|                                 |  |   Overlay Manager      |  |  |
|                                 |  |   (transparent window) |  |  |
|                                 |  +------------------------+  |  |
|                                 +------------------------------+  |
+------------------------------------------------------------------+
```

### 5.3 Threading Model

```
Main Thread          -- Tauri event loop, UI rendering
Camera Thread        -- Frame capture (producer), ring buffer
CV Pipeline Thread   -- Face detection + gaze estimation (consumer)
Window Tracker       -- Polls window positions every 500ms
```

Use a `tokio::sync::watch` channel between camera and CV threads -- this automatically drops intermediate frames, ensuring the CV thread always processes the latest frame.

### 5.4 Why Not Other Stacks

| Stack | Rejection Reason |
|-------|-----------------|
| **Electron** | Too heavy for always-on app (150-300MB RAM baseline, ~120MB bundle size) |
| **Swift-only** | No cross-platform path to Linux or browser |
| **Flutter Desktop** | Too many layers of indirection for real-time CV; FFI overhead for every frame |
| **Python + PyQt** | Distribution nightmare (200-500MB bundles), GIL limits true parallelism |
| **Qt/C++** | Slower development velocity than Rust+Tauri, LGPL licensing complications |

### 5.5 Alternative Worth Considering: Rust + egui

If the UI is kept simple (settings panel + overlay, no complex dashboard), a pure Rust app using **egui** is simpler:

- No web frontend, no Tauri IPC overhead
- egui's immediate-mode rendering aligns well with real-time video apps
- Compiles to WASM for browser target
- Fewer moving parts

**Trade-off:** Less UI capability than a web frontend. Consider this for the MVP if rapid iteration on the UI isn't critical.

---

## 6. Key Technical Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Webcam gaze accuracy insufficient for window detection** | Medium | Critical | Prototype early (Phase 0 in Python). Use temporal smoothing + probabilistic window assignment. Fall back to "region" mode if per-window is unreliable. |
| 2 | **Calibration is annoying / users won't do it** | High | High | Implement implicit calibration (mouse clicks as training data). Make explicit calibration fast (<15 seconds) and infrequent. |
| 3 | **High CPU/battery drain for always-on tracking** | Medium | High | Adaptive frame rate (reduce to 5fps when idle). Neural Engine acceleration on Apple Silicon. Process at 640x480 not full resolution. |
| 4 | **False positive facial gestures during normal face movement (talking, etc.)** | High | Medium | Temporal filtering, cooldown periods, compound gestures for important actions. Let users disable specific gestures. |
| 4b | **Hand gestures not visible in camera FOV** | Medium | Medium | Guide users on hand placement. Facial gestures always available as fallback. Consider suggesting external webcam with wider FOV. |
| 5 | **Glasses degrade accuracy significantly** | Medium | Medium | Test extensively with glasses wearers. Consider offering "glasses mode" with relaxed thresholds and larger gaze zones. |
| 6 | **`nokhwa` or `ort` Rust crates are immature** | Low-Medium | Medium | Prototype the camera+inference pipeline in Rust early (Phase 2). Have fallback plan to use Swift bridge for camera (AVFoundation) if nokhwa has issues. |
| 7 | **Users uncomfortable with always-on camera** | Medium | High | Clear privacy messaging. Camera indicator always visible. Easy on/off toggle. Open-source the CV pipeline for auditability. |
| 8 | **Wayland (Linux) lacks window enumeration APIs** | High (for Linux) | Low (macOS first) | Defer Linux to Phase 4. Support X11 first. Add compositor-specific Wayland support later. |

### Recommended: Phase 0 -- Python Prototype (1-2 weeks)

Before committing to the full Rust/Tauri stack, validate the core hypothesis with a throwaway Python prototype:

```python
# Validate: can we reliably detect which window the user is looking at?
MediaPipe Face Mesh -> iris landmarks -> gaze estimation -> calibration -> 
CGWindowListCopyWindowInfo -> window hit test
```

**Success criteria for prototype:**
- >75% window detection accuracy in controlled conditions
- Works with your specific webcam and monitor setup
- Subjectively feels responsive

If the prototype fails to meet these criteria, re-evaluate before investing in the full app.

---

## 7. Privacy & Permissions

### Required macOS Permissions

| Permission | Why | When Requested |
|-----------|-----|----------------|
| **Camera** (`NSCameraUsageDescription`) | Core functionality -- face/eye tracking | First launch |
| **Accessibility** (`AXIsProcessTrusted`) | Read UI element details in windows | When user enables "element-level detection" |
| **Screen Recording** (`CGPreflightScreenCaptureAccess`) | Read window titles/names via CGWindowList (macOS 10.15+) | First launch |

### Privacy Architecture

1. **All processing is local.** No camera frames, gaze data, or analytics ever leave the device.
2. **No frame storage.** Camera frames are processed in memory and immediately discarded. Never written to disk.
3. **Minimal data retention.** Only aggregated analytics are stored (e.g., "45 minutes looking at VS Code today"), never raw gaze coordinates or frame data.
4. **Open source CV pipeline.** The face detection and gaze estimation code should be open source so users can audit exactly what happens with their camera feed.
5. **Visible status.** Menu bar icon clearly shows when tracking is active. macOS green camera dot is always visible (cannot be hidden).
6. **User control.** One-click pause/resume. Keyboard shortcut to instantly disable.

---

## 8. Performance Budget

### Target Metrics (Apple Silicon Mac)

| Metric | Target | Stretch |
|--------|--------|---------|
| Face detection + landmarks | <10ms | <5ms |
| Hand detection + landmarks | <10ms | <5ms |
| Gaze estimation | <5ms | <3ms |
| Gesture detection (face + hand) | <3ms | <1ms |
| End-to-end latency (frame to window ID) | <33ms (30fps) | <16ms (60fps) |
| CPU usage (tracking active) | <5% | <3% |
| RAM usage | <100MB | <60MB |
| App bundle size | <50MB | <30MB |
| Battery impact | <20% reduction | <10% reduction |

### Optimization Strategies

- **Resolution:** Capture at 640x480 (sufficient for face detection, saves bandwidth)
- **Adaptive frame rate:** 30fps when user is active, 5fps when idle (no mouse/keyboard for 30s)
- **Frame skipping:** If CV pipeline takes longer than one frame period, drop intermediate frames
- **Neural Engine:** Use CoreML execution provider in ONNX Runtime on Apple Silicon
- **Face detection cadence:** Run full face detection every 3-5 frames; track landmarks on intermediate frames (cheaper)
- **Window polling:** Poll window positions every 500ms (windows don't move that often)

---

## 9. UX Design Principles

### 1. Gaze Detection Should Enhance, Not Replace

VibEyes adds a **passive awareness layer** -- it should enhance the existing mouse/keyboard workflow, not try to replace it. The user should never feel forced to use gaze control.

### 2. Design for Region, Not Pixel

Since webcam gaze is accurate to ~100-200px, design all interactions around large regions (windows, screen halves, large UI elements) rather than precise pointing. Never promise or imply pixel-level accuracy.

### 3. Gradual Confidence

When detecting which window the user is looking at, build confidence over time:
- 0-300ms: uncertain (no action)
- 300-800ms: probable (show subtle indicator)
- 800ms+: confident (take action if auto-raise enabled)

This prevents flickering between windows during saccades (rapid eye movements).

### 4. Gestures Should Be Intentional, Not Accidental

- Safe actions (scroll, highlight) can use simple gestures (blink, nod, open palm)
- Destructive actions (close, delete) require compound gestures (point + nod, or multi-modal confirmation)
- Always provide undo for gesture-triggered actions
- Cooldown periods prevent rapid-fire accidental triggers
- Hand gestures require intentional raise into camera view, which naturally reduces false positives

### 5. Transparent About Uncertainty

When the system isn't sure which window the user is looking at (gaze near window boundaries), show this uncertainty visually rather than guessing wrong. A "not sure" state is better than a wrong state.

### 6. Calibration Should Be Invisible

After the initial 9-point calibration, the system should maintain accuracy through implicit recalibration (tracking mouse clicks). If explicit recalibration is needed, make it fast (<15 seconds) and explain why.

---

## 10. Go-to-Market Considerations

### Positioning

**VibEyes is NOT:**
- A replacement for Tobii or other hardware eye trackers (different accuracy class)
- A medical/research-grade eye tracking tool
- A mouse replacement

**VibEyes IS:**
- A lightweight attention-awareness layer for your desktop
- A "no extra hardware" hands-free interaction system
- A privacy-first productivity tool

### Pricing Model Considerations

| Model | Pros | Cons |
|-------|------|------|
| **Free + open source** | Maximum adoption, community contributions, privacy trust | No revenue, sustainability challenge |
| **Freemium** (core free, analytics/gestures paid) | Good adoption, revenue path | Must decide what's core vs premium |
| **One-time purchase** ($15-30) | Simple, user-friendly | Limited ongoing revenue |
| **Subscription** ($3-5/month) | Sustainable revenue | Users dislike subscriptions for utilities |

**Recommendation:** Start **free and open source** to build trust (camera app = trust-sensitive). Monetize later through a premium analytics dashboard or enterprise API.

### Distribution

- **macOS App Store:** Sandboxed, trusted distribution. May have issues with Accessibility API usage.
- **Direct download (DMG):** More flexibility with permissions. Requires notarization.
- **Homebrew cask:** Developer audience.
- **Recommendation:** Direct download + Homebrew for MVP. App Store later if sandboxing permits.

---

## 11. Open Questions

### Product Questions

1. **Should auto-raise be the default?** Auto-raising the gazed window is the "wow" feature but could be annoying if accuracy isn't high enough. Should it default to off (analytics/indicator only)?

2. **What's the minimum useful accuracy?** If we can only reliably detect screen halves (not individual windows), is that still a useful product?

3. **Multi-monitor: priority or parity?** Is multi-monitor support essential for MVP (many power users have multiple monitors) or can it wait?

4. **Gesture vocabulary:** How many gestures are too many? Should we start with just 3-4 facial + 3-4 hand gestures and expand based on user feedback? Should face and hand gestures be enabled separately or together by default?

5. **Attention analytics privacy:** Even local analytics could feel surveillance-like. How transparent should the analytics be? Should they be opt-in?

### Technical Questions

1. **CoreML vs ONNX Runtime on macOS:** Should we use Apple's Vision framework + CoreML directly for macOS (best performance) and only use ONNX Runtime for cross-platform builds?

2. **Calibration persistence:** How long does a calibration stay valid? Should we save per-user per-monitor calibrations?

3. **External webcam support:** Should we support external webcams (different FOVs, positions) or optimize only for built-in FaceTime camera?

4. **Dual-model approach:** Run a lightweight model at 30fps for tracking + a heavier model at 5fps for accuracy correction?

5. **Screen Recording permission:** macOS screen recording permission enables reading window titles via CGWindowList. Without it, we can still get window bounds but not titles. Is bounds-only sufficient for MVP?

---

## Appendix A: Key Models to Evaluate

| Model | Type | Source | Size | Platform |
|-------|------|--------|------|----------|
| MediaPipe Face Mesh | Face landmarks (478) | Google | ~2MB | All |
| MediaPipe Iris | Iris landmarks (10) | Google | ~1MB | All |
| MediaPipe Hand Landmarker | Hand landmarks (21 per hand) | Google | ~3MB | All |
| L2CS-Net | Gaze estimation | Academic | ~25MB | ONNX exportable |
| ETH-XGaze (ResNet-18) | Gaze estimation | ETH Zurich | ~45MB | ONNX exportable |
| OpenFace | Gaze + AU detection | CMU | ~100MB | C++ (reference) |
| 6DRepNet | Head pose estimation | Academic | ~20MB | ONNX exportable |

## Appendix B: Reference Projects & Papers

**Papers:**
- Krafka et al. (2016) -- "Eye Tracking for Everyone" (GazeCapture dataset, iTracker model)
- Zhang et al. (2020) -- "ETH-XGaze: A Large Scale Dataset for Gaze Estimation" 
- Abdelrahman et al. (2022) -- "L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments"
- Timm & Barth (2011) -- "Accurate Eye Centre Localisation by Means of Gradients"
- Baltrusaitis et al. (2018) -- "OpenFace 2.0: Facial Behavior Analysis Toolkit"

**Code repositories:**
- `google/mediapipe` -- Face mesh + iris detection
- `Ahmednull/L2CS-Net` -- Gaze estimation model
- `xucong-zhang/ETH-XGaze` -- Gaze estimation dataset/model
- `TadasBaltrusaitis/OpenFace` -- Facial behavior analysis
- `brownhci/WebGazer` -- Browser-based eye tracking
- `google/mediapipe` (Hand Landmarker) -- Hand landmark detection (21 points per hand)
- `nickynicolson/gaze-estimation` -- Gaze calibration reference

## Appendix C: macOS API Quick Reference

```swift
// Get all visible windows
let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]]

// Each window dict contains:
// kCGWindowOwnerName  -- "Google Chrome"
// kCGWindowName       -- "VibEyes PRD - Google Docs"  
// kCGWindowBounds     -- {"X": 100, "Y": 200, "Width": 1200, "Height": 800}
// kCGWindowOwnerPID   -- 12345
// kCGWindowLayer      -- 0 (normal), 25 (overlay)

// Get frontmost app
NSWorkspace.shared.frontmostApplication?.localizedName  // "Google Chrome"

// Accessibility: get focused element
let systemWide = AXUIElementCreateSystemWide()
var focusedElement: CFTypeRef?
AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute, &focusedElement)
```
