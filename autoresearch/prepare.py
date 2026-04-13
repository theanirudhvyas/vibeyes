"""Replay harness for vibeyes autoresearch. READ-ONLY — do not modify."""

import json
import math
import os
import sys
import time

import cv2
import numpy as np


RECORDINGS_DIR = os.path.expanduser("~/.cache/vibeyes/recordings")
TEST_SPLIT_RATIO = 0.2
MIN_CLICKS = 30


def load_sessions(recordings_dir: str = RECORDINGS_DIR) -> list[dict]:
    """Load all recorded sessions."""
    sessions = []
    if not os.path.isdir(recordings_dir):
        print(f"No recordings directory found at {recordings_dir}")
        sys.exit(1)

    for name in sorted(os.listdir(recordings_dir)):
        session_dir = os.path.join(recordings_dir, name)
        clicks_path = os.path.join(session_dir, "clicks.jsonl")
        frames_dir = os.path.join(session_dir, "frames")

        if not os.path.isfile(clicks_path) or not os.path.isdir(frames_dir):
            continue

        clicks = []
        with open(clicks_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    clicks.append(json.loads(line))

        if len(clicks) < MIN_CLICKS:
            print(f"  Skipping {name}: only {len(clicks)} clicks (need {MIN_CLICKS})")
            continue

        sessions.append({
            "name": name,
            "frames_dir": frames_dir,
            "clicks": clicks,
        })

    if not sessions:
        print("No usable sessions found. Need at least one session with "
              f"{MIN_CLICKS}+ clicks in {recordings_dir}")
        sys.exit(1)

    total_clicks = sum(len(s["clicks"]) for s in sessions)
    print(f"Loaded {len(sessions)} session(s), {total_clicks} total clicks")
    return sessions


def load_frame(frames_dir: str, frame_id: str) -> np.ndarray:
    """Load a single frame by ID."""
    path = os.path.join(frames_dir, f"{frame_id}.jpg")
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return frame


def split_session(session: dict) -> tuple[list[dict], list[dict]]:
    """Split a session into calibration clicks and test clicks."""
    clicks = session["clicks"]
    split_idx = int(len(clicks) * (1 - TEST_SPLIT_RATIO))
    return clicks[:split_idx], clicks[split_idx:]


def evaluate(predictions: list[tuple[float, float]],
             actuals: list[tuple[float, float]],
             screen_w: float = 3360.0,
             screen_h: float = 2100.0) -> dict:
    """Compute accuracy metrics including per-region breakdown."""
    assert len(predictions) == len(actuals), \
        f"Length mismatch: {len(predictions)} predictions vs {len(actuals)} actuals"

    errors = []
    region_errors = {name: [] for name in [
        "TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]}

    for (px, py), (ax, ay) in zip(predictions, actuals):
        err = math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        errors.append(err)

        col = min(2, int(ax / (screen_w / 3)))
        row = min(2, int(ay / (screen_h / 3)))
        region_name = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"][row * 3 + col]
        region_errors[region_name].append(err)

    errors.sort()
    n = len(errors)

    region_medians = {}
    for name, errs in region_errors.items():
        if errs:
            errs.sort()
            region_medians[name] = errs[len(errs) // 2]
        else:
            region_medians[name] = None

    return {
        "median_error_px": errors[n // 2],   # PRIMARY METRIC
        "avg_error_px": sum(errors) / n,
        "p90_error_px": errors[int(n * 0.9)],
        "min_error_px": errors[0],
        "max_error_px": errors[-1],
        "n_test_clicks": n,
        "region_errors": region_medians,
    }


def run_evaluation():
    """Main entry point: load sessions, run pipeline, print metrics."""
    from pipeline import replay_calibration, predict

    sessions = load_sessions()

    all_predictions = []
    all_actuals = []

    for session in sessions:
        cal_clicks, test_clicks = split_session(session)
        frames_dir = session["frames_dir"]

        print(f"\n  Session {session['name']}: "
              f"{len(cal_clicks)} calibration + {len(test_clicks)} test clicks")

        t0 = time.time()
        state = replay_calibration(frames_dir, cal_clicks)
        cal_time = time.time() - t0

        t0 = time.time()
        for click in test_clicks:
            frame = load_frame(frames_dir, click["frame_id"])
            px, py = predict(frame, state)
            all_predictions.append((px, py))
            all_actuals.append((click["click_x"], click["click_y"]))
        eval_time = time.time() - t0

        print(f"    Calibration: {cal_time:.1f}s, Evaluation: {eval_time:.1f}s")

    metrics = evaluate(all_predictions, all_actuals)

    print("\n---")
    print(f"median_error_px: {metrics['median_error_px']:.1f}")
    print(f"avg_error_px:    {metrics['avg_error_px']:.1f}")
    print(f"p90_error_px:    {metrics['p90_error_px']:.1f}")
    print(f"min_error_px:    {metrics['min_error_px']:.1f}")
    print(f"max_error_px:    {metrics['max_error_px']:.1f}")
    print(f"n_test_clicks:   {metrics['n_test_clicks']}")

    r = metrics["region_errors"]
    def _fmt(v):
        return f"{v:.1f}" if v is not None else "n/a"
    print(f"\nPer-region median error (3x3 grid):")
    print(f"  TL: {_fmt(r['TL']):>7s}  TC: {_fmt(r['TC']):>7s}  TR: {_fmt(r['TR']):>7s}")
    print(f"  ML: {_fmt(r['ML']):>7s}  MC: {_fmt(r['MC']):>7s}  MR: {_fmt(r['MR']):>7s}")
    print(f"  BL: {_fmt(r['BL']):>7s}  BC: {_fmt(r['BC']):>7s}  BR: {_fmt(r['BR']):>7s}")
    print("---")


if __name__ == "__main__":
    run_evaluation()
