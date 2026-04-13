.PHONY: help setup calibrate run run-overlay run-debug record list-cameras test clean-calibration clean-recordings
.DEFAULT_GOAL := help

VENV = .venv/bin
PYTHON = $(VENV)/python

# ─── Help ────────────────────────────────────────────────────

help:
	@echo "VibEyes - Webcam gaze tracking for window focus detection"
	@echo ""
	@echo "Setup:"
	@echo "  make setup               Install venv, deps, and download model"
	@echo ""
	@echo "Usage:"
	@echo "  make calibrate           Run 16-point calibration with overlay"
	@echo "  make run                 Track gaze, print detected window"
	@echo "  make run-overlay         Track with translucent gaze dot on screen"
	@echo "  make run-debug           Track with overlay + debug output"
	@echo "  make list-cameras        List available cameras"
	@echo ""
	@echo "  Specify camera:  make calibrate CAMERA=1"
	@echo ""
	@echo "Testing & Metrics:"
	@echo "  make test                Run all 52 tests"
	@echo "  make test-quick          Run tests (compact output)"
	@echo "  make metrics             Show click accuracy stats"
	@echo ""
	@echo "Autoresearch:"
	@echo "  make autoresearch-recordings  List recorded sessions + click counts"
	@echo "  make autoresearch-baseline    Run baseline evaluation on recordings"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-calibration   Delete calibration (forces recalibrate)"
	@echo "  make clean-recordings    Delete all recorded sessions"
	@echo "  make clean-metrics       Delete accuracy metrics database"

# ─── Setup ───────────────────────────────────────────────────

setup: .venv models/face_landmarker.task
	@echo "Ready. Run: make calibrate"

.venv:
	/opt/homebrew/bin/python3.12 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -e ".[dev]"

models/face_landmarker.task:
	mkdir -p models
	curl -L -o models/face_landmarker.task \
		"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# ─── Calibrate & Run ────────────────────────────────────────

calibrate:
	$(PYTHON) -m vibeyes.main --calibrate --overlay

run:
	$(PYTHON) -m vibeyes.main

run-overlay:
	$(PYTHON) -m vibeyes.main --overlay

run-debug:
	$(PYTHON) -m vibeyes.main --overlay --debug

record:
	$(PYTHON) -m vibeyes.main --overlay --debug

list-cameras:
	$(PYTHON) -m vibeyes.main --list-cameras

# ─── Camera selection ───────────────────────────────────────
# Usage: make calibrate CAMERA=1
ifdef CAMERA
calibrate: CAMERA_FLAG=--camera $(CAMERA)
run: CAMERA_FLAG=--camera $(CAMERA)
run-overlay: CAMERA_FLAG=--camera $(CAMERA)
run-debug: CAMERA_FLAG=--camera $(CAMERA)
record: CAMERA_FLAG=--camera $(CAMERA)

calibrate run run-overlay run-debug record:
	$(PYTHON) -m vibeyes.main $(CAMERA_FLAG) $(filter-out $@,$(MAKECMDGOALS))
endif

# ─── Tests ──────────────────────────────────────────────────

test:
	$(VENV)/pytest -v

test-quick:
	$(VENV)/pytest -q

# ─── Autoresearch ───────────────────────────────────────────

autoresearch-baseline:
	cd autoresearch && $(CURDIR)/$(PYTHON) prepare.py

autoresearch-recordings:
	@echo "Recordings in ~/.cache/vibeyes/recordings/:"
	@ls -la ~/.cache/vibeyes/recordings/ 2>/dev/null || echo "  No recordings yet. Run: make record"
	@for d in ~/.cache/vibeyes/recordings/session_*; do \
		if [ -f "$$d/clicks.jsonl" ]; then \
			count=$$(wc -l < "$$d/clicks.jsonl"); \
			echo "  $$(basename $$d): $$count clicks"; \
		fi \
	done 2>/dev/null

# ─── Metrics ────────────────────────────────────────────────

metrics:
	@$(PYTHON) -c "\
	import sqlite3; \
	db = sqlite3.connect('vibeyes_metrics.db'); \
	row = db.execute('SELECT COUNT(*), AVG(error_px), MIN(error_px), MAX(error_px) FROM gaze_errors').fetchone(); \
	recent = db.execute('SELECT AVG(error_px) FROM (SELECT error_px FROM gaze_errors ORDER BY id DESC LIMIT 20)').fetchone(); \
	print(f'Total clicks: {row[0]}'); \
	print(f'Avg error: {row[1]:.0f}px') if row[1] else None; \
	print(f'Recent 20 avg: {recent[0]:.0f}px') if recent[0] else None; \
	db.close()" 2>/dev/null || echo "No metrics yet."

# ─── Cleanup ────────────────────────────────────────────────

clean-calibration:
	rm -f calibration.json
	@echo "Calibration removed. Run: make calibrate"

clean-recordings:
	rm -rf ~/.cache/vibeyes/recordings/*
	@echo "All recordings deleted."

clean-metrics:
	rm -f vibeyes_metrics.db
	@echo "Metrics database deleted."
