"""Accuracy metrics tracking with SQLite timeseries storage."""

import math
import os
import sqlite3
import time


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "vibeyes_metrics.db")


class MetricsTracker:
    """Tracks gaze prediction error over time using SQLite."""

    def __init__(self, db_path: str = DB_PATH):
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS gaze_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                predicted_x REAL NOT NULL,
                predicted_y REAL NOT NULL,
                actual_x REAL NOT NULL,
                actual_y REAL NOT NULL,
                error_px REAL NOT NULL,
                calibration_points INTEGER NOT NULL
            )
        """)
        self._db.commit()

    def record_click(self, predicted_x: float, predicted_y: float,
                     actual_x: float, actual_y: float, calibration_points: int):
        """Record a click event with predicted vs actual gaze position."""
        error_px = math.sqrt((predicted_x - actual_x) ** 2 + (predicted_y - actual_y) ** 2)
        self._db.execute(
            "INSERT INTO gaze_errors (timestamp, predicted_x, predicted_y, actual_x, actual_y, error_px, calibration_points) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), predicted_x, predicted_y, actual_x, actual_y, error_px, calibration_points),
        )
        self._db.commit()
        return error_px

    def get_recent_avg_error(self, n: int = 20) -> float | None:
        """Get average error of the last N clicks."""
        row = self._db.execute(
            "SELECT AVG(error_px) FROM (SELECT error_px FROM gaze_errors ORDER BY id DESC LIMIT ?)",
            (n,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def get_stats(self) -> dict:
        """Get overall accuracy statistics."""
        row = self._db.execute("""
            SELECT COUNT(*), AVG(error_px), MIN(error_px), MAX(error_px)
            FROM gaze_errors
        """).fetchone()

        recent = self._db.execute("""
            SELECT AVG(error_px) FROM (
                SELECT error_px FROM gaze_errors ORDER BY id DESC LIMIT 20
            )
        """).fetchone()

        return {
            "total_clicks": row[0],
            "avg_error_px": round(row[1], 1) if row[1] else 0,
            "min_error_px": round(row[2], 1) if row[2] else 0,
            "max_error_px": round(row[3], 1) if row[3] else 0,
            "recent_avg_error_px": round(recent[0], 1) if recent[0] else 0,
        }

    def close(self):
        self._db.close()
