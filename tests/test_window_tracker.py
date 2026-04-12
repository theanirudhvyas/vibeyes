"""Tests for window_tracker module."""

import pytest

from vibeyes import WindowInfo
from vibeyes.window_tracker import get_visible_windows, hit_test


class TestGetVisibleWindows:
    """Tests for macOS window enumeration."""

    def test_returns_list(self):
        """get_visible_windows() should return a list."""
        windows = get_visible_windows()
        assert isinstance(windows, list)

    def test_windows_are_window_info(self):
        """Each element should be a WindowInfo instance."""
        windows = get_visible_windows()
        if not windows:
            pytest.skip("No visible windows found")
        for w in windows:
            assert isinstance(w, WindowInfo)

    def test_windows_have_bounds(self):
        """Each window should have valid bounds (x, y, width, height)."""
        windows = get_visible_windows()
        if not windows:
            pytest.skip("No visible windows found")
        for w in windows:
            x, y, width, height = w.bounds
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert width >= 0
            assert height >= 0

    def test_windows_have_app_name(self):
        """Each window should have a non-empty app_name."""
        windows = get_visible_windows()
        if not windows:
            pytest.skip("No visible windows found")
        for w in windows:
            assert isinstance(w.app_name, str)
            assert len(w.app_name) > 0

    def test_excludes_vibeyes_windows(self):
        """Windows belonging to VibEyes itself should be filtered out."""
        windows = get_visible_windows(exclude_app="VibEyes")
        for w in windows:
            assert w.app_name != "VibEyes"


class TestHitTest:
    """Tests for gaze-to-window hit testing."""

    def test_hit_inside_window(self, sample_windows):
        """A point inside a window should return that window."""
        result = hit_test(480, 300, sample_windows)
        assert result is not None
        assert result.app_name == "Code"

    def test_hit_second_window(self, sample_windows):
        """A point inside the second window should return it."""
        result = hit_test(1200, 300, sample_windows)
        assert result is not None
        assert result.app_name == "Slack"

    def test_hit_bottom_window(self, sample_windows):
        """A point in the bottom window should return Terminal."""
        result = hit_test(500, 800, sample_windows)
        assert result is not None
        assert result.app_name == "Terminal"

    def test_miss_returns_none(self, sample_windows):
        """A point outside all windows should return None."""
        result = hit_test(5000, 5000, sample_windows)
        assert result is None

    def test_edge_point_on_boundary(self, sample_windows):
        """A point exactly on the window boundary should count as inside."""
        result = hit_test(0, 0, sample_windows)
        assert result is not None

    def test_front_window_wins_on_overlap(self):
        """When windows overlap, the first (frontmost) window should be returned."""
        overlapping = [
            WindowInfo("Front", "App1", (100, 100, 400, 400), 1, 0),
            WindowInfo("Back", "App2", (200, 200, 400, 400), 2, 0),
        ]
        result = hit_test(300, 300, overlapping)
        assert result.title == "Front"

    def test_empty_window_list(self):
        """hit_test with empty list should return None."""
        result = hit_test(500, 500, [])
        assert result is None
