"""Tests for pane_tracker module (zellij pane detection)."""

import pytest

from vibeyes.pane_tracker import PaneInfo, parse_active_tab_panes, compute_pane_bounds, hit_test_pane


SAMPLE_LAYOUT_SIMPLE = """
layout {
    tab name="other" hide_floating_panes=true {
        pane size=1 borderless=true {
            plugin location="zellij:tab-bar"
        }
        pane cwd="/some/path"
        pane size=1 borderless=true {
            plugin location="zellij:status-bar"
        }
    }
    tab name="vibeyes" focus=true hide_floating_panes=true {
        pane size=1 borderless=true {
            plugin location="zellij:tab-bar"
        }
        pane split_direction="vertical" {
            pane command="claude" cwd="code/vibeyes" focus=true size="50%" {
                args "--resume" "abc"
            }
            pane command="python" cwd="code/vibeyes" size="50%" {
                args "-m" "vibeyes.main"
            }
        }
        pane size=1 borderless=true {
            plugin location="zellij:status-bar"
        }
    }
}
"""

SAMPLE_LAYOUT_HSPLIT = """
layout {
    tab name="work" focus=true hide_floating_panes=true {
        pane size=1 borderless=true {
            plugin location="zellij:tab-bar"
        }
        pane split_direction="horizontal" {
            pane cwd="/top" size="60%"
            pane cwd="/bottom" size="40%"
        }
        pane size=1 borderless=true {
            plugin location="zellij:status-bar"
        }
    }
}
"""

SAMPLE_LAYOUT_SINGLE = """
layout {
    tab name="solo" focus=true hide_floating_panes=true {
        pane size=1 borderless=true {
            plugin location="zellij:tab-bar"
        }
        pane command="vim" cwd="/code"
        pane size=1 borderless=true {
            plugin location="zellij:status-bar"
        }
    }
}
"""


class TestParseActiveTabPanes:
    """Tests for parsing zellij layout dump."""

    def test_finds_focused_tab(self):
        """Should parse the tab with focus=true."""
        panes = parse_active_tab_panes(SAMPLE_LAYOUT_SIMPLE)
        assert panes is not None
        assert len(panes) == 2

    def test_extracts_pane_info(self):
        """Each pane should have command, cwd, and is_focused."""
        panes = parse_active_tab_panes(SAMPLE_LAYOUT_SIMPLE)
        assert panes[0].command == "claude"
        assert panes[0].cwd == "code/vibeyes"
        assert panes[0].is_focused is True
        assert panes[1].command == "python"
        assert panes[1].is_focused is False

    def test_extracts_split_direction_and_sizes(self):
        """Panes should have their size percentages and split direction."""
        panes = parse_active_tab_panes(SAMPLE_LAYOUT_SIMPLE)
        assert panes[0].size_percent == 50.0
        assert panes[1].size_percent == 50.0
        assert panes[0].split_direction == "vertical"

    def test_horizontal_split(self):
        """Should handle horizontal splits."""
        panes = parse_active_tab_panes(SAMPLE_LAYOUT_HSPLIT)
        assert len(panes) == 2
        assert panes[0].split_direction == "horizontal"
        assert panes[0].size_percent == 60.0
        assert panes[1].size_percent == 40.0

    def test_single_pane(self):
        """Should handle a single pane (no split)."""
        panes = parse_active_tab_panes(SAMPLE_LAYOUT_SINGLE)
        assert len(panes) == 1
        assert panes[0].command == "vim"

    def test_no_focused_tab_returns_none(self):
        """If no tab has focus=true, return None."""
        layout = 'layout { tab name="x" { pane } }'
        panes = parse_active_tab_panes(layout)
        assert panes is None


class TestComputePaneBounds:
    """Tests for computing pixel bounds from pane info."""

    def test_vertical_split_two_panes(self):
        """Two 50/50 vertical panes should split the width."""
        panes = [
            PaneInfo(command="a", cwd="/a", is_focused=True, size_percent=50, split_direction="vertical", index=0),
            PaneInfo(command="b", cwd="/b", is_focused=False, size_percent=50, split_direction="vertical", index=1),
        ]
        # Window at (100, 200) size 1000x800
        bounds = compute_pane_bounds(panes, window_x=100, window_y=200, window_w=1000, window_h=800)
        assert len(bounds) == 2
        # First pane: left half
        assert bounds[0][0] == pytest.approx(100, abs=5)
        assert bounds[0][2] == pytest.approx(500, abs=5)  # width
        # Second pane: right half
        assert bounds[1][0] == pytest.approx(600, abs=5)
        assert bounds[1][2] == pytest.approx(500, abs=5)

    def test_horizontal_split_two_panes(self):
        """Two 60/40 horizontal panes should split the height."""
        panes = [
            PaneInfo(command="a", cwd="/a", is_focused=True, size_percent=60, split_direction="horizontal", index=0),
            PaneInfo(command="b", cwd="/b", is_focused=False, size_percent=40, split_direction="horizontal", index=1),
        ]
        bounds = compute_pane_bounds(panes, window_x=0, window_y=0, window_w=1000, window_h=800)
        assert len(bounds) == 2
        # First pane: top 60%
        assert bounds[0][3] == pytest.approx(480, abs=5)  # height
        # Second pane: bottom 40%
        assert bounds[1][3] == pytest.approx(320, abs=5)

    def test_single_pane_fills_window(self):
        """A single pane should fill the entire window."""
        panes = [
            PaneInfo(command="vim", cwd="/", is_focused=True, size_percent=100, split_direction="vertical", index=0),
        ]
        bounds = compute_pane_bounds(panes, window_x=0, window_y=0, window_w=1000, window_h=800)
        assert bounds[0] == (0, 0, 1000, 800)


class TestHitTestPane:
    """Tests for pane hit testing."""

    def test_hit_left_pane(self):
        """Point in left half should hit the left pane."""
        panes = [
            PaneInfo(command="claude", cwd="/code", is_focused=True, size_percent=50, split_direction="vertical", index=0),
            PaneInfo(command="python", cwd="/code", is_focused=False, size_percent=50, split_direction="vertical", index=1),
        ]
        result = hit_test_pane(300, 400, panes, window_x=0, window_y=0, window_w=1000, window_h=800)
        assert result is not None
        assert result.command == "claude"

    def test_hit_right_pane(self):
        """Point in right half should hit the right pane."""
        panes = [
            PaneInfo(command="claude", cwd="/code", is_focused=True, size_percent=50, split_direction="vertical", index=0),
            PaneInfo(command="python", cwd="/code", is_focused=False, size_percent=50, split_direction="vertical", index=1),
        ]
        result = hit_test_pane(700, 400, panes, window_x=0, window_y=0, window_w=1000, window_h=800)
        assert result is not None
        assert result.command == "python"

    def test_miss_returns_none(self):
        """Point outside window should return None."""
        panes = [
            PaneInfo(command="vim", cwd="/", is_focused=True, size_percent=100, split_direction="vertical", index=0),
        ]
        result = hit_test_pane(5000, 5000, panes, window_x=0, window_y=0, window_w=1000, window_h=800)
        assert result is None
