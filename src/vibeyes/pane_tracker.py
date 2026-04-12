"""Zellij pane detection -- maps gaze position to a specific zellij pane."""

import re
import subprocess
from dataclasses import dataclass


@dataclass
class PaneInfo:
    """Information about a zellij pane."""
    command: str
    cwd: str
    is_focused: bool
    size_percent: float
    split_direction: str  # "vertical" or "horizontal"
    index: int  # position within the split (0-based)


def get_zellij_layout() -> str | None:
    """Get the current zellij layout dump. Returns None if zellij is not running."""
    try:
        result = subprocess.run(
            ["zellij", "action", "dump-layout"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_active_tab_panes(layout_text: str) -> list[PaneInfo] | None:
    """Parse the focused tab's pane structure from a zellij layout dump.

    Returns a flat list of content panes (excludes tab-bar/status-bar plugins).
    Returns None if no focused tab is found.
    """
    # Find the focused tab block
    # Match: tab name="..." focus=true ... { ... }
    # We need to handle nested braces, so we count them
    focus_tab_start = None
    for match in re.finditer(r'tab\s+name="([^"]*)"[^{]*focus=true[^{]*\{', layout_text):
        focus_tab_start = match.start()
        tab_name = match.group(1)
        brace_pos = match.end() - 1
        break

    if focus_tab_start is None:
        return None

    # Extract the tab's content by counting braces
    tab_content = _extract_brace_block(layout_text, brace_pos)
    if tab_content is None:
        return None

    # Find the main content split (not tab-bar or status-bar plugins)
    # Look for pane with split_direction, or a non-plugin pane
    panes = []

    # Check for split pane
    split_match = re.search(r'pane\s+split_direction="(vertical|horizontal)"\s*\{', tab_content)
    if split_match:
        split_dir = split_match.group(1)
        split_content = _extract_brace_block(tab_content, split_match.end() - 1)
        if split_content:
            # Parse child panes within the split
            panes = _parse_child_panes(split_content, split_dir)
    else:
        # Single pane (no split) - find the non-plugin, non-bar pane
        for pane_match in re.finditer(r'pane\s+([^\n{]*)', tab_content):
            attrs = pane_match.group(1).strip()

            # Skip bar panes (size=1 borderless=true with plugin)
            if 'size=1' in attrs and 'borderless=true' in attrs:
                continue

            command_m = re.search(r'command="([^"]*)"', attrs)
            cwd_m = re.search(r'cwd="([^"]*)"', attrs)
            focus_m = re.search(r'focus=true', attrs)

            panes.append(PaneInfo(
                command=command_m.group(1) if command_m else "shell",
                cwd=cwd_m.group(1) if cwd_m else "",
                is_focused=focus_m is not None,
                size_percent=100.0,
                split_direction="vertical",
                index=0,
            ))
            break

    return panes if panes else None


def _parse_child_panes(split_content: str, split_direction: str) -> list[PaneInfo]:
    """Parse child panes within a split block."""
    panes = []
    index = 0

    # Match pane declarations with their attributes
    for match in re.finditer(
        r'pane\s+((?:command="[^"]*"\s*)?(?:cwd="[^"]*"\s*)?(?:focus=true\s*)?(?:size="[^"]*"\s*)?)',
        split_content
    ):
        attrs = match.group(1)

        # Skip plugin panes
        rest = split_content[match.start():]
        if 'plugin location=' in rest[:200] and 'zellij:' in rest[:200]:
            continue

        command_m = re.search(r'command="([^"]*)"', attrs)
        cwd_m = re.search(r'cwd="([^"]*)"', attrs)
        focus_m = re.search(r'focus=true', attrs)
        size_m = re.search(r'size="([^"]*)"', attrs)

        command = command_m.group(1) if command_m else "shell"
        cwd = cwd_m.group(1) if cwd_m else ""
        is_focused = focus_m is not None
        size_str = size_m.group(1) if size_m else ""

        # Parse size percentage
        if size_str.endswith("%"):
            size_percent = float(size_str[:-1])
        else:
            size_percent = 0  # Will be computed

        panes.append(PaneInfo(
            command=command,
            cwd=cwd,
            is_focused=is_focused,
            size_percent=size_percent,
            split_direction=split_direction,
            index=index,
        ))
        index += 1

    # If sizes don't add up, distribute evenly
    total = sum(p.size_percent for p in panes)
    if total == 0 and panes:
        even = 100.0 / len(panes)
        for p in panes:
            p.size_percent = even

    return panes


def _extract_brace_block(text: str, open_brace_pos: int) -> str | None:
    """Extract content between matching braces starting at open_brace_pos."""
    if text[open_brace_pos] != '{':
        return None
    depth = 1
    i = open_brace_pos + 1
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[open_brace_pos + 1:i - 1]
    return None


def compute_pane_bounds(
    panes: list[PaneInfo],
    window_x: float, window_y: float,
    window_w: float, window_h: float,
) -> list[tuple[float, float, float, float]]:
    """Compute pixel bounds (x, y, w, h) for each pane given the window geometry."""
    if not panes:
        return []

    if len(panes) == 1:
        return [(window_x, window_y, window_w, window_h)]

    bounds = []
    split_dir = panes[0].split_direction
    total_percent = sum(p.size_percent for p in panes)

    if split_dir == "vertical":
        # Split along x-axis (side by side)
        cursor_x = window_x
        for pane in panes:
            frac = pane.size_percent / total_percent if total_percent > 0 else 1.0 / len(panes)
            pw = window_w * frac
            bounds.append((cursor_x, window_y, pw, window_h))
            cursor_x += pw
    else:
        # Split along y-axis (stacked)
        cursor_y = window_y
        for pane in panes:
            frac = pane.size_percent / total_percent if total_percent > 0 else 1.0 / len(panes)
            ph = window_h * frac
            bounds.append((window_x, cursor_y, window_w, ph))
            cursor_y += ph

    return bounds


def hit_test_pane(
    gaze_x: float, gaze_y: float,
    panes: list[PaneInfo],
    window_x: float, window_y: float,
    window_w: float, window_h: float,
) -> PaneInfo | None:
    """Return the pane at the given gaze position, or None."""
    bounds = compute_pane_bounds(panes, window_x, window_y, window_w, window_h)
    for pane, (bx, by, bw, bh) in zip(panes, bounds):
        if bx <= gaze_x <= bx + bw and by <= gaze_y <= by + bh:
            return pane
    return None
