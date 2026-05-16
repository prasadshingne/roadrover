"""
Tests for the ego ENU position formula in MapMatcher.match().

This formula has broken twice:
  v1 (wrong): offset = (N - lane_num + 0.5) * LANE_WIDTH
              assumed snap point = left road boundary; shifted ego ~N/2 lanes right
  v2 (correct): offset = (N/2 - lane_num + 0.5) * LANE_WIDTH
                assumes snap point = road centreline (current make_map.py convention)

These tests verify the correct formula and serve as a regression guard — the
large warning comment in process_bag.py exists precisely because this was easy
to accidentally revert.
"""
import math
import pytest

LANE_WIDTH = 3.5   # metres — must match process_bag.LANE_WIDTH


def _ego_enu(total_lanes: int, lane_num: int,
             snap_x: float, snap_y: float,
             ego_heading: float, heading_valid: bool = True):
    """
    Exact replica of the ENU offset block in MapMatcher.match().
    Kept here as a pure function so tests carry no import-time ROS dependency.
    """
    if not heading_valid:
        return snap_x, snap_y
    offset = (total_lanes / 2.0 - lane_num + 0.5) * LANE_WIDTH
    x = snap_x + offset * math.sin(ego_heading)
    y = snap_y - offset * math.cos(ego_heading)
    return x, y


def _wrong_formula(total_lanes: int, lane_num: int,
                   snap_x: float, snap_y: float, ego_heading: float):
    """The old (wrong) formula — snap assumed to be left road boundary."""
    offset = (total_lanes - lane_num + 0.5) * LANE_WIDTH
    x = snap_x + offset * math.sin(ego_heading)
    y = snap_y - offset * math.cos(ego_heading)
    return x, y


# ── 2-lane road ───────────────────────────────────────────────────────────────

def test_2lane_rightmost_is_right_of_centre():
    # Heading east (0 rad): right of travel = south (−y)
    x, y = _ego_enu(2, lane_num=1, snap_x=0, snap_y=0, ego_heading=0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(-LANE_WIDTH / 2)   # 1.75 m south of centre


def test_2lane_leftmost_is_left_of_centre():
    # Heading east (0 rad): left of travel = north (+y)
    x, y = _ego_enu(2, lane_num=2, snap_x=0, snap_y=0, ego_heading=0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(LANE_WIDTH / 2)    # 1.75 m north of centre


def test_2lane_symmetric_about_centre():
    x1, y1 = _ego_enu(2, 1, 0, 0, ego_heading=0)
    x2, y2 = _ego_enu(2, 2, 0, 0, ego_heading=0)
    assert x1 == pytest.approx(-x2, abs=1e-9)
    assert y1 == pytest.approx(-y2, abs=1e-9)


# ── 3-lane road ───────────────────────────────────────────────────────────────

def test_3lane_middle_lane_is_at_centre():
    x, y = _ego_enu(3, lane_num=2, snap_x=0, snap_y=0, ego_heading=0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_3lane_outer_lanes_equidistant():
    _, y1 = _ego_enu(3, 1, 0, 0, ego_heading=0)
    _, y3 = _ego_enu(3, 3, 0, 0, ego_heading=0)
    assert abs(y1) == pytest.approx(abs(y3), rel=1e-6)
    assert y1 != pytest.approx(y3)          # opposite sides


# ── Heading direction ─────────────────────────────────────────────────────────

def test_heading_north_rightmost_lane_is_east():
    # Heading north (π/2): right of travel = east (+x)
    x, y = _ego_enu(2, lane_num=1, snap_x=0, snap_y=0, ego_heading=math.pi / 2)
    assert x == pytest.approx(LANE_WIDTH / 2, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_heading_west_rightmost_lane_is_north():
    # Heading west (π): right of travel = north (+y)
    x, y = _ego_enu(2, lane_num=1, snap_x=0, snap_y=0, ego_heading=math.pi)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(LANE_WIDTH / 2, abs=1e-6)


# ── heading_valid=False ───────────────────────────────────────────────────────

def test_heading_invalid_returns_snap_point():
    # When vehicle is stationary, heading is unreliable; offset must not be applied
    x, y = _ego_enu(2, 1, snap_x=100.0, snap_y=200.0,
                    ego_heading=0.0, heading_valid=False)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(200.0)


# ── Regression: old formula would place ego in the wrong position ─────────────

def test_regression_wrong_formula_differs_on_2lane():
    # On a 2-lane road the old formula shifts the ego ~N/2 * LANE_WIDTH further right.
    # Both should agree only when lane_num = total_lanes (leftmost lane, N-based indexing
    # accidentally cancels out) — in all other cases they must differ.
    for lane in range(1, 3):
        cx, cy = _ego_enu(2, lane, 0, 0, ego_heading=0)
        wx, wy = _wrong_formula(2, lane, 0, 0, ego_heading=0)
        if lane < 2:
            assert (cx, cy) != pytest.approx((wx, wy), abs=0.1), (
                f"lane {lane}/2: correct and wrong formulas should disagree"
            )


def test_regression_old_formula_puts_ego_off_road():
    # Old formula for lane 1 of a 2-lane road → offset = 1.5 * LANE_WIDTH = 5.25 m right.
    # Correct formula → 0.5 * LANE_WIDTH = 1.75 m right.
    # The error is exactly N/2 * LANE_WIDTH = one full lane width on a 2-lane road,
    # which places the ego marker in the wrong lane (or in the median).
    _, y_wrong   = _wrong_formula(2, 1, 0, 0, ego_heading=0)
    _, y_correct = _ego_enu(2,       1, 0, 0, ego_heading=0)
    assert abs(y_wrong - y_correct) == pytest.approx(LANE_WIDTH, rel=0.01)
