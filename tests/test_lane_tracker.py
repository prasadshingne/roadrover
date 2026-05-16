"""
Tests for LaneTracker — the BEV polynomial lane tracker in process_bag.py.

Covers:
  _fit()       — every rejection gate that was added to fix the staircase bug
  bev_lateral()— self-calibrating lateral distance measurement
  update()     — EMA smoothing and stale-frame counter
"""
import numpy as np
import pytest

from process_bag import BEV_H, BEV_MARGIN, BEV_W, LANE_WIDTH, LaneTracker


# ── helpers ───────────────────────────────────────────────────────────────────

def _vertical_pts(x: float, n: int = 30):
    """Points along a vertical line at column x, spread over the full BEV height."""
    ys = np.linspace(0, BEV_H - 1, n)
    return [(x, y) for y in ys]


def _curved_pts(a: float, x0: float, n: int = 30):
    """Points along x = a*(y - BEV_H/2)^2 + x0 (parabola centred at mid-height)."""
    ys = np.linspace(0, BEV_H - 1, n)
    return [(a * (y - BEV_H / 2) ** 2 + x0, y) for y in ys]


def _tracker_with_polys(left_x: float, right_x: float) -> LaneTracker:
    """Return a LaneTracker whose polynomials represent two vertical lane lines."""
    t = LaneTracker()
    t.left_poly  = np.array([0.0, 0.0, left_x])
    t.right_poly = np.array([0.0, 0.0, right_x])
    return t


# ── _fit() rejection gates ────────────────────────────────────────────────────

def test_fit_rejects_too_few_points():
    t = LaneTracker()
    pts = _vertical_pts(100.0, n=5)   # only 5 pts, minimum is 6
    assert t._fit(pts, 'left') is None


def test_fit_rejects_insufficient_y_spread():
    t = LaneTracker()
    # All points confined to a 50-pixel band (< BEV_H * 0.20 = 60 px)
    pts = [(100.0, y) for y in np.linspace(100, 149, 20)]
    assert t._fit(pts, 'left') is None


def test_fit_rejects_high_curvature():
    t = LaneTracker()
    # Parabola with a=0.02 > 0.01 limit
    pts = _curved_pts(a=0.02, x0=100.0)
    assert t._fit(pts, 'left') is None


def test_fit_rejects_left_line_in_right_half():
    t = LaneTracker()
    # A vertical line at x=300 (right half) must be rejected when side='left'
    pts = _vertical_pts(300.0)
    assert t._fit(pts, 'left') is None


def test_fit_rejects_right_line_in_left_half():
    t = LaneTracker()
    # A vertical line at x=100 (left half) must be rejected when side='right'
    pts = _vertical_pts(100.0)
    assert t._fit(pts, 'right') is None


def test_fit_accepts_valid_left_line():
    t = LaneTracker()
    # x=120 is inside left half: BEV_MARGIN*0.5=30 … BEV_W*0.55=220
    pts = _vertical_pts(120.0)
    poly = t._fit(pts, 'left')
    assert poly is not None
    assert abs(np.polyval(poly, BEV_H) - 120.0) < 2.0


def test_fit_accepts_valid_right_line():
    t = LaneTracker()
    # x=300 is inside right half: BEV_W*0.45=180 … BEV_W-BEV_MARGIN*0.5=370
    pts = _vertical_pts(300.0)
    poly = t._fit(pts, 'right')
    assert poly is not None
    assert abs(np.polyval(poly, BEV_H) - 300.0) < 2.0


def test_fit_accepts_mild_curve():
    t = LaneTracker()
    # a=0.005 < 0.01 threshold; x_base lands in left half
    pts = _curved_pts(a=0.005, x0=100.0)
    assert t._fit(pts, 'left') is not None


# ── bev_lateral() ─────────────────────────────────────────────────────────────

def test_bev_lateral_ego_centred():
    # Left at x=100, right at x=300 → ego (x=200) is centred → d_left = LANE_WIDTH/2
    t = _tracker_with_polys(left_x=100.0, right_x=300.0)
    d, valid = t.bev_lateral()
    assert valid
    assert abs(d - LANE_WIDTH / 2.0) < 0.05


def test_bev_lateral_missing_left_poly():
    t = LaneTracker()
    t.right_poly = np.array([0.0, 0.0, 300.0])
    _, valid = t.bev_lateral()
    assert not valid


def test_bev_lateral_missing_right_poly():
    t = LaneTracker()
    t.left_poly = np.array([0.0, 0.0, 100.0])
    _, valid = t.bev_lateral()
    assert not valid


def test_bev_lateral_invalid_lane_too_narrow():
    # Lane width < BEV_W * 0.10 = 40 px → invalid
    t = _tracker_with_polys(left_x=190.0, right_x=210.0)  # only 20 px wide
    _, valid = t.bev_lateral()
    assert not valid


def test_bev_lateral_invalid_lane_too_wide():
    # Lane width > BEV_W * 0.70 = 280 px → invalid
    t = _tracker_with_polys(left_x=10.0, right_x=390.0)   # 380 px wide
    _, valid = t.bev_lateral()
    assert not valid


# ── update() — EMA and stale counter ─────────────────────────────────────────

def test_update_ema_smooths_toward_new_poly():
    t = LaneTracker(alpha=0.5)
    # Seed with a line at x=100
    t.update(_vertical_pts(100.0), _vertical_pts(300.0))
    poly0_c = float(t.left_poly[2])   # constant term ≈ 100

    # Next frame: line moves to x=200; with alpha=0.5, EMA → midpoint
    t.update(_vertical_pts(200.0), _vertical_pts(300.0))
    poly1_c = float(t.left_poly[2])
    assert 100.0 < poly1_c < 200.0


def test_stale_counter_drops_poly_after_limit():
    t = LaneTracker()
    t.update(_vertical_pts(120.0), _vertical_pts(300.0))
    assert t.left_poly is not None

    # Feed empty point lists for STALE_LIMIT + 1 frames
    for _ in range(LaneTracker.STALE_LIMIT + 1):
        t.update([], [])

    assert t.left_poly is None


def test_stale_counter_resets_on_good_detection():
    t = LaneTracker()
    t.update(_vertical_pts(120.0), _vertical_pts(300.0))

    # Go almost stale
    for _ in range(LaneTracker.STALE_LIMIT - 1):
        t.update([], [])
    assert t.left_poly is not None   # still alive

    # Good frame resets counter
    t.update(_vertical_pts(120.0), _vertical_pts(300.0))
    assert t._left_stale == 0
