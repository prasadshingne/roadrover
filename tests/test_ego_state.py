"""
Tests for EgoStateEstimator — speed, heading, yaw rate, and acceleration
derived from GPS /vel messages.

All assertions are on the estimator's internal state variables
(yaw_rate, lon_accel, etc.) rather than the returned ROS messages,
so these tests carry no dependency on the mocked ROS types.
"""
import math
from unittest.mock import MagicMock

import pytest

from process_bag import EgoStateEstimator

MIN_SPEED = EgoStateEstimator._MIN_SPEED
ALPHA     = EgoStateEstimator._ALPHA


# ── helpers ───────────────────────────────────────────────────────────────────

def _vel(vx_east: float, vy_north: float) -> MagicMock:
    msg = MagicMock()
    msg.twist.linear.x = vx_east
    msg.twist.linear.y = vy_north
    return msg


def _ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


# ── speed calculation ─────────────────────────────────────────────────────────

def test_speed_3_4_triangle():
    est = EgoStateEstimator()
    est.update(_vel(3.0, 4.0), _ns(0.0))
    assert est._speed_prev == pytest.approx(5.0)


def test_speed_pure_east():
    est = EgoStateEstimator()
    est.update(_vel(10.0, 0.0), _ns(0.0))
    assert est._speed_prev == pytest.approx(10.0)


def test_speed_zero():
    est = EgoStateEstimator()
    est.update(_vel(0.0, 0.0), _ns(0.0))
    assert est._speed_prev == pytest.approx(0.0)


# ── heading ───────────────────────────────────────────────────────────────────

def test_heading_pure_east():
    est = EgoStateEstimator()
    est.update(_vel(1.0, 0.0), _ns(0.0))
    assert est._heading_prev == pytest.approx(0.0, abs=1e-9)


def test_heading_pure_north():
    est = EgoStateEstimator()
    est.update(_vel(0.0, 1.0), _ns(0.0))
    assert est._heading_prev == pytest.approx(math.pi / 2)


def test_heading_northeast_diagonal():
    est = EgoStateEstimator()
    est.update(_vel(1.0, 1.0), _ns(0.0))
    assert est._heading_prev == pytest.approx(math.pi / 4)


def test_heading_not_updated_below_min_speed():
    est = EgoStateEstimator()
    # First update fast → sets a heading
    est.update(_vel(5.0, 0.0), _ns(0.0))
    heading_after_fast = est._heading_prev

    # Second update slow → heading must not change
    est.update(_vel(MIN_SPEED * 0.5, 0.0), _ns(1.0))
    assert est._heading_prev == pytest.approx(heading_after_fast)


def test_heading_none_when_always_slow():
    est = EgoStateEstimator()
    est.update(_vel(MIN_SPEED * 0.5, 0.0), _ns(0.0))
    assert est._heading_prev is None


# ── yaw rate and angle unwrapping ─────────────────────────────────────────────

def test_yaw_rate_straight_line():
    est = EgoStateEstimator()
    est.update(_vel(10.0, 0.0), _ns(0.0))
    est.update(_vel(10.0, 0.0), _ns(1.0))
    assert est.yaw_rate == pytest.approx(0.0, abs=1e-9)


def test_yaw_rate_sign_left_turn():
    # Heading changes from east (0) to north-east (π/4) in 1 s → positive yaw rate
    est = EgoStateEstimator()
    est.update(_vel(10.0, 0.0),  _ns(0.0))   # heading = 0
    est.update(_vel(10.0, 10.0), _ns(1.0))   # heading = π/4
    assert est.yaw_rate > 0


def test_yaw_rate_wraps_at_pi():
    # CCW (left) turn crossing the ±π wrap boundary should produce a small positive
    # yaw rate, not a ≈ −2π / +2π jump.
    est = EgoStateEstimator()
    small = 0.05
    # Heading just north of due-west: atan2(+small, -large) ≈ π − small
    est.update(_vel(-10.0,  math.tan(small) * 10), _ns(0.0))
    # Heading just south of due-west (CCW past ±π): atan2(−small, −large) ≈ −(π − small)
    est.update(_vel(-10.0, -math.tan(small) * 10), _ns(1.0))

    # Raw Δheading ≈ −2π; after unwrapping it becomes +2*small ≈ 0.1 rad (left turn).
    expected_raw = 2 * small
    ema_yaw = ALPHA * expected_raw + (1 - ALPHA) * 0.0
    assert est.yaw_rate == pytest.approx(ema_yaw, rel=0.05)


# ── longitudinal acceleration ─────────────────────────────────────────────────

def test_lon_accel_first_ema_step():
    est = EgoStateEstimator()
    est.update(_vel(0.0, 0.0), _ns(0.0))
    est.update(_vel(10.0, 0.0), _ns(1.0))   # Δspeed=10 in 1 s → a_raw=10
    expected = ALPHA * 10.0   # EMA from zero
    assert est.lon_accel == pytest.approx(expected, rel=1e-6)


def test_lon_accel_deceleration_is_negative():
    est = EgoStateEstimator()
    est.update(_vel(10.0, 0.0), _ns(0.0))
    est.update(_vel(0.0,  0.0), _ns(1.0))   # sudden stop
    assert est.lon_accel < 0


# ── lateral acceleration ──────────────────────────────────────────────────────

def test_lat_accel_straight_is_zero():
    est = EgoStateEstimator()
    est.update(_vel(10.0, 0.0), _ns(0.0))
    est.update(_vel(10.0, 0.0), _ns(1.0))
    assert est.lat_accel == pytest.approx(0.0, abs=1e-9)
