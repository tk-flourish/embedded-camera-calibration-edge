"""Characterization tests for the line-shift subpixel path.

Pins the INIT_SUBPIX capture -> DATA_SUBPIX Gaussian-fit wire output, including
the weak-signal NaN fallback and the not-finished guard.
"""

import math
import struct

import numpy as np
import pytest

import decoding
import session as calib
from connection import Request

REGION = [(r, c) for r in range(5, 11) for c in range(5, 11)]  # 36 px


class FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, status, data=b""):
        self.sent.append((status, bytes(data)))


class FakeCamera:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def capture(self):
        f = self._frames[self._i]
        self._i += 1
        return f


def frame(value):
    """Uniform region -> mean luma equals ``value`` (luma weights sum to 1)."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:11, 5:11, :] = value
    return img


def u32(n):
    return n.to_bytes(4, "big")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(calib.time, "sleep", lambda *_: None)


def test_data_subpix_before_capture_reports_not_finished():
    session = calib.CalibrationSession(camera=object())
    conn = FakeConn()
    session.dispatch(conn, Request("DATA_SUBPIX", b""))
    assert conn.sent == [("NOT_FINISHED_OR_STARTED", b"")]


def test_init_then_data_subpix_fits_peak():
    # Two Gaussian profiles (x then y), both peaking at projector coord 15.
    x_span = range(10, 21)   # 11 samples -> values[0..10]
    y_starts_on = 11
    y_span = range(10, 21)   # 11 samples -> values[11..21]

    def g(i, center=5):
        return int(200 * math.exp(-((i - center) ** 2) / (2 * 2.0**2)) + 30)

    values = [g(i) for i in range(11)] + [g(j) for j in range(11)]  # 22 frames
    frames = [frame(v) for v in values]

    session = calib.CalibrationSession(FakeCamera(frames))
    session.areas = {1: REGION}  # set by a prior INIT in real use

    # entry 0 is the (0,0) sentinel span (skipped); entry 1 is for projector id 1.
    payload = u32(2)
    payload += u32(0) + u32(0) + u32(0) + u32(0) + u32(0)
    payload += u32(x_span.start) + u32(x_span.stop - 1)
    payload += u32(y_span.start) + u32(y_span.stop - 1) + u32(y_starts_on)
    payload += u32(22) + b"".join(u32(0) for _ in range(22))

    conn = FakeConn()
    session.dispatch(conn, Request("INIT_SUBPIX", payload))
    assert conn.sent == [("OK", b"")]
    assert session.subpix_done is True

    conn2 = FakeConn()
    session.dispatch(conn2, Request("DATA_SUBPIX", b""))
    status, data = conn2.sent[0]
    assert status == "OK"
    assert len(data) == 20  # one projector: u32 id + 2x f64

    pid = int.from_bytes(data[0:4], "big")
    mu_x = struct.unpack(">d", data[4:12])[0]
    mu_y = struct.unpack(">d", data[12:20])[0]
    assert pid == 1
    assert mu_x == pytest.approx(15.0, abs=0.1)
    assert mu_y == pytest.approx(15.0, abs=0.1)


def test_weak_signal_yields_nan():
    # All probed intensities below MIN_LINE_INTENSITY -> NaN, no fit attempted.
    session = calib.CalibrationSession(camera=object())
    session.subpix_done = True
    session.spans_by_proj = [
        calib.ProjectorSpan(range(0, 0), range(0, 0), 0),  # index 0 sentinel
        calib.ProjectorSpan(range(10, 13), range(10, 13), 3),
    ]
    low = decoding.MIN_LINE_INTENSITY - 1
    session.line_shift_values = {1: [low, low, low, low, low, low]}

    conn = FakeConn()
    session.dispatch(conn, Request("DATA_SUBPIX", b""))
    status, data = conn.sent[0]
    assert status == "OK"
    mu_x = struct.unpack(">d", data[4:12])[0]
    mu_y = struct.unpack(">d", data[12:20])[0]
    assert math.isnan(mu_x) and math.isnan(mu_y)
