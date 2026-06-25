"""End-to-end characterization tests for CalibrationSession command handlers.

Drives a full INIT -> DATA sequence through scripted camera frames and a fake
connection, pinning the decoded [id, x, y, center_x, center_y] wire output. This
is the behaviour most at risk from the planned refactors, so it is locked here.
"""

import struct

import numpy as np
import pytest

import session as calib
from connection import Request

REGION_ROWS = (5, 11)  # 6 rows
REGION_COLS = (5, 11)  # 6 cols, 36 px total (>= AREA_MIN_PIXELS)


class FakeConn:
    """Records (status, data) from send(); never touches a socket."""

    def __init__(self):
        self.sent = []

    def send(self, status, data=b""):
        self.sent.append((status, bytes(data)))


class FakeCamera:
    """Yields pre-scripted BGR frames in order."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def capture(self):
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def capture_pil(self):  # pragma: no cover - not used here
        raise NotImplementedError


def frame(region_value):
    """A 20x20 BGR frame with the region set to a uniform value."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[REGION_ROWS[0]:REGION_ROWS[1], REGION_COLS[0]:REGION_COLS[1], :] = region_value
    return img


def u32(n):
    return n.to_bytes(4, "big")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(calib.time, "sleep", lambda *_: None)


def test_rtt_replies_ok():
    session = calib.CalibrationSession(camera=object())
    conn = FakeConn()
    session.dispatch(conn, Request("RTT", b""))
    assert conn.sent == [("OK", b"")]


def test_data_before_capture_reports_not_finished():
    session = calib.CalibrationSession(camera=object())
    conn = FakeConn()
    session.dispatch(conn, Request("DATA", b""))
    assert conn.sent == [("NOT_FINISHED_OR_STARTED", b"")]


def test_init_then_data_decodes_region():
    # 1 bit each of ID/X/Y -> 8 frames total: white, black, then 3 complementary
    # pairs. "bright then dark" decodes to bit 1; "dark then bright" to bit 0.
    BRIGHT, DARK = 90, 0
    frames = [
        frame(BRIGHT),  # 0 white (position)
        frame(DARK),    # 1 black (position)
        frame(BRIGHT),  # 2 ID former (bright) -> id bit 1
        frame(DARK),    # 3 ID latter
        frame(BRIGHT),  # 4 X former (bright) -> x gray bit 1
        frame(DARK),    # 5 X latter
        frame(DARK),    # 6 Y former (dark)  -> y gray bit 0
        frame(BRIGHT),  # 7 Y latter
    ]
    session = calib.CalibrationSession(FakeCamera(frames))
    conn = FakeConn()

    payload = u32(1) + u32(1) + u32(1) + b"".join(u32(0) for _ in range(8))
    session.dispatch(conn, Request("INIT", payload))
    assert conn.sent == [("OK", b"")]  # INIT replies OK immediately
    assert session.capture_done is True

    conn2 = FakeConn()
    session.dispatch(conn2, Request("DATA", b""))
    status, data = conn2.sent[0]
    assert status == "OK"

    values = list(struct.unpack(">5I", data))
    decoded_id, x, y, center_x, center_y = values
    # decode_gray_code(1)=1, decode_gray_code(0)=0; centroid of rows/cols 5..10.
    assert decoded_id == 1
    assert x == 1
    assert y == 0
    assert (center_x, center_y) == (7, 7)
