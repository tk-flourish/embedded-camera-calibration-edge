"""Tests that Camera serialises concurrent Picamera2 access.

The server shares one Camera across per-connection threads; concurrent hardware
reads would corrupt frames. Camera is built via __new__ with a fake picam2 so no
real hardware is needed.
"""

import threading
import time

import numpy as np

import camera


class TrackingPicam2:
    """Records the peak number of threads inside capture_image at once."""

    def __init__(self):
        self.active = 0
        self.max_active = 0

    def capture_image(self):
        # Deliberately non-atomic: if access is serialised these never overlap,
        # so max_active stays 1; if not, overlap is observed.
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        self.active -= 1
        return np.zeros((4, 4, 3), np.uint8)


def _make_camera(lock):
    cam = camera.Camera.__new__(camera.Camera)
    cam._lock = lock
    cam._picam2 = TrackingPicam2()
    return cam


def _hammer(method, n=10):
    threads = [threading.Thread(target=method) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_capture_is_serialised_under_lock():
    cam = _make_camera(threading.Lock())
    _hammer(cam.capture)
    assert cam._picam2.max_active == 1


def test_capture_pil_is_serialised_under_lock():
    cam = _make_camera(threading.Lock())
    _hammer(cam.capture_pil)
    assert cam._picam2.max_active == 1


def test_without_lock_access_overlaps():
    # Negative control: a no-op lock lets the threads overlap, confirming the
    # tracking actually detects concurrency (so the lock tests are meaningful).
    class NoLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    cam = _make_camera(NoLock())
    _hammer(cam.capture)
    assert cam._picam2.max_active > 1
