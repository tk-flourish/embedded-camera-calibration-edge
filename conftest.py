"""Test bootstrap.

``main`` imports ``picamera2`` and ``libcamera``, which only exist on the
Raspberry Pi. Inject lightweight stand-ins into ``sys.modules`` before any test
imports ``main`` so the pure decoding logic can be exercised off-device. The
stubs only need to satisfy import and ``Camera.__init__``; tests that touch the
camera use their own fakes instead.
"""

import sys
import types


def _install_hardware_stubs() -> None:
    if "libcamera" not in sys.modules:
        libcamera = types.ModuleType("libcamera")

        class _AfModeEnum:
            Manual = 0

        controls = types.SimpleNamespace(AfModeEnum=_AfModeEnum)
        libcamera.controls = controls
        sys.modules["libcamera"] = libcamera

    if "picamera2" not in sys.modules:
        picamera2 = types.ModuleType("picamera2")

        class _Picamera2:  # pragma: no cover - never instantiated in tests
            def __init__(self, *a, **k):
                raise RuntimeError("real Picamera2 is unavailable in tests")

        picamera2.Picamera2 = _Picamera2
        sys.modules["picamera2"] = picamera2


_install_hardware_stubs()
