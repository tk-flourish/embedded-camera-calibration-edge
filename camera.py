"""Picamera2 wrapper for the fixed calibration rig.

Owns the camera hardware and serialises access to it. The capture server shares
one Camera across per-connection threads, so every hardware read is guarded by a
lock. Importing this module requires the Raspberry Pi packages (picamera2,
libcamera); the decoding logic lives in ``decoding`` precisely so it does not.
"""

import datetime
import os
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from libcamera import controls
from picamera2 import Picamera2
from PIL.Image import Image

# Set SAVE_CAPTURES=1 to archive every captured frame, plus the named
# white/black/ID frames, to disk. Off by default.
SAVE_CAPTURES = os.environ.get("SAVE_CAPTURES", "0") == "1"


class Camera:
    """Wraps Picamera2 with the manual focus/exposure used by the fixed rig."""

    _picam2: Picamera2
    capture_dir = Path("~/Captured/").expanduser()

    def __init__(self) -> None:
        # Serialises Picamera2 access: the server shares one Camera across
        # per-connection threads, and the underlying hardware cannot be read
        # from two threads at once without corrupting frames.
        self._lock = Lock()
        self._picam2 = Picamera2()
        camera_config = self._picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores", queue=False)
        self._picam2.configure(camera_config)
        self._picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0, "ExposureTime": 333_333, "AnalogueGain": 1.0})
        self._picam2.start()

    def capture(self) -> np.ndarray:
        """Capture a frame as a BGR array.

        When SAVE_CAPTURES is enabled, the frame is also archived under
        capture_dir with a timestamped filename.
        """
        with self._lock:
            image = self._picam2.capture_image()
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if SAVE_CAPTURES:
            os.makedirs(str(self.capture_dir), exist_ok=True)
            now = datetime.datetime.now()
            filename = now.strftime("%Y%m%d-%H%M%S") + f"-{now.microsecond//1000:03d}.png"
            cv2.imwrite(str(self.capture_dir / filename), opencv_image)

        return opencv_image

    def capture_pil(self) -> Image:
        """Capture a frame as a PIL image (no disk write)."""
        with self._lock:
            return self._picam2.capture_image()
