"""Embedded-camera capture-and-decode server (Raspberry Pi side).

Runs on the Raspberry Pi carrying an embedded camera and serves a calibration
host over TCP. It drives the camera through a timed structured-light sequence
and, for each illuminated region, decodes which projector lit it (ID), the
projector pixel coordinate (Gray code), and a subpixel refinement (line-shift
Gaussian fit). See README.md for the protocol and the relationship to the paper.

This module is just the server entrypoint; the work lives in:
  camera.py     - Picamera2 hardware wrapper
  decoding.py   - pure structured-light decoding math
  session.py    - per-connection command handlers
  connection.py - TCP framing and payload parsing
"""

import logging
from threading import Thread

from camera import Camera
from connection import ClientConnection, ServerStream
from session import CalibrationSession

logger = logging.getLogger(__name__)

# TCP port the capture server listens on.
SERVER_PORT = 58919


def deal_with_connection(conn: ClientConnection, camera: Camera):
    """Serve one client connection until it disconnects."""
    with conn:
        session = CalibrationSession(camera)

        while True:
            request = conn.read()
            if request:
                logger.debug("request: %s", request.command)
            if request is None:
                logger.info("client disconnected")
                break
            session.dispatch(conn, request)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    camera = Camera()

    with ServerStream(SERVER_PORT) as s:
        logger.info("listening on port %d", SERVER_PORT)
        while True:
            conn, addr = s.accept()
            logger.info("client connected: %s", addr)
            thread = Thread(target=deal_with_connection, args=(conn, camera))
            thread.start()

if __name__ == "__main__":
    main()
