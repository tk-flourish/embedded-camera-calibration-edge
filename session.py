"""Per-connection calibration state and command handlers.

One CalibrationSession exists per client connection. It drives the camera
through the timed structured-light sequences (INIT / INIT_SUBPIX), stores the
per-region capture history, and answers the decode queries (DATA / DATA_SUBPIX)
using the pure helpers in ``decoding``.
"""

import logging
import struct
import time
from collections import defaultdict, deque
from io import BytesIO
from typing import Literal, NamedTuple

import numpy as np

from camera import Camera
from connection import ByteReader, ClientConnection, Request
from decoding import (
    AREA_MIN_PIXELS,
    MIN_LINE_INTENSITY,
    PixelValues,
    Regions,
    compare_lists,
    decode_gray_code,
    find_projector_areas,
    fit_distribution,
    get_values_of_area,
    mean_luma,
)

logger = logging.getLogger(__name__)


class ProjectorSpan(NamedTuple):
    """The line-shift probe spans for one projector."""

    x_span: range     # projector x coordinates probed (vertical line shifted in x)
    y_span: range     # projector y coordinates probed (horizontal line shifted in y)
    y_starts_on: int  # index in the per-frame value list where the y samples begin


class CalibrationSession:
    """Per-connection state and command handlers for one calibration client."""

    def __init__(self, camera: Camera) -> None:
        # Connection-wide
        self.camera = camera
        self.areas: Regions = {}

        # INIT / DATA phase
        self.requested_x_capture_count: int = 0
        self.requested_y_capture_count: int = 0
        self.graycode_history: dict[int, list[PixelValues]] = {}
        self.capture_done: bool = False

        # INIT_SUBPIX / DATA_SUBPIX phase
        self.spans_by_proj: list[ProjectorSpan] = []
        self.line_shift_values: defaultdict[int, list[float]] = defaultdict(list)
        self.subpix_done: bool = False

    def dispatch(self, conn: ClientConnection, request: Request) -> None:
        """Route a request to the handler for its command."""
        command = request.command
        if command == "RTT":
            conn.send("OK")
        elif command == "CAPTURE":
            self.handle_capture(conn)
        elif command == "INIT":
            self.handle_init(conn, request)
        elif command == "DATA":
            self.handle_data(conn)
        elif command == "INIT_SUBPIX":
            self.handle_init_subpix(conn, request)
        elif command == "DATA_SUBPIX":
            self.handle_data_subpix(conn)

    def handle_capture(self, conn: ClientConnection) -> None:
        """CAPTURE: capture a single frame and reply with its PNG bytes."""
        capture = self.camera.capture_pil()
        buffer = BytesIO()
        capture.save(buffer, format="PNG")
        conn.send("OK", buffer.getvalue())

    def handle_init(self, conn: ClientConnection, request: Request) -> None:
        """INIT: run the white/black + ID + Gray-code capture sequence.

        Payload (big-endian uint32): id_bits, x_bits, y_bits, followed by
        ``2 + 2 * (id_bits + x_bits + y_bits)`` capture timestamps in ms. The two
        leading frames are the all-white and all-black position frames; the rest
        are complementary pattern pairs. Replies "OK" immediately, then captures
        on the schedule. Results are fetched separately via DATA.
        """
        reader = ByteReader(request.data)
        requested_id_capture_count = 2 * reader.read_u32()
        self.requested_x_capture_count = 2 * reader.read_u32()
        self.requested_y_capture_count = 2 * reader.read_u32()
        timestamps: deque[int] = deque()
        capture_total = 2 + requested_id_capture_count + self.requested_x_capture_count + self.requested_y_capture_count
        for _ in range(capture_total):
            timestamps.append(reader.read_u32())
        capture_count: int = 0
        self.areas = {}
        capture_stage: Literal['position', 'id', 'xy'] = 'position'
        last_capture: np.ndarray | None = None
        id_map: dict[int, int] = {}
        self.graycode_history = {}
        start_time: int = time.time_ns() // 1_000_000

        conn.send("OK")
        logger.debug("INIT schedule: %d frames (%d/%d/%d ID/X/Y bits)",
                     capture_total, requested_id_capture_count // 2,
                     self.requested_x_capture_count // 2, self.requested_y_capture_count // 2)

        while len(timestamps) != 0:
            timestamp = timestamps.popleft()
            now = time.time_ns() // 1_000_000
            diff = timestamp - (now - start_time)
            if diff > 0:
                time.sleep(diff / 1_000.)
            else:
                logger.warning("capture %dms behind schedule", abs(diff))
            if capture_stage == 'position':
                if last_capture is None:
                    last_capture = self.camera.capture()
                    logger.debug("captured white")
                    continue
                else:
                    black_result = self.camera.capture()
                    logger.debug("captured black")
                    self.areas = find_projector_areas(last_capture, black_result, AREA_MIN_PIXELS)
                    logger.info("found %d lit regions", len(self.areas))
                    logger.debug("region sizes: %s", { area_id: len(members) for area_id, members in self.areas.items() })

                    capture_stage = 'id'
                    last_capture = None
                    id_map = { area_key: 0 for area_key in self.areas }
                    continue
            if capture_stage == 'id':
                if capture_count % 2 == 0:
                    last_capture = self.camera.capture()
                else:
                    capture = self.camera.capture()
                    for area_key in self.areas:
                        former = get_values_of_area(last_capture, self.areas[area_key])
                        latter = get_values_of_area(capture, self.areas[area_key])
                        id_map[area_key] <<= 1
                        id_map[area_key] += 0 if compare_lists(former, latter) < 0 else 1
                    if capture_count == requested_id_capture_count - 1:
                        new_areas: Regions = {}
                        for area_key in self.areas:
                            decoded_id = id_map[area_key]
                            if decoded_id not in new_areas:
                                new_areas[decoded_id] = self.areas[area_key]
                            else:
                                new_areas[decoded_id].extend(self.areas[area_key])
                        self.areas = new_areas
                        logger.info("decoded %d projector IDs", len(self.areas))
                        logger.debug("region sizes: %s", { area_id: len(members) for area_id, members in self.areas.items() })
                        last_capture = None
                        capture_count = 0
                        capture_stage = 'xy'
                        for area_id in self.areas:
                            self.graycode_history[area_id] = []
                capture_count += 1
                continue
            if capture_stage == 'xy':
                capture = self.camera.capture()
                for area_id, area_center in self.areas.items():
                    self.graycode_history[area_id].append(get_values_of_area(capture, area_center))
                capture_count += 1
        self.capture_done = True

    def handle_data(self, conn: ClientConnection) -> None:
        """DATA: reply with the decoded result per region.

        For each region, decodes the x and y Gray codes from the recorded
        pattern history and computes the region centroid, then replies with a
        flat big-endian uint32 array of 5 values per region:
        id, x, y, center_x, center_y.
        """
        if not self.capture_done:
            logger.warning("DATA requested before capture finished")
            conn.send("NOT_FINISHED_OR_STARTED")
            return
        results: list[int] = []
        for area_id, history in self.graycode_history.items():
            x = 0
            for i in range(0, self.requested_x_capture_count // 2):
                x <<= 1
                x += 0 if compare_lists(history[2 * i], history[2 * i + 1]) < 0 else 1
            y = 0
            for i in range(self.requested_x_capture_count // 2, (self.requested_x_capture_count + self.requested_y_capture_count) // 2):
                y <<= 1
                y += 0 if compare_lists(history[2 * i], history[2 * i + 1]) < 0 else 1

            area_center_x = 0
            area_center_y = 0
            for pixel in self.areas[area_id]:
                area_center_y += pixel[0]
                area_center_x += pixel[1]
            area_center_y /= len(self.areas[area_id])
            area_center_x /= len(self.areas[area_id])
            results.extend([area_id, decode_gray_code(x), decode_gray_code(y), int(area_center_x), int(area_center_y)])
        raw_data = bytearray()
        for item in results:
            raw_data.extend(item.to_bytes(4, 'big'))
        conn.send("OK", raw_data)
        logger.info("DATA: returned %d regions", len(results) // 5)

    def handle_init_subpix(self, conn: ClientConnection, request: Request) -> None:
        """INIT_SUBPIX: run the line-shift capture sequence.

        Payload (big-endian uint32): entry_count, then per projector entry five
        values (x_span_start, x_span_end, y_span_start, y_span_end, y_starts_on),
        then timestamp_count and that many capture timestamps in ms. For each
        frame, the per-region mean luma is recorded; results are fetched via
        DATA_SUBPIX.
        """
        start_time: int = time.time_ns() // 1_000_000

        reader = ByteReader(request.data)
        entry_count = reader.read_u32()

        self.spans_by_proj = []

        for _ in range(entry_count):
            x_span_start = reader.read_u32()
            x_span_end = reader.read_u32()
            y_span_start = reader.read_u32()
            y_span_end = reader.read_u32()
            y_starts_on = reader.read_u32()
            x_span = range(x_span_start, x_span_end + 1)
            y_span = range(y_span_start, y_span_end + 1)
            self.spans_by_proj.append(ProjectorSpan(x_span, y_span, y_starts_on))
        timestamps: deque[int] = deque()

        timestamp_count = reader.read_u32()
        for _ in range(timestamp_count):
            timestamps.append(reader.read_u32())
        self.subpix_done = False

        self.line_shift_values = defaultdict(list)

        logger.info("INIT_SUBPIX: %d entries, %d frames", entry_count, timestamp_count)
        conn.send("OK")

        while len(timestamps) != 0:
            timestamp = timestamps.popleft()
            now = time.time_ns() // 1_000_000
            diff = timestamp - (now - start_time)
            if diff > 0:
                time.sleep(diff / 1_000.)
            else:
                logger.warning("capture %dms behind schedule", abs(diff))

            capture = self.camera.capture()
            for area_id, area_center in self.areas.items():
                values = get_values_of_area(capture, area_center)
                self.line_shift_values[area_id].append(mean_luma(values))
        self.subpix_done = True

    def handle_data_subpix(self, conn: ClientConnection) -> None:
        """DATA_SUBPIX: reply with the subpixel peak per projector.

        Fits a Gaussian to each projector's x and y line-shift intensity
        profiles and replies, per projector, with id (big-endian uint32) followed
        by mu_x and mu_y (big-endian float64). A projector whose signal is too
        weak yields NaN.
        """
        if not self.subpix_done:
            logger.warning("DATA_SUBPIX requested before capture finished")
            conn.send("NOT_FINISHED_OR_STARTED")
            return
        raw_result = bytearray()
        for projector_id, values in self.line_shift_values.items():
            if projector_id >= len(self.spans_by_proj):
                continue
            x_span, y_span, y_starts_on = self.spans_by_proj[projector_id]
            if x_span.start == 0 and y_span.start == 0:
                continue
            x_profile: dict[int, float] = { x: values[i] for i, x in enumerate(x_span) }
            y_profile: dict[int, float] = { y: values[i + y_starts_on] for i, y in enumerate(y_span) }
            logger.debug("projector %d: x=%s y=%s", projector_id, x_profile, y_profile)
            if all(item < MIN_LINE_INTENSITY for item in x_profile.values()) or all(item < MIN_LINE_INTENSITY for item in y_profile.values()):
                mu_x = float('nan')
                mu_y = float('nan')
            else:
                try:
                    px, _ = fit_distribution(x_profile)
                    py, _ = fit_distribution(y_profile)
                    mu_x, mu_y = px[1], py[1]
                except Exception:
                    logger.warning("subpixel fit failed for projector %d", projector_id)
                    mu_x = float('nan')
                    mu_y = float('nan')
            logger.debug("projector %d: mu=(%s, %s)", projector_id, mu_x, mu_y)
            raw_result.extend(projector_id.to_bytes(4, 'big'))
            raw_result.extend(struct.pack('>d', mu_x))
            raw_result.extend(struct.pack('>d', mu_y))
        conn.send("OK", raw_result)
        logger.info("DATA_SUBPIX: returned %d projectors", len(raw_result) // 20)
