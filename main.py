from collections import defaultdict, deque
import datetime
from io import BytesIO
import os
from pathlib import Path
import struct
from threading import Thread
import time
from typing import Literal, cast
import PIL
from PIL.Image import Image
from picamera2 import Picamera2
import cv2
import numpy as np
from scipy.optimize import curve_fit
from libcamera import controls

from unionfind import UnionFind
from connection import ServerStream, ClientConnection, Request

class Camera:
    _picam2: Picamera2
    capture_dir = Path("~/Captured/").expanduser()

    def __init__(self) -> None:
        self._picam2 = Picamera2()
        camera_config = self._picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores", queue=False)
        self._picam2.configure(camera_config)
        # self._picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0, "ExposureTime": 16666, "AnalogueGain": 1.0})
        self._picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0, "ExposureTime": 333_333, "AnalogueGain": 1.0})
        self._picam2.start()

    def capture(self) -> np.ndarray:
        image = self._picam2.capture_image()
        opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        os.makedirs(str(self.capture_dir), exist_ok=True)
        now = datetime.datetime.now()
        filename = now.strftime("%Y%m%d-%H%M%S") + f"-{now.microsecond//1000:03d}.png"
        cv2.imwrite(str(self.capture_dir / filename), opencvImage)

        return opencvImage
    
    def capture_pil(self) -> Image:
        return self._picam2.capture_image()
    
def safe_timestamp() -> str:
    return datetime.datetime.now().isoformat().replace(':', '-')
    
def find_projector_areas(white: np.ndarray, black: np.ndarray, area_threshold: int) -> dict[int, list[tuple[int, int]]]:
    # Signed difference white - black (cast to int16 so negative values are kept)
    diff = white.astype(np.int16) - black.astype(np.int16)

    # Per-pixel channel sum (axis=2 sums over RGB)
    summed = np.sum(diff, axis=2)

    # Threshold at 1/16 of the maximum summed value
    threshold = np.max(summed) * 1 / 16

    # Keep values at or above the threshold, set the rest to 0
    result = np.where(summed >= threshold, summed, 0)

    light_points = [tuple(item) for item in np.array(np.nonzero(result)).T]

    reverse_lookup = {position: index for index, position in enumerate(light_points) }

    uf = UnionFind(len(light_points))

    for index, point in enumerate(light_points):
        for neighbour in [(point[0] - 1, point[1]), (point[0] + 1, point[1]), (point[0], point[1] - 1), (point[0], point[1] + 1), 
        (point[0] - 1, point[1] - 1), (point[0] + 1, point[1] + 1), (point[0] + 1, point[1] - 1), (point[0] - 1, point[1] + 1)]:
            neighbour_index = reverse_lookup.get(neighbour)
            if neighbour_index is not None:
                uf.union(index, neighbour_index)
    
    areas: dict[int, list[int]] = {}

    for i in range(len(light_points)):
        key = uf.find(i)
        if not key in areas:
            areas[key] = []
        areas[key].append(i)
    
    return {index: [light_points[index] for index in area] for index, area in areas.items() if len(area) >= area_threshold}

def get_values_of_area(target: np.ndarray, area: list[tuple[int, int]]) -> list[list[int]]:
    return [target[position].tolist() for position in area]

def decode_gray_code(code: int) -> int:
    a = code
    b = code
    while a != 0:
        a >>= 1
        b ^= a
    return b

def compare_lists(list1: list[int], list2: list[int]):
    count1 = 0
    count2 = 0

    for a, b in zip(list1, list2):
        a_count = 0
        b_count = 0
        for a1, b1 in zip(a, b):
            if a1 > b1:
                a_count += 1
            elif b1 > a1:
                b_count += 1
        if a_count > b_count:
            count1 += 1
        elif b_count > a_count:
            count2 += 1

    if count1 > count2:
        return 1
    elif count2 > count1:
        return -1
    else:
        return 0

# --- Gaussian function ---
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

# --- Curve fitting ---
def fit_distribution(data_dict):
    xs = np.array(list(data_dict.keys()))
    ys = np.array(list(data_dict.values()))

    # Trim the zero tails
    nonzero_indices = np.where(ys > 0)[0]
    start, end = nonzero_indices[0], nonzero_indices[-1]
    xs = xs[start:end + 1]
    ys = ys[start:end + 1]

    # Initial parameter estimates
    A0 = ys.max() - ys.min()
    mu0 = xs[ys.argmax()]
    sigma0 = 1.0
    C0 = ys.min()

    popt, pcov = curve_fit(gaussian, xs, ys, p0=[A0, mu0, sigma0, C0])
    A, mu, sigma, C = popt
    return popt, np.sqrt(np.diag(pcov))



def deal_with_connection(conn: ClientConnection, camera: Camera):
    with conn:
        requested_id_capture_count: int = 0
        requested_x_capture_count: int = 0
        requested_y_capture_count: int = 0
        graycode_history: dict[int, list[float]] = {}
        is_finished: bool = False

        while True:
            request = conn.read()
            if request:
                print(f"Request Accepted: {request.command}")
            if request is None:
                print("disconnected, breaking...")
                break
            if request.command == "RTT":
                conn.send("OK")
                continue
            if request.command == "CAPTURE":
                capture = camera.capture_pil()
                buffer = BytesIO()
                capture.save(buffer, format="PNG")
                conn.send("OK", buffer.getvalue())
            if request.command == "INIT":
                requested_id_capture_count = 2 * int.from_bytes(request.data[0:4], 'big')
                requested_x_capture_count = 2 * int.from_bytes(request.data[4:8], 'big')
                requested_y_capture_count = 2 * int.from_bytes(request.data[8:12], 'big')
                timestamps: deque[int] = deque()
                for i in range(2 + requested_id_capture_count + requested_x_capture_count + requested_y_capture_count):
                    timestamps.append(int.from_bytes(request.data[(12 + i * 4):(12 + (i + 1) * 4)], 'big'))
                capture_count: int = 0
                areas: dict[int, list[tuple[int, int]]] = {}
                capture_stage: Literal['position', 'id', 'xy'] = 'position'
                last_capture: np.ndarray | None = None
                id_map: dict[int, int] = {}
                graycode_history = {}
                start_time: int = time.time_ns() // 1_000_000

                print(f"initialized")
                conn.send("OK")
                print(timestamps)

                while len(timestamps) != 0:
                    timestamp = timestamps.popleft()
                    now = time.time_ns() // 1_000_000
                    diff = timestamp - (now - start_time)
                    if diff > 0:
                        time.sleep(diff / 1_000.)
                    else:
                        print(f"{abs(diff)}ms behind")
                    if capture_stage == 'position':
                        if last_capture is None:
                            last_capture = camera.capture()
                            cv2.imwrite(f"./Captures/{safe_timestamp()}-0-White.png", last_capture)
                            print("captured white")
                            continue
                        else:
                            black_result = camera.capture()
                            cv2.imwrite(f"./Captures/{safe_timestamp()}-1-Black.png", black_result)
                            print("captured black")
                            areas = find_projector_areas(last_capture, black_result, 30)
                            print(f"area calculated: { { id: len(area) for id, area in areas.items() } }")

                            capture_stage = 'id'
                            last_capture = None
                            id_map = { area_index: 0 for area_index in areas }
                            continue
                    if capture_stage == 'id':
                        if capture_count % 2 == 0:
                            last_capture = camera.capture()
                        else:
                            capture = camera.capture()
                            cv2.imwrite(f"./Captures/{safe_timestamp()}-2-ID-{capture_count}.png", capture)
                            for index in areas:
                                former = get_values_of_area(last_capture, areas[index])
                                latter = get_values_of_area(capture, areas[index])
                                # print(former, latter)
                                id_map[index] <<= 1
                                id_map[index] += 0 if compare_lists(former, latter) < 0 else 1
                            if capture_count == requested_id_capture_count - 1:
                                new_areas: dict[int, list[tuple[int, int]]] = {}
                                for index in areas:
                                    id = id_map[index]
                                    if not id in new_areas:
                                        new_areas[id] = areas[index]
                                    else:
                                        new_areas[id].extend(areas[index])
                                areas = new_areas
                                print(f"added index: { { id: len(area) for id, area in areas.items() } }")
                                last_capture = None
                                capture_count = 0
                                capture_stage = 'xy'
                                for id in areas:
                                    graycode_history[id] = []
                        capture_count += 1
                        continue
                    if capture_stage == 'xy':
                        capture = camera.capture()
                        for id, area_center in areas.items():
                            graycode_history[id].append(get_values_of_area(capture, area_center))
                        capture_count += 1
                is_finished = True
            if request.command == "DATA":
                if not is_finished:
                    conn.send("NOT_FINISHED_OR_STARTED")
                    continue
                results: list[int] = []
                for id, history in graycode_history.items():
                    x = 0
                    for i in range(0, requested_x_capture_count // 2):
                        x <<= 1
                        x += 0 if compare_lists(history[2 * i], history[2 * i + 1]) < 0 else 1
                    y = 0
                    for i in range(requested_x_capture_count // 2, (requested_x_capture_count + requested_y_capture_count) // 2):
                        y <<= 1
                        y += 0 if compare_lists(history[2 * i], history[2 * i + 1]) < 0 else 1
                    
                    area_center_x = 0
                    area_center_y = 0
                    for pixel in areas[id]:
                        area_center_y += pixel[0]
                        area_center_x += pixel[1]
                    area_center_y /= len(areas[id])
                    area_center_x /= len(areas[id])
                    results.extend([id, decode_gray_code(x), decode_gray_code(y), int(area_center_x), int(area_center_y)])
                raw_data = bytearray([])
                for item in results:
                    raw_data.extend(item.to_bytes(4, 'big'))
                conn.send("OK", raw_data)
                print("sent data")
            if request.command == "INIT_SUBPIX":
                start_time: int = time.time_ns() // 1_000_000

                current_cursor = 0
                def read_next():
                    nonlocal current_cursor
                    value = int.from_bytes(request.data[current_cursor:current_cursor + 4], 'big')
                    current_cursor += 4
                    return value
                
                
                entry_count = read_next()

                spans_by_proj: list[tuple[range, range, int]] = []

                for i in range(entry_count):
                    x_span_start = read_next()
                    x_span_end = read_next()
                    y_span_start = read_next()
                    y_span_end = read_next()
                    y_starts_on = read_next()
                    x_span = range(x_span_start, x_span_end + 1)
                    y_span = range(y_span_start, y_span_end + 1)
                    spans_by_proj.append((x_span, y_span, y_starts_on))
                timestamps: deque[int] = deque()

                timespamp_count = read_next()
                for i in range(timespamp_count):
                    timestamps.append(read_next())
                print(len(timestamps))
                is_finished = False

                gained_values: defaultdict[int, list[float]] = defaultdict(list)
                
                print(f"initialized")
                conn.send("OK")

                while len(timestamps) != 0:
                    timestamp = timestamps.popleft()
                    now = time.time_ns() // 1_000_000
                    diff = timestamp - (now - start_time)
                    if diff > 0:
                        time.sleep(diff / 1_000.)
                    else:
                        print(f"{abs(diff)}ms behind")
                    
                    capture = camera.capture()
                    for id, area_center in areas.items():
                        values = get_values_of_area(capture, area_center)
                        gray_scaled = [0.299 * item[2] + 0.587 * item[1] + 0.114 * item[0] for item in values]
                        average = sum(gray_scaled) / len(gray_scaled)
                        gained_values[id].append(average)
                is_finished = True
            if request.command == "DATA_SUBPIX":
                if not is_finished:
                    conn.send("NOT_FINISHED_OR_STARTED")
                    continue
                raw_result = bytearray([])
                for projector_id, values in gained_values.items():
                    if projector_id >= len(spans_by_proj):
                        continue
                    x_span, y_span, y_starts_on = spans_by_proj[projector_id]
                    if x_span.start == 0 and y_span.start == 0:
                        continue
                    x_gained: dict[int, float] = { x: values[i] for i, x in enumerate(x_span) }
                    y_gained: dict[int, float] = { y: values[i + y_starts_on] for i, y in enumerate(y_span) }
                    print(projector_id)
                    print(x_gained)
                    print(y_gained)
                    if all(item < 20 for item in x_gained.values()) or all(item < 20 for item in y_gained.values()):
                        mu_x = float('nan')
                        mu_y = float('nan')
                    else:
                        try:
                            px, _ = fit_distribution(x_gained)
                            py, _ = fit_distribution(y_gained)
                            mu_x, mu_y = px[1], py[1]
                        except Exception:
                            mu_x = float('nan')
                            mu_y = float('nan')
                    print(mu_x, mu_y)
                    raw_result.extend(projector_id.to_bytes(4, 'big'))
                    raw_result.extend(struct.pack('>d', mu_x))
                    raw_result.extend(struct.pack('>d', mu_y))
                conn.send("OK", raw_result)


def main():
    camera = Camera()

    with ServerStream(58919) as s:
        while True:
            conn, addr = s.accept()
            print(f"connected by {addr}")
            thread = Thread(target=deal_with_connection, args=(conn, camera))
            thread.start()

if __name__ == "__main__":
    main()

