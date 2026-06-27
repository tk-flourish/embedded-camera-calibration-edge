# embedded-camera-calibration-edge

The **edge (on-device) capture-and-decode agent** for multi-projector geometric
calibration using cameras embedded in the calibration target. It runs on a
Raspberry Pi with the embedded camera; a calibration host connects over TCP to
drive the camera through a timed structured-light sequence and retrieve the
decoded projector correspondences.

It is the embedded-device component of the system behind
[*"Breaking the Scalability Limit of Multi-Projector Calibration with Embedded
Cameras"*](https://cvpr.thecvf.com/virtual/2026/oral/40265) (CVPR 2026, oral).
The calibration core (parameter estimation, optical-center homography) runs on a
separate machine — the
[`embedded-camera-calibration`](https://github.com/tk-flourish/embedded-camera-calibration) host.

> **License:** source-available, **All Rights Reserved** (patent pending).
> See [LICENSE](LICENSE).

## What it does

Because the embedded cameras directly receive the projection light, light from
projectors at different positions lands on different camera pixels, so
simultaneously projected patterns can be separated by incident direction. This
agent performs the camera-side capture and decoding:

- **Capture** via `picamera2` (manual focus / exposure for a fixed rig).
- **Light-ray separation** (`find_projector_areas`): white−black difference,
  threshold at 1/16 of the max, 8-neighbour Union-Find connected components, and
  an area filter (≥30 px) to find each projector's illuminated region.
- **Projector-ID decoding**: complementary binary patterns are compared to read a
  ⌈log₂M⌉-bit ID per region, so all projectors can be projected at once.
- **Gray-code decoding**: horizontal/vertical Gray-code patterns are decoded to
  the integer projector pixel coordinate (region pixels are averaged for noise).
- **Subpixel refinement**: line-shift intensity profiles are fit with a Gaussian
  to recover subpixel coordinates.

These steps follow the method described in the paper (§3) and the corresponding
master's thesis (implementation chapter): the threshold, neighbourhood, area
filter, complementary ID patterns, ±5 px line shift and Gaussian fit match that
description.

## Protocol

A minimal length-prefixed TCP protocol (`connection.py`). The host sends a
command with a binary payload; the agent replies with a status string and
optional binary data. Commands:

| Command | Purpose |
| --- | --- |
| `RTT` | Round-trip latency check. |
| `CAPTURE` | Capture one frame and return it as PNG. |
| `INIT` / `DATA` | Run the timed white/black + ID + Gray-code sequence, then return decoded `[id, x, y, center_x, center_y]` per region. |
| `INIT_SUBPIX` / `DATA_SUBPIX` | Run the line-shift sequence and return Gaussian-fit subpixel `(mu_x, mu_y)` per projector. |

Capture timing is driven by a schedule of timestamps supplied by the host so the
agent's captures stay aligned with the host's pattern projection.

## Code layout

The agent is split into cohesive modules so the decoding math stays free of
hardware and I/O (and therefore testable off-device):

| Module | Responsibility |
| --- | --- |
| `main.py` | Server entrypoint: opens the TCP server (port 58919) and spawns one session thread per connection. |
| `connection.py` | Minimal length-prefixed TCP framing (4-byte length + payload) between host and agent. |
| `camera.py` | `picamera2` wrapper for the fixed rig; owns the camera and serialises hardware access behind a lock. |
| `decoding.py` | Pure structured-light math — `find_projector_areas`, ID/Gray-code brightness comparison, per-region luma, Gaussian subpixel fit. No camera or network imports. |
| `session.py` | Per-connection state and command handlers; drives the timed sequences and answers decode queries using `decoding`. |

Only `camera.py` needs the Raspberry Pi packages (`picamera2`, `libcamera`), so
`decoding.py` imports and tests on any machine.

## Running

```
python main.py
```

Starts a TCP server (port 58919) that accepts connections from the calibration
host. Intended to run on the Raspberry Pi that carries the embedded camera.

Progress is reported through the `logging` module (INFO by default; raise the
logger to DEBUG for verbose decode dumps). Set the environment variable
`SAVE_CAPTURES=1` to archive captured frames to disk (off by default).

### Dependencies

Python 3 and the following packages, installed separately (none are bundled):

- `picamera2`, `libcamera` — camera capture (system packages on Raspberry Pi OS)
- `opencv-python` — colour conversion and image I/O
- `numpy` — array operations
- `scipy` — Gaussian curve fitting for subpixel refinement
- `Pillow` — PNG encoding for the `CAPTURE` command

### Tests

The decoding logic is pure (no camera or network), so the test suite runs on any
machine — **no Raspberry Pi or camera required**. `conftest.py` injects
lightweight stand-ins for the `picamera2` / `libcamera` packages so the modules
import off-device.

```
pip install -r requirements-dev.txt
python -m pytest
```

## Notes / limitations

- Targets a fixed calibration rig: focus, exposure and the capture schedule are
  tuned for that setup.
- One embedded camera per device; the host runs one agent per Raspberry Pi.
- This is a refactored release. The **original implementation used to produce the
  paper's results is preserved as this repository's initial commit** — check it
  out to see the pre-refactor code.
- The refactor is behavior-preserving on the capture-and-decode path; its pure
  decoding logic is covered by the off-device tests (see Tests). This revision has
  not been re-run on the physical rig, to which we no longer have access.

## License

Source-available under an **All Rights Reserved** license — see [LICENSE](LICENSE).
One or more related inventions are the subject of a pending patent application;
no patent license is granted.
