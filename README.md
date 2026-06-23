# embedded-camera-calibration-edge

> **Note — original implementation.** This commit is the code **as used to
> produce the paper's results**. Later commits refactor it into modules and add
> tests *without changing the capture-and-decode behavior*; check out the latest
> revision for the cleaned-up version and fuller documentation.

The edge (on-device) capture-and-decode agent for multi-projector geometric
calibration using cameras embedded in the calibration target. It runs on a
Raspberry Pi with the embedded camera; a calibration host connects over TCP to
drive the camera through a timed structured-light sequence and retrieve the
decoded projector correspondences.

It is the embedded-device component of the system behind
[*"Breaking the Scalability Limit of Multi-Projector Calibration with Embedded
Cameras"*](https://cvpr.thecvf.com/virtual/2026/oral/40265) (CVPR 2026, oral).
The calibration core (parameter estimation, optical-center homography) runs on a
separate machine.

> **License:** source-available, **All Rights Reserved** (patent pending).
> See [LICENSE](LICENSE).

## What it does

The embedded cameras directly receive the projection light, so light from
projectors at different positions lands on different camera pixels and
simultaneously projected patterns can be separated by incident direction. This
agent performs the camera-side capture and decoding:

- **Light-ray separation**: white−black difference, threshold at 1/16 of the
  max, 8-neighbour Union-Find connected components, and an area filter to find
  each projector's illuminated region.
- **Projector-ID decoding**: complementary binary patterns are compared to read a
  per-region ID, so all projectors can be projected at once.
- **Gray-code decoding**: horizontal/vertical Gray-code patterns are decoded to
  the integer projector pixel coordinate.
- **Subpixel refinement**: line-shift intensity profiles are fit with a Gaussian
  to recover subpixel coordinates.

## Files

- `main.py` — capture-and-decode server (camera wrapper, the decoding steps
  above, and the TCP command loop).
- `connection.py` — minimal length-prefixed TCP protocol between host and agent.
- `unionfind.py` — Union-Find used for connected-component region grouping.

## Running

```
python main.py
```

Starts a TCP server (port 58919) that accepts connections from the calibration
host. Intended to run on the Raspberry Pi carrying the embedded camera. Requires
`picamera2` / `libcamera` (Raspberry Pi OS system packages), `opencv-python`,
`numpy`, `scipy`, and `Pillow`, installed separately.
