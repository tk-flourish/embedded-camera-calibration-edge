"""Pure structured-light decoding math.

Hardware- and I/O-free helpers for turning captured frames into projector
correspondences: light-region separation, Gray-code/ID brightness comparison,
per-region luma, and the Gaussian fit used for subpixel refinement. Nothing here
touches the camera or the network, so it imports and tests without the
Raspberry Pi packages.
"""

import cv2
import numpy as np
from scipy.optimize import curve_fit

# Connected components smaller than this many pixels are discarded.
AREA_MIN_PIXELS = 30

# In line-shift decoding, a projector whose probed intensities never exceed this
# value is treated as not observed (subpixel coordinate set to NaN).
MIN_LINE_INTENSITY = 20

# BT.601 luma weights, applied to OpenCV BGR pixels (item[0]=B, item[1]=G, item[2]=R).
LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114

# Type aliases for the structured-light data structures.
Pixel = tuple[int, int]        # (row, col) pixel coordinate
Region = list[Pixel]           # the pixels of one connected component
Regions = dict[int, Region]    # region key -> region
PixelValues = list[list[int]]  # per-pixel channel values sampled in one frame


def find_projector_areas(white: np.ndarray, black: np.ndarray, area_threshold: int) -> Regions:
    """Separate the projector light regions from a white/black image pair.

    Subtracts the all-black frame from the all-white frame, thresholds at 1/16
    of the peak summed intensity, and groups the surviving pixels into
    8-connected components. Components smaller than ``area_threshold`` pixels are
    dropped. Returns a mapping from an opaque region key to the list of
    (row, col) pixels in that region.
    """
    # Signed difference white - black (cast to int16 so negative values are kept)
    diff = white.astype(np.int16) - black.astype(np.int16)

    # Per-pixel channel sum (axis=2 sums over RGB)
    summed = np.sum(diff, axis=2)

    # Threshold at 1/16 of the maximum summed value. Pixels exactly at 0 are
    # excluded to match the previous nonzero-based selection.
    threshold = np.max(summed) * 1 / 16
    mask = ((summed >= threshold) & (summed != 0)).astype(np.uint8)

    # 8-connected components; label 0 is the background.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions: Regions = {}
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < area_threshold:
            continue
        rows, cols = np.nonzero(labels == label)
        regions[label] = [(int(r), int(c)) for r, c in zip(rows, cols)]
    return regions


def get_values_of_area(image: np.ndarray, area: Region) -> PixelValues:
    """Return the pixel values at each (row, col) position in ``area``."""
    return [image[position].tolist() for position in area]


def mean_luma(values: PixelValues) -> float:
    """Mean BT.601 luma over a region's BGR pixel values."""
    gray_scaled = [LUMA_R * item[2] + LUMA_G * item[1] + LUMA_B * item[0] for item in values]
    return sum(gray_scaled) / len(gray_scaled)


def decode_gray_code(code: int) -> int:
    """Convert a Gray-code integer to its plain binary value."""
    a = code
    b = code
    while a != 0:
        a >>= 1
        b ^= a
    return b


def compare_lists(pixels1: PixelValues, pixels2: PixelValues) -> int:
    """Compare two equally shaped lists of pixel values by brightness.

    Returns 1 if ``pixels1`` is brighter overall, -1 if ``pixels2`` is brighter,
    or 0 on a tie. For each pixel, the channels in which one side exceeds the
    other are counted to decide which side "wins" that pixel; the winning sides
    are then tallied across all pixels.
    """
    a = np.asarray(pixels1)
    b = np.asarray(pixels2)
    if a.size == 0:
        return 0

    # Per-pixel count of channels each side wins, then which side won the pixel.
    brighter1 = np.count_nonzero(a > b, axis=1)
    brighter2 = np.count_nonzero(b > a, axis=1)
    wins1 = int(np.count_nonzero(brighter1 > brighter2))
    wins2 = int(np.count_nonzero(brighter2 > brighter1))

    if wins1 > wins2:
        return 1
    elif wins2 > wins1:
        return -1
    else:
        return 0


def gaussian(x: np.ndarray, A: float, mu: float, sigma: float, C: float) -> np.ndarray:
    """Gaussian with amplitude A, mean mu, standard deviation sigma, offset C."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C


def fit_distribution(profile: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian to a {position: intensity} profile.

    Trims leading/trailing zero entries, then fits ``gaussian`` and returns
    (optimal parameters, parameter standard deviations). The fitted mean
    (parameter index 1) is the subpixel peak position.
    """
    xs = np.array(list(profile.keys()))
    ys = np.array(list(profile.values()))

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
    return popt, np.sqrt(np.diag(pcov))
