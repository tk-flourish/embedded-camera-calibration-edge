"""Characterization tests for ``find_projector_areas``.

This is the prime refactor target (Python union-find -> cv2.connectedComponents).
The region *keys* are opaque union-find roots, so tests compare the set of
regions as frozensets of (row, col) pixels, independent of key assignment.
"""

import numpy as np

import decoding


def regions_as_sets(regions):
    """Normalise the {key: [pixels]} result to a set of frozenset(pixels)."""
    return {frozenset(map(tuple, pixels)) for pixels in regions.values()}


def make_pair(shape=(20, 20)):
    """Return (white, black) BGR uint8 frames; black is all zero."""
    white = np.zeros((*shape, 3), dtype=np.uint8)
    black = np.zeros((*shape, 3), dtype=np.uint8)
    return white, black


def fill(frame, rows, cols, value=90):
    frame[rows[0]:rows[1], cols[0]:cols[1], :] = value


def test_single_block_is_one_region():
    white, black = make_pair()
    fill(white, (5, 11), (5, 11))  # 6x6 = 36 px
    regions = decoding.find_projector_areas(white, black, area_threshold=30)
    expected = {(r, c) for r in range(5, 11) for c in range(5, 11)}
    assert regions_as_sets(regions) == {frozenset(expected)}


def test_two_separated_blocks_are_two_regions():
    white, black = make_pair((30, 30))
    fill(white, (2, 8), (2, 8))     # 36 px
    fill(white, (20, 26), (20, 26))  # 36 px
    regions = decoding.find_projector_areas(white, black, area_threshold=30)
    a = frozenset((r, c) for r in range(2, 8) for c in range(2, 8))
    b = frozenset((r, c) for r in range(20, 26) for c in range(20, 26))
    assert regions_as_sets(regions) == {a, b}


def test_block_below_area_threshold_is_dropped():
    white, black = make_pair()
    fill(white, (5, 11), (5, 11), value=90)  # 36 px, kept
    fill(white, (0, 2), (0, 2), value=90)    # 4 px, dropped at threshold 30
    regions = decoding.find_projector_areas(white, black, area_threshold=30)
    kept = frozenset((r, c) for r in range(5, 11) for c in range(5, 11))
    assert regions_as_sets(regions) == {kept}


def test_diagonal_pixels_join_via_8_connectivity():
    # A staircase of single pixels is only connected under 8-connectivity.
    white, black = make_pair()
    coords = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    for r, c in coords:
        white[r, c, :] = 90
    regions = decoding.find_projector_areas(white, black, area_threshold=5)
    assert regions_as_sets(regions) == {frozenset(coords)}


def test_degenerate_frames_yield_no_regions():
    # When no pixel is brighter in white than black, max(summed) <= 0 and the
    # threshold (max/16) sits closer to zero than max, so nothing is selected.
    # Guards against a full-frame "everything is lit" region on dark input.
    shape = (20, 20)
    all_zero = (np.zeros((*shape, 3), np.uint8), np.zeros((*shape, 3), np.uint8))
    black_brighter = (np.full((*shape, 3), 10, np.uint8), np.full((*shape, 3), 200, np.uint8))
    for white, black in (all_zero, black_brighter):
        assert decoding.find_projector_areas(white, black, area_threshold=1) == {}


def test_dim_block_below_intensity_threshold_excluded():
    # Bright block sets max; a block under 1/16 of the peak sum is excluded.
    white, black = make_pair((30, 30))
    fill(white, (2, 8), (2, 8), value=255)  # summed peak 765
    # 765/16 ~= 47.8 -> per-channel value 15 (summed 45) is below threshold.
    fill(white, (20, 26), (20, 26), value=15)
    regions = decoding.find_projector_areas(white, black, area_threshold=30)
    bright = frozenset((r, c) for r in range(2, 8) for c in range(2, 8))
    assert regions_as_sets(regions) == {bright}
