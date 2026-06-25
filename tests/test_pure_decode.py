"""Characterization tests for the stateless decoding helpers in ``decoding``.

These pin the *current* behaviour so the planned refactors (cv2-based
connected components, vectorised ``compare_lists``) can be checked for
equivalence rather than guessed at.
"""

import numpy as np
import pytest

import connection
import decoding


class TestDecodeGrayCode:
    @pytest.mark.parametrize(
        "code, expected",
        [(0, 0), (1, 1), (2, 3), (3, 2), (4, 7), (5, 6), (6, 4), (7, 5)],
    )
    def test_known_values(self, code, expected):
        assert decoding.decode_gray_code(code) == expected

    def test_is_inverse_of_binary_to_gray(self):
        # Gray(n) = n ^ (n >> 1); decode_gray_code must invert it.
        for n in range(256):
            gray = n ^ (n >> 1)
            assert decoding.decode_gray_code(gray) == n


class TestCompareLists:
    def test_first_brighter_returns_1(self):
        assert decoding.compare_lists([[10, 10, 10]], [[0, 0, 0]]) == 1

    def test_second_brighter_returns_minus_1(self):
        assert decoding.compare_lists([[0, 0, 0]], [[10, 10, 10]]) == -1

    def test_equal_returns_0(self):
        assert decoding.compare_lists([[5, 5, 5]], [[5, 5, 5]]) == 0

    def test_empty_returns_0(self):
        assert decoding.compare_lists([], []) == 0

    def test_per_pixel_channel_majority(self):
        # pixel1 wins only channel 0; pixel2 wins channels 1 and 2 -> pixel2 wins.
        assert decoding.compare_lists([[10, 0, 0]], [[0, 10, 10]]) == -1

    def test_pixel_level_tie_breaks_to_zero(self):
        # one pixel won by each side -> overall tie.
        p1 = [[10, 10, 10], [0, 0, 0]]
        p2 = [[0, 0, 0], [10, 10, 10]]
        assert decoding.compare_lists(p1, p2) == 0

    def test_channel_tie_within_pixel_is_no_win(self):
        # ch0 ties, ch1 -> p1, ch2 -> p2: pixel is a draw, overall 0.
        assert decoding.compare_lists([[5, 9, 1]], [[5, 1, 9]]) == 0

    def test_matches_scalar_reference_on_random_input(self):
        # Pin equivalence with the original nested-loop implementation.
        def reference(pixels1, pixels2):
            wins1 = wins2 = 0
            for p1, p2 in zip(pixels1, pixels2):
                b1 = sum(1 for c1, c2 in zip(p1, p2) if c1 > c2)
                b2 = sum(1 for c1, c2 in zip(p1, p2) if c2 > c1)
                if b1 > b2:
                    wins1 += 1
                elif b2 > b1:
                    wins2 += 1
            return 1 if wins1 > wins2 else -1 if wins2 > wins1 else 0

        rng = np.random.default_rng(20260625)
        for _ in range(500):
            n = int(rng.integers(1, 12))
            a = rng.integers(0, 6, size=(n, 3)).tolist()
            b = rng.integers(0, 6, size=(n, 3)).tolist()
            assert decoding.compare_lists(a, b) == reference(a, b)


class TestGetValuesOfArea:
    def test_returns_pixel_values_in_order(self):
        img = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
        area = [(0, 0), (1, 1)]
        assert decoding.get_values_of_area(img, area) == [[0, 1, 2], [9, 10, 11]]


class TestByteReader:
    def test_sequential_big_endian_u32(self):
        data = (1).to_bytes(4, "big") + (258).to_bytes(4, "big") + (0xFFFFFFFF).to_bytes(4, "big")
        reader = connection.ByteReader(data)
        assert reader.read_u32() == 1
        assert reader.read_u32() == 258
        assert reader.read_u32() == 0xFFFFFFFF


class TestGaussianFit:
    def test_recovers_known_mean(self):
        true_mu, true_sigma, true_A, true_C = 12.3, 2.0, 100.0, 5.0
        profile = {
            x: float(decoding.gaussian(np.array(x), true_A, true_mu, true_sigma, true_C))
            for x in range(5, 20)
        }
        popt, perr = decoding.fit_distribution(profile)
        assert popt.shape == (4,) and perr.shape == (4,)
        assert popt[1] == pytest.approx(true_mu, abs=1e-3)

    def test_trims_leading_and_trailing_zeros(self):
        # Zeros outside the peak must not pull the fitted mean.
        profile = {x: 0.0 for x in range(0, 30)}
        for x in range(10, 21):
            profile[x] = float(decoding.gaussian(np.array(x), 80.0, 15.0, 1.5, 0.0))
        popt, _ = decoding.fit_distribution(profile)
        assert popt[1] == pytest.approx(15.0, abs=1e-2)
