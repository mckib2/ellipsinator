"""Tests for fast guaranteed ellipse fitting method."""

import unittest

import numpy as np

from ellipsinator import fast_guaranteed_ellipse_estimate
from ellipsinator.tests.fgee_test_data import data_points


class TestGuaranteedFit(unittest.TestCase):
    """TestGuaranteedFit."""

    def test_fit_ellipse(self):
        """Use test ellipse given in original MATLAB script."""

        c = fast_guaranteed_ellipse_estimate(
            data_points[:, 0], data_points[:, 1])

        # coefficients from MATLAB script:
        c_matlab = np.array([
            3.96548501597781e-06,
            -5.01792302311930e-07,
            1.88176818319181e-05,
            -1.82736162233016e-03,
            -9.30825073431272e-03,
            9.99955007411677e-01,
        ])
        self.assertTrue(np.allclose(c, c_matlab))

    def test_multiple_same_ellipse(self):
        """Fit the same ellipse multiple times."""

        num_copies = 7
        x, y = data_points[:, 0], data_points[:, 1]
        x = np.tile(x, (num_copies, 1))
        y = np.tile(y, (num_copies, 1))
        c = fast_guaranteed_ellipse_estimate(x, y)

        # coefficients from MATLAB script:
        c_matlab = np.array([
            3.96548501597781e-06,
            -5.01792302311930e-07,
            1.88176818319181e-05,
            -1.82736162233016e-03,
            -9.30825073431272e-03,
            9.99955007411677e-01,
        ])
        c_matlab = np.tile(c_matlab[None, :], (num_copies, 1))
        self.assertTrue(np.allclose(c, c_matlab))

    def test_multiple_ellipses_different_iters(self):
        """Test similar ellipses which take different iters."""

        num_copies = 7
        x, y = data_points[:, 0], data_points[:, 1]
        x = np.tile(x, (num_copies, 1))
        y = np.tile(y, (num_copies, 1))
        np.random.seed(7)
        x += np.random.randn(*x.shape)
        y += np.random.randn(*y.shape)
        _c, niters = fast_guaranteed_ellipse_estimate(x, y, ret_iters=True)
        self.assertEqual(niters.tolist(), [5, 4, 4, 4, 4, 4, 4])


if __name__ == '__main__':
    unittest.main()
