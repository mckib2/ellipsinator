
import unittest

import numpy as np

from ellipsinator import fast_guaranteed_ellipse_estimate
from .fgee_test_data import data_points


class TestHalir(unittest.TestCase):

    def test_fit_ellipse(self):
        c, _niter = fast_guaranteed_ellipse_estimate(
            data_points[:, 0], data_points[:, 1])

        # coeffs from MATLAB script:
        c_matlab = np.array([
            3.96548501597781e-06,
            -5.01792302311930e-07,
            1.88176818319181e-05,
            -1.82736162233016e-03,
            -9.30825073431272e-03,
            9.99955007411677e-01,
        ])
        self.assertTrue(np.allclose(c, c_matlab))


if __name__ == '__main__':
    unittest.main()
