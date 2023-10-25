
import unittest

import numpy as np

from ellipsinator import fit_ellipse_halir
from ellipsinator.tests.fgee_test_data import data_points


class TestHalir(unittest.TestCase):

    def test_fit_ellipse(self):
        c = fit_ellipse_halir(data_points[:, 0], data_points[:, 1])

        # coefficients from MATLAB script:
        c_matlab = np.array([
            4.0537e-06,
            -1.3488e-07,
            1.8535e-05,
            -1.9658e-03,
            -9.2426e-03,
            9.9996e-01,
        ])
        self.assertTrue(np.allclose(c, c_matlab))


if __name__ == '__main__':
    unittest.main()
