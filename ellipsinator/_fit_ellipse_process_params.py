"""Process common input arguments to ellipse fitting methods."""

from typing import Optional
import logging

import numpy as np


def _fit_ellipse_process_params(x: np.ndarray, y: Optional[np.ndarray]):

    # Convert complex array: (x, y) <=> (x.real, x.imag)
    if y is None:
        assert np.iscomplex(x), "if y not provided, x must be a complex-valued array"
        x, y = x.real, x.imag
    assert x.shape == y.shape, 'x, y must have the same shape!'

    # Deal with multiple ellipses
    only_one = False
    if x.ndim == 1:
        x = x[None, :]
        y = y[None, :]
        only_one = True
    elif x.ndim != 2:
        raise ValueError('x (and y) must have 1 or 2 dimensions: ([M,] N)')

    # Make sure we have enough points to fit
    if x.shape[-1] < 6:
        logging.warning('6 or more points are required '
                        'for fitting an ellipse!')

    # TODO: normalize here?

    return x, y, only_one


if __name__ == '__main__':
    pass
