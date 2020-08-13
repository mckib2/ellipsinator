
import logging
import numpy as np

def _fit_ellipse_process_params(x, y):

    # Convert to complex array: (x, y) <=> (x.real, x.imag)
    if y is not None:
        assert x.shape == y.shape, 'x, y must have the same shape!'
        x = x + 1j*y

    # Deal with multiple ellipses
    only_one = False
    if x.ndim == 1:
        x = x[None, :]
        only_one = True
    elif x.ndim != 2:
        raise ValueError('x (and y) must have 1 or 2 dimensions: ([M,] N)')

    # Make sure we have enough points to fit
    if x.shape[-1] < 6:
        logging.warning('6 or more points are required for fitting an ellipse!')

    # TODO: normalize here?

    return x, only_one


if __name__ == '__main__':
    pass
