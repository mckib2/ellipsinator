'''Process common input arguments to ellipse fitting methods.'''

import logging


def _fit_ellipse_process_params(x, y):

    # Convert complex array: (x, y) <=> (x.real, x.imag)
    if y is None:
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
