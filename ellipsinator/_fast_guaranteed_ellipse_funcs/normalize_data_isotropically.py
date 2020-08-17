'''Normalize coordinates so that they lie inside a unit box.'''

import numpy as np


def normalize_data_isotropically(x, y):
    '''Normalzie data isotropically.

    This procedure takes as input a matrix of two-dimensional
    coordinates and normalizes the coordinates so that they
    lie inside a unit box.

    Parameters
    ----------
    dataPts : array_like (N, 2)
        an nPoints x 2 matrix of coordinates

    Returns
    -------
    normalizedPts : array_like (N, 2)
        an nPoints x 2 matrix of coordinates which are
        constrained to lie inside a unit box.
    T : array_lile (3, 3)
        a 3x3 affine transformation matrix
        T that was used to transform the
        (homogenous coordinates) of the data
        points so that they lie inside a
        unit box.

    Notes
    -----
    Author: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
    Date: February 2013

    References
    ----------
    .. [1]_ W. Chojnacki and M. Brookes, "On the Consistency of the
            Normalized Eight-Point Algorithm", J Math Imaging Vis (2007)
            28: 19-27
    '''

    assert x.shape == y.shape, 'x, y must be the same shape!'
    nEllipses, nPoints = x.shape[:]

    # homogenous representation of data points resulting in a 3 x nPoints
    # matrix, where the first row contains all the x-coordinates, the second
    # row contains all the y-coordinates and the last row contains the
    # homogenous coordinate 1.
    points = np.concatenate((
        x[None, ...], y[None, ...], np.ones((1, nEllipses, nPoints))), axis=0)

    meanX = np.mean(points[0, ...], axis=-1)
    meanY = np.mean(points[1, ...], axis=-1)

    # isotropic scaling factor
    s = np.sqrt((1/(2*nPoints))*np.sum(
        (points[0, ...] - meanX[:, None])**2 + (points[1, ...] - meanY[:, None])**2, axis=-1))
    T = np.concatenate((
        np.concatenate((1/s[:, None], np.zeros((nEllipses, 1)), (-meanX/s)[:, None]), axis=1)[:, None, :],
        np.concatenate((np.zeros((nEllipses, 1)), 1/s[:, None], (-meanY/s)[:, None]), axis=1)[:, None, :],
        np.concatenate((np.zeros((nEllipses, 1)), np.zeros((nEllipses, 1)), np.ones((nEllipses, 1))), axis=1)[:, None, :],
    ), axis=1)

    normalizedPts = np.einsum('fij,jfk->fik', T, points)
    return(normalizedPts[:, 0, :], normalizedPts[:, 1, :], T)
