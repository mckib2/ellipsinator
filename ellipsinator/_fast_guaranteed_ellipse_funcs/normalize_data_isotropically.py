
import numpy as np

def normalize_data_isotropically(dataPts):
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

    nPoints = dataPts.shape[0]

    # homogenous representation of data points resulting in a 3 x nPoints
    # matrix, where the first row contains all the x-coordinates, the second
    # row contains all the y-coordinates and the last row contains the
    # homogenous coordinate 1.
    points = np.concatenate((dataPts, np.ones((nPoints, 1))), axis=1).T

    meanX = np.mean(points[0, :])
    meanY = np.mean(points[1, :])

    # isotropic scaling factor
    s = np.sqrt((1/(2*nPoints))*np.sum((points[0, :] - meanX)**2 + (points[1, :] - meanY)**2))
    T = np.array([
        [1/s,    0,     -meanX/s],
        [0,      1/s,   -meanY/s],
        [0,      0,     1],
    ])

    normalizedPts = T @ points
    # remove homogenous coordinate
    normalizedPts = normalizedPts.T[:, :-1]
    return(normalizedPts, T)
