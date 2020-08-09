
import logging

import numpy as np

def fit_ellipse_halir(x, y=None):
    '''Improved ellipse fitting algorithm by Halir and Flusser.

    Parameters
    ----------
    x : array_like ([M,] N)
        If y is None, x is an array of complex numbers that plot M
        ellipses in the complex plane (i.e., plot x.real vs x.imag).
        If y is not None, x are the x-axis coordinates assumed to
        be on M ellipses with N points (x, y).
    y : None or array_like ([M,] N), optional
        If y is not None, y are the y-axis coordinates assumed to
        be on M ellipses with N points (x, y).

    Returns
    -------
    res : array_like ([M,] 6)
        Ellipse coefficients of the M ellipses.

    Notes
    -----
    Note that there should be at least 6 pairs of (x,y).

    If an ellipse fit fails, the coefficients for that ellipse
    will be all 0s.

    From the paper's conclusion:

        "Due to its systematic bias, the proposed fitting algorithm
        cannot be used directly in applications where excellent
        accuracy of the fitting is required. But even in that
        applications our method can be useful as a fast and robust
        estimator of a good initial solution of the fitting
        problem..."

    See figure 2 from [1]_.

    Paper and reference MATLAB implementation can be accessed
    at [2]_.

    References
    ----------
    .. [1] A. W. Fitzgibbon, M. Pilu, R. B. Fisher "Direct Least
           Squares Fitting of Ellipses" IEEE Trans. PAMI, Vol. 21,
           pages 476-480 (1999)
    .. [2] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7559&rep=rep1&type=pdf
    '''

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

    # quadratic pt of design matrix
    D1 = np.stack((
        x.real**2,
        x.real*x.imag,
        x.imag**2), axis=1).transpose((0, 2, 1))
    # lin part design matrix
    D2 = np.stack((
        x.real,
        x.imag,
        np.ones(x.shape)), axis=1).transpose((0, 2, 1))

    # quadratic part of the scatter matrix
    S1 = np.einsum('fji,fjk->fik', D1, D1)
    # combined part of the scatter matrix
    S2 = np.einsum('fji,fjk->fik', D1, D2)
    # linear part of the scatter matrix
    S3 = np.einsum('fji,fjk->fik', D2, D2)

    # for getting a2 from a1
    # TODO(mckib2): Use np.linalg.lstsq? np.linalg.solve doesn't
    #               handle close-to-singular matrices very well
    #               MATLAB version uses inv, but we might be able
    #               to do better
    T = np.einsum('fij,fkj->fik', -1*np.linalg.pinv(S3), S2)

    # reduced scatter matrix; premult by C1^-1
    M = S1 + np.einsum('fij,fjk->fik', S2, T)
    M = np.stack((
        M[:, 2, :]/2,
        -1*M[:, 1, :],
        M[:, 0, :]/2), axis=1)

    # solve eigensystem
    _eval, evec = np.linalg.eig(M)

    # TODO(mckib2): vectorize this?
    a1 = np.empty((M.shape[0], 3))
    for ii in range(M.shape[0]):
        # evaluate a’Ca
        cond = 4*evec[ii, 0, :]*evec[ii, 2, :] - evec[ii, 1, :]**2
        # eigenvector for min. pos. eigenvalue
        if not np.sum(cond > 0):
            # Failed to fit the ellipse! send back 0s, let user
            # handle this failure case
            a1[ii, :] = 0
        else:
            a1[ii, :] = evec[ii, :, cond > 0]

    # ellipse coefficients
    a = np.concatenate((a1, np.einsum('fij,fj->fi', T, a1)), axis=-1)
    a /= np.linalg.norm(a) # normalize? not in MATLAB version

    if only_one:
        return a[0, :]
    return a

if __name__ == '__main__':
    pass
