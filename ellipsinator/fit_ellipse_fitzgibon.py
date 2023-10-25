"""Ellipse fitting algorithm due to Fitzgibon et al."""

import numpy as np

from ellipsinator._fit_ellipse_process_params import _fit_ellipse_process_params


def fit_ellipse_fitzgibon(x: np.ndarray, y: np.ndarray=None):
    """Python port of direct ellipse fitting algorithm by Fitzgibon et al.

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
    See Figure 1 from [1]_.
    Also see previous python port: ttp://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    References
    ----------
    .. [1] Halır, Radim, and Jan Flusser. "Numerically stable direct least
           squares fitting of ellipses." Proc. 6th International Conference in
           Central Europe on Computer Graphics and Visualization. WSCG. Vol.
           98. 1998.
    """

    # Process the parameters
    x, y, only_one = _fit_ellipse_process_params(x, y)
    assert only_one, 'This method only works with a single ellipse!'
    x, y = x.squeeze(), y.squeeze()  # TODO: make M-ellipsable

    # Do the thing
    x = x[:, None]
    y = y[:, None]
    D = np.hstack((
        x*x, x*y, y*y, x, y, np.ones_like(x.real)))  # Design matrix
    S = D.T @ D  # Scatter matrix
    C = np.zeros([6, 6])  # Constraint matrix
    C[(0, 2), (0, 2)] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))  # solve eigensystem
    n = np.argmax(np.abs(E))  # find positive eigenvalue
    a = V[:, n].squeeze()  # corresponding eigenvector
    return a/np.linalg.norm(a)
