
import numpy as np

from ellipsinator import fit_ellipse_halir
from ellipsinator._fast_guaranteed_ellipse_funcs.normalize_data_isotropically import normalize_data_isotropically


def compute_directellipse_estimates(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    x, y : array_like (N,)
        (N,) matrices where N is the number of data points.

    Returns
    -------
    res
        a length-6 vector [a b c d e f] representing the parameters of the
        equation

            a x^2 + b x y + c y^2 + d x + e y + f = 0

         with the additional result that b^2 - 4 a c < 0.

    Notes
    -----
    This function is a wrapper for the numerically stable direct ellipse
    fit due to [1]_ which is a modification of [2]_.

    It first shifts all the data points to a new
    coordinate system so that the origin of the coordinate system is at the
    center of the data points, and then scales all data points so that they
    lie more or less within a unit box. It is within this transformed
    coordinate system that the ellipse is estimated. The resulting ellipse
    parameters are then transformed back to the original data space.

    Zygmunt L. Szpak (c) 2012
    Last modified 27/3/2014

    References
    ---------
    .. [1] R. Halif and J. Flusser "Numerically stable direct least squares
           fitting of ellipses" Proc. 6th International Conference in Central
           Europe on Computer Graphics and Visualization. WSCG '98 Czech
           Republic,125--132, feb, 1998
    .. [2] A. W. Fitzgibbon, M. Pilu, R. B. Fisher "Direct Least Squares
           Fitting of Ellipses" IEEE Trans. PAMI, Vol. 21, pages 476-480 (1999)
    """

    assert x.shape == y.shape, 'x, y must have the same shape!'

    # scale and translate data points so that they lie inside a unit box
    xn, yn, T = normalize_data_isotropically(x, y)

    # theta = direct_ellipse_fit(normalizedPoints.T)
    # or equivalently:
    theta = fit_ellipse_halir(xn, yn)
    theta /= np.linalg.norm(theta, axis=-1, keepdims=True)

    a = theta[:, 0]
    b = theta[:, 1]
    c = theta[:, 2]
    d = theta[:, 3]
    e = theta[:, 4]
    f = theta[:, 5]
    C = np.concatenate((
        np.concatenate((a[:, None], b[:, None]/2, d[:, None]/2), axis=1)[:, None, :],
        np.concatenate((b[:, None]/2, c[:, None], e[:, None]/2), axis=1)[:, None, :],
        np.concatenate((d[:, None]/2, e[:, None]/2, f[:, None]), axis=1)[:, None, :],
    ), axis=1)

    # denormalize C
    C = np.einsum('fji,fjk,fkl->fil', T, C, T)
    # C = T.T @ C @ T
    aa = C[:, 0, 0]
    bb = C[:, 0, 1]*2
    dd = C[:, 0, 2]*2
    cc = C[:, 1, 1]
    ee = C[:, 1, 2]*2
    ff = C[:, 2, 2]
    theta = np.concatenate((
        aa[:, None],
        bb[:, None],
        cc[:, None],
        dd[:, None],
        ee[:, None],
        ff[:, None]), axis=-1)
    return theta/np.linalg.norm(theta, axis=-1, keepdims=True)
