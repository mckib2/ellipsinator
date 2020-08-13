
import numpy as np

#from .direct_ellipse_fit import direct_ellipse_fit
from ellipsinator import fit_ellipse_halir
from .normalize_data_isotropically import normalize_data_isotropically


def compute_directellipse_estimates(dataPts):
    '''
    Parameters
    ----------
    dataPts
        a Nx2 matrix where N is the number of data points

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
    fit due to

    R. Halif and J. Flusser
    "Numerically stable direct least squares fitting of ellipses"
    Proc. 6th International Conference in Central Europe on Computer
    Graphics and Visualization. WSCG '98 Czech Republic,125--132, feb, 1998

    which is a modificaiton of

    A. W. Fitzgibbon, M. Pilu, R. B. Fisher
    "Direct Least Squares Fitting of Ellipses"
    IEEE Trans. PAMI, Vol. 21, pages 476-480 (1999)

    It first shifts all the data points to a new
    coordinate system so that the origin of the coordinate system is at the
    center of the data points, and then scales all data points so that they
    lie more or less within a unit box. It is within this transformed
    coordinate system that the ellipse is estimated. The resulting ellipse
    parameters are then transformed back to the original data space.

    Zygmunt L. Szpak (c) 2012
    Last modified 27/3/2014
    '''

    nPts = dataPts.shape[0]

    # scale and translate data points so that they lie inside a unit box
    normalizedPoints, T = normalize_data_isotropically(dataPts)
    normalizedPoints = np.concatenate((normalizedPoints, np.ones((nPts, 1))), axis=1)

    #theta = direct_ellipse_fit(normalizedPoints.T)
    theta = fit_ellipse_halir(normalizedPoints[:, 0], normalizedPoints[:, 1]) # equivalent
    theta /= np.linalg.norm(theta)

    a = theta[0]
    b = theta[1]
    c = theta[2]
    d = theta[3]
    e = theta[4]
    f = theta[5]
    C = np.array([
        [a, b/2, d/2],
        [b/2, c, e/2],
        [d/2, e/2, f],
    ])

    # denormalise C
    C = T.T @ C @ T
    aa = C[0, 0]
    bb = C[0, 1]*2
    dd = C[0, 2]*2
    cc = C[1, 1]
    ee = C[1, 2]*2
    ff = C[2, 2]
    theta = np.array([aa, bb, cc, dd, ee, ff])
    theta /= np.linalg.norm(theta)

    return theta
