"""Python port of MATLAB implementation of available at [1]_.

References
----------
.. [1] https://github.com/zygmuntszpak/guaranteed-ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure
"""

import numpy as np

from ellipsinator._fit_ellipse_process_params import _fit_ellipse_process_params
from ellipsinator._fast_guaranteed_ellipse_funcs.compute_directellipse_estimate import compute_directellipse_estimates
from ellipsinator._fast_guaranteed_ellipse_funcs.normalize_data_isotropically import normalize_data_isotropically
from ellipsinator._fast_guaranteed_ellipse_funcs.fastGuaranteedEllipseFit import fastGuaranteedEllipseFit


def fast_guaranteed_ellipse_estimate(x: np.ndarray, y: np.ndarray=None, cov: np.ndarray=None, ret_iters: bool=False):
    """Guaranteed Ellipse Fitting with a Confidence Region and an
       Uncertainty Measure for Centre, Axes and Orientation

    This procedure takes as input a matrix of two-dimensional
    coordinates and estimates a best fit ellipse using the
    sampson distance between a data point and the ellipse
    equations as the error measure. The Sampson distance is
    an excellent approximation to the orthogonal distance for
    small noise levels. The Sampson distance is often also
    referred to as the approximate maximum likelihood (AML).
    The user can specify a list of covariance matrices for the
    data points. If the user does not specify a list of
    covariance matrices then isotropic homogeneous Gaussian
    noise is assumed.

    Parameters
    ---------
    x : array_like ([M,] N)
        If y is None, x is an array of complex numbers that plot M
        ellipses in the complex plane (i.e., plot x.real vs x.imag).
        If y is not None, x are the x-axis coordinates assumed to
        be on M ellipses with N points (x, y).
    y : None or array_like ([M,] N), optional
        If y is not None, y are the y-axis coordinates assumed to
        be on M ellipses with N points (x, y).
    cov : None or array_like (N, M, 2, 2), optional
        NxM 2x2 covariance matrices representing the uncertainty of the
        coordinates of each data point. If this parameter is not specified
        then  default isotropic (diagonal) and homogeneous (same noise level
        for each data point) covariance matrices are assumed.
    ret_iters : bool, optional
        Return the number of iterations taken to fit each ellipse.

    Returns
    -------
    res : array_like ([M], 6)
        M length-6 vectors containing estimates of the M ellipse
        parameters theta = [a b c d e f] associated with the ellipse
        equation

            a*x^2+ b * x y + c * y^2 + d * x + e*y + f = 0

        with the additional result that b^2 - 4 a c < 0.
    niters : array_like (M,), optional
        If ret_iters=True, then the number of iterations taken to fit
        each of the M ellipses is returned.

    Notes
    -----
    A vectorized implementation the algorithm described in [2]_.

    Original Author of MATLAB: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
    Date: March 2014

    References
    ----------
    .. [2] Z. L. Szpak, W. Chojnacki, and A. van den Hengel. Guaranteed ellipse
           fitting with a confidence region and an uncertainty measure for
           centre, axes, and orientation. J. Math. Imaging Vision,
           52(2):173-199, 2015.
    .. [3] Z. Szpak, W. Chojnacki and A. van den Hengel, "A comparison of
           ellipse fitting methods and implications for multiple view
           geometry", Digital Image Computing Techniques and Applications,
           Dec 2012, pp 1--8
    """

    x, y, only_one = _fit_ellipse_process_params(x, y)
    nPts = x.shape[1]

    # Check to see if the user passed in their own list of covariance matrices
    if cov is None:
        # Generate a list of diagonal covariance matrices
        class UnformDict:
            """Return val for every requested key."""
            def __init__(self, val):
                self.val = val

            def __getitem__(self, _key):
                return self.val
        cov = UnformDict(np.eye(2))

    # estimate an initial ellipse using the direct ellipse fit method
    initialEllipseParameters = compute_directellipse_estimates(x, y)

    # scale and translate data points so that they lie inside a unit box
    xn, yn, T = normalize_data_isotropically(x, y)

    # transfer initialParameters to normalized coordinate system
    # the formula appears in the paper [3]_
    initialEllipseParameters /= np.linalg.norm(
        initialEllipseParameters, axis=-1, keepdims=True)
    E = np.diag([1, 1/2, 1, 1/2, 1/2, 1])
    # permutation matrix for interchanging 3rd and 4th
    # entries of a length-6 vector
    P34 = np.kron(
        np.diag([0, 1, 0]),
        np.array([[0, 1], [1, 0]])) + np.kron(np.diag([1, 0, 1]), np.eye(2))
    # 9 x 6 duplication matrix
    D3 = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])

    P34D3pinv = P34 @ np.linalg.pinv(D3)
    # kTT = np.kron(T, T)
    kTT = np.einsum('fik,fjl->fijkl', T, T)
    kTT.shape = (T.shape[0], T.shape[1]**2, T.shape[2]**2)
    D3P34E = D3 @ P34 @ E
    # P34D3pinv @ np.linalg.inv(kTT).T @ D3P34E @ initialEllipseParameters
    b = np.einsum(
        'ij,fkj,kl,fl->fi',
        P34D3pinv, np.linalg.inv(kTT), D3P34E, initialEllipseParameters)
    initialEllipseParametersNormalizedSpace = np.linalg.solve(E, b.T).T
    initialEllipseParametersNormalizedSpace /= np.linalg.norm(
        initialEllipseParametersNormalizedSpace, axis=-1, keepdims=True)

    # Because the data points are now in a new normalised coordinate system,
    # the data covariance matrices also need to be transformed into the
    # new normalised coordinate system. The transformation of the covariance
    # matrices into the new coordinate system can be achieved by embedding the
    # covariance matrices in a 3x3 matrix (by padding the 2x2 covariance
    # matrices by zeros) and by  multiply the covariance matrices by the
    # matrix T from the left and T' from the right.

    normalised_Cov = np.empty((nPts, x.shape[0], 2, 2))
    for iPts in range(nPts):
        covX_i = np.zeros((3, 3))
        covX_i[0:2, 0:2] = cov[iPts]
        # covX_i = T @ covX_i @ T.T
        covX_i = np.einsum('fij,jk,flk->fil', T, covX_i, T)
        # the upper-left 2x2 matrix now represents the covariance of the
        # coordinates of the data point in the normalised coordinate system
        normalised_Cov[iPts] = covX_i[:, 0:2, 0:2]

    # To guarantee an ellipse we utilise a special parameterization which
    # by definition excludes the possibility of a hyperbola. In theory
    # a parabola could be estimated, but this is very unlikely because
    # an equality constraint is difficult to satisfy when there is noisy data.
    # As an extra guard to ensure that a parabolic fit is not possible we
    # terminate our algorithm when the discriminant of the conic equation
    # approaches zero.

    # convert our original parameterization to one that excludes hyperbolas
    # NB, it is assumed that the initialParameters that were passed into the
    # function do not represent a hyperbola or parabola.
    para = initialEllipseParametersNormalizedSpace
    p = para[:, 1]/(2*para[:, 0])
    q = np.sqrt(para[:, 2]/para[:, 0] - (para[:, 1]/(2*para[:, 0]))**2)
    r = para[:, 3] / para[:, 0]
    s = para[:, 4] / para[:, 0]
    t = para[:, 5] / para[:, 0]

    latentParameters = np.concatenate((
        p[:, None],
        q[:, None],
        r[:, None],
        s[:, None],
        t[:, None],
    ), axis=-1)

    ellipseParametersFinal, iterations = fastGuaranteedEllipseFit(
        latentParameters, xn, yn, normalised_Cov)
    ellipseParametersFinal /= np.linalg.norm(
        ellipseParametersFinal, axis=-1, keepdims=True)

    # convert final ellipse parameters back to the original coordinate system
    # P34D3pinv @ kTT.T @ D3P34E @ ellipseParametersFinal
    b = np.einsum(
        'ij,fkj,kl,fl->fi',
        P34D3pinv, kTT, D3P34E, ellipseParametersFinal)
    estimatedParameters = np.linalg.solve(E, b.T).T
    estimatedParameters /= np.linalg.norm(
        estimatedParameters, axis=-1, keepdims=True)
    estimatedParameters *= np.sign(estimatedParameters[:, -1][:, None])

    if only_one:
        estimatedParameters = estimatedParameters.squeeze()
    if ret_iters:
        return estimatedParameters, iterations
    return estimatedParameters


if __name__ == '__main__':
    pass
