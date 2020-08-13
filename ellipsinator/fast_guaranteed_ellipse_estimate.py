'''Python port of MATLAB implementation of available at [1]_.

References
----------
.. [1] https://github.com/zygmuntszpak/guaranteed-ellipse-fitting-with-a-confidence-region-and-an-uncertainty-measure
'''

import numpy as np

from ._fit_ellipse_process_params import _fit_ellipse_process_params
from ._fast_guaranteed_ellipse_funcs.compute_directellipse_estimate import compute_directellipse_estimates
from ._fast_guaranteed_ellipse_funcs.normalize_data_isotropically import normalize_data_isotropically
from ._fast_guaranteed_ellipse_funcs.fastGuaranteedEllipseFit import fastGuaranteedEllipseFit

def fast_guaranteed_ellipse_estimate(x, y=None, covList=None):
    '''Guaranteed Ellipse Fitting with a Confidence Region and an
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
    x : array_like (N,)
        If y is None, x is an array of complex numbers that plots an
        ellipse in the complex plane (i.e., plot x.real vs x.imag).
        If y is not None, x are the x-axis coordinates assumed to
        be on the ellipse with N points (x, y).
    y : None or array_like (N,), optional
        If y is not None, y are the y-axis coordinates assumed to
        be on the ellipse with N points (x, y).
    covList : list of array_like, optional
        a list of N 2x2 covariance matrices
        representing the uncertainty of the
        coordinates of each data point.
        if this parameter is not specified
        then  default isotropic  (diagonal)
        and homogeneous (same noise level
        for each data point) covariance
        matrices are assumed.

    Returns
    -------
    res : array_like (6,)
        a length-6 vector containing an estimate of the ellipse
        parameters theta = [a b c d e f] associated with the ellipse
        equation

            a*x^2+ b * x y + c * y^2 + d * x + e*y + f = 0

        with the additional result that b^2 - 4 a c < 0.

    Notes
    -----
    Implements the algorithm described in [2]_.

    Much of the Python port was a mechanical process, but some obvious
    optimizations were taken opportunistically.  Also note that this
    function differs from some of the other fitting methods in this
    package in that it only fits a single ellipse at a time.

    Original Author of MATLAB: Zygmunt L. Szpak (zygmunt.szpak@gmail.com)
    Date: March 2014

    References
    ----------
    .. [2] Z. L. Szpak, W. Chojnacki, and A. van den Hengel. Guaranteed ellipse
           fitting with a confidence region and an uncertainty measure for
           centre, axes, and orientation. J. Math. Imaging Vision, 52(2):173-199,
           2015.
    .. [3] Z.Szpak, W. Chojnacki and A. van den Hengel, "A comparison of ellipse
           fitting methods and implications for multiple view geometry", Digital
           Image Computing Techniques and Applications, Dec 2012, pp 1--8
    '''

    x, only_one = _fit_ellipse_process_params(x, y)
    assert only_one, 'This function only fits a single ellipse!'

    # We expect dataPts to have shape (N, 2):
    dataPts = np.concatenate((x.real, x.imag), axis=0).T
    nPts = dataPts.shape[0]

    # Check to see if the user passed in their own list of covariance matrices
    if covList is None:
        # Generate a list of diagonal covariance matrices
        #covList = mat2cell(repmat(np.eye(2), 1, nPts), 2, 2*(np.ones((1, nPts))))
        class UnformDict:
            def __init__(self, val):
                self.val = val
            def __getitem__(self, _key):
                return self.val
        covList = UnformDict(np.eye(2))

    # estimate an initial ellipse using the direct ellipse fit method
    initialEllipseParameters = compute_directellipse_estimates(dataPts)

    # scale and translate data points so that they lie inside a unit box
    normalizedPoints, T = normalize_data_isotropically(dataPts)

    # transfer initialParameters to normalized coordinate system
    # the formula appears in the paper [3]_
    initialEllipseParameters /= np.linalg.norm(initialEllipseParameters)
    E = np.diag([1, 1/2, 1, 1/2, 1/2, 1])
    # permutation matrix for interchanging 3rd and 4th
    # entries of a length-6 vector
    P34 = np.kron(np.diag([0, 1, 0]), np.array([[0, 1], [1, 0]])) + np.kron(np.diag([1, 0, 1]), np.eye(2))
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
    kTT = np.kron(T, T)
    D3P34E = D3 @ P34 @ E
    initialEllipseParametersNormalizedSpace = np.linalg.solve(E, P34D3pinv @ np.linalg.inv(kTT).T @ D3P34E @ initialEllipseParameters)
    initialEllipseParametersNormalizedSpace /= np.linalg.norm(initialEllipseParametersNormalizedSpace)

    # Becase the data points are now in a new normalised coordinate system,
    # the data covariance matrices also need to be tranformed into the
    # new normalised coordinate system. The transformation of the covariance
    # matrices into the new coordinate system can be achieved by embedding the
    # covariance matrices in a 3x3 matrix (by padding the 2x2 covariance
    # matrices by zeros) and by  multiply the covariance matrices by the
    # matrix T from the left and T' from the right.
    normalised_CovList = {} #cell(1, nPts)
    for iPts in range(nPts):
        covX_i = np.zeros((3, 3))
        covX_i[0:2, 0:2] = covList[iPts]
        covX_i = T @ covX_i @ T.T
        # the upper-left 2x2 matrix now represents the covariance of the
        # coordinates of the data point in the normalised coordinate system
        normalised_CovList[iPts] = covX_i[0:2, 0:2]

    # To guarantee an ellipse we utilise a special parameterisation which
    # by definition excludes the possiblity of a hyperbola. In theory
    # a parabola could be estimated, but this is very unlikely because
    # an equality constraint is difficult to satisfy when there is noisy data.
    # As an extra guard to ensure that a parabolic fit is not possible we
    # terminate our algorithm when the discriminant of the conic equation
    # approaches zero.

    # convert our original parameterisation to one that excludes hyperbolas
    # NB, it is assumed that the initialParameters that were passed into the
    # function do not represent a hyperbola or parabola.
    para = initialEllipseParametersNormalizedSpace
    # p = para(2)/(2*para(1))
    # q = 1 / sqrt(para(3)/para(1) - (para(2)/(2*para(1)))^2)
    # r = para(4) / para(1)
    # s = para(5) / para(1)
    # t = para(6) / para(1)
    p = para[1]/(2*para[0])
    q = np.sqrt(para[2]/para[0] - (para[1]/(2*para[0]))**2)
    r = para[3] / para[0]
    s = para[4] / para[0]
    t = para[5] / para[0]

    latentParameters = np.array([p, q, r, s, t])

    ellipseParametersFinal, iterations = fastGuaranteedEllipseFit(latentParameters, normalizedPoints.T, normalised_CovList)

    ellipseParametersFinal /= np.linalg.norm(ellipseParametersFinal)

    # convert final ellipse parameters back to the original coordinate system
    estimatedParameters = np.linalg.solve(E, P34D3pinv @ kTT.T @ D3P34E @ ellipseParametersFinal)
    estimatedParameters /= np.linalg.norm(estimatedParameters)
    estimatedParameters *= np.sign(estimatedParameters[-1])

    return(estimatedParameters, iterations)

if __name__ == '__main__':
    pass
