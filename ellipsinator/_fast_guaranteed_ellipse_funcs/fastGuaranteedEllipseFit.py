'''Main fitting loop.'''

import numpy as np

from .fastLevenbergMarquardtStep import fastLevenbergMarquardtStep


def fastGuaranteedEllipseFit(latentParameters, x, y, covList, maxiter=200):
    '''

    This function implements the ellipse fitting algorithm described
    in [1]_.

    Parameters
    ----------
    latentParameters : array_like (5,)
        an initial seed for latent parameters
        [p q r s t] which through a transformation
        are related to parameters  [a b c d e f]
        associated with the conic equation

                             a x^2 + b x y + c y^2 + d x + e y + f = 0

    dataPts : array_like (2, N)
        a 2xN matrix where N is the number of data
        points
    covList : list of array_like
        a list of N 2x2 covariance matrices
        representing the uncertainty of the
        coordinates of each data point.

    Returns
    -------
    res : array_like (6,)
        a length-6 vector [a b c d e f] representing the parameters of the
        equation

            a x^2 + b x y + c y^2 + d x + e y + f = 0

     with the additional result that b^2 - 4 a c < 0.

    Notes
    -----
    Zygmunt L. Szpak (c) 2014
    Last modified 18/3/2014

    References
    ----------
    .. [1] Z.Szpak, W. Chojnacki and A. van den Hengel
          "Guaranteed Ellipse Fitting with an Uncertainty
          Measure for Centre, Axes, and Orientation"
    '''

    nEllipses = x.shape[0]

    eta = latentParameters  # (nEllipses, 5)

    # convert latent variables into length-6 vector (called t) representing
    # the equation of an ellipse
    t = np.concatenate((  # (nEllipses, 6)
        np.ones((nEllipses, 1)),
        2*eta[:, 0][:, None],
        (eta[:, 0]**2 + np.abs(eta[:, 1])**2)[:, None],
        eta[:, 2][:, None],
        eta[:, 3][:, None],
        eta[:, 4][:, None],
    ), axis=1)
    t /= np.linalg.norm(t, axis=-1)

    # various variable initialisations
    ##################################
    # primary loop variable
    keep_going = True
    # in some case a LevenbergMarquardtStep does not decrease the cost
    # function and so the parameters (eta) are not updated

    class struct_t:
        '''simply hold parameters.'''
        def __init__(self):
            self.nEllipses = nEllipses
            self.eta_updated = False
            self.lamda = 0.01  # damping parameter in LevenbergMarquadtStep
            self.k = 0  # loop counter
            # used to modify the tradeoff between gradient descent and hessian based
            # descent in LevenbergMarquadtStep
            self.damping_multiplier = 15
            # used to modify the tradeoff between gradient descent and hessian based
            # descent in LevenbergMarquadtStep
            self.damping_divisor = 1.2
            # number of data points
            self.numberOfPoints = x.shape[1]
            # data points that we are going to fit an ellipse to
            # self.data_points = np.concatenate((x[:, None, :], y[:, None, :]), axis=1)
            self.x = x
            self.y = y
            # a list of 2x2 covariance matrices representing the uncertainty
            # in the coordinates of the data points
            self.covList = covList

            # various parameters that determine stopping criteria
            #####################################################
            # step-size tolerance
            self.tolDelta = 1e-7
            # cost tolerance
            self.tolCost = 1e-7
            # parameter tolerance
            self.tolEta = 1e-7
            # gradient tolerance
            self.tolGrad = 1e-7
            # barrier tolerance (prevent ellipse from converging on parabola)
            self.tolBar = 15.5
            # minimum allowable magnitude of conic determinant (prevent ellipse from
            # convering on degenerate parabola (eg. two parallel lines)
            self.tolDet = 1e-5

            # various initial memory allocations
            ####################################
            # allocate space for cost of each iteration
            self.cost = np.zeros((nEllipses, maxiter))
            # allocate space for the latent parameters of each iteration
            self.eta = np.zeros((nEllipses, 5, maxiter))
            # and for the parameters representing the ellipse equation
            self.t = np.zeros((nEllipses, 6, maxiter))
            # allocate space for the parameter direction of each iteration
            self.delta = np.zeros((nEllipses, 5, maxiter))
            # make parameter vector a unit norm vector for numerical stability
            # t = t / norm(t);
            # store the parameters associated with the first iteration
            self.t[..., self.k] = t
            self.eta[..., self.k] = eta
            # start with some random search direction (here we choose all 1's)
            # we can initialise with anything we want, so long as the norm of the
            # vector is not smaller than tolDeta. The initial search direction
            # is not used in any way in the algorithm.
            self.delta[..., self.k] = np.ones(5)

            # More params
            self.jacobian_matrix = None
            self.H = None
            self.r = None
            self.jacob_latentParameters = None

    struct = struct_t()

    Fprim = np.array([
        [0, 0, 2],
        [0, -1, 0],
        [2, 0, 0],
    ])
    F = np.concatenate([
        np.concatenate([Fprim, np.zeros((3, 3))], axis=1),
        np.concatenate([np.zeros((3, 3)), np.zeros((3, 3))], axis=1),
    ], axis=0)
    Identity = np.eye(6)

    # main estimation loop
    while keep_going and struct.k < maxiter:

        # allocate space for residuals
        # struct.r = np.zeros(struct.numberOfPoints)
        # allocate space for the jacobian matrix based on AML component
        # struct.jacobian_matrix = np.zeros((struct.numberOfPoints, 5))
        # grab the current latent parameter estimates
        eta = struct.eta[..., struct.k]
        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        t = np.concatenate((  # (nEllipses, 6)
            np.ones((nEllipses, 1)),
            2*eta[:, 0][:, None],
            (eta[:, 0]**2 + np.abs(eta[:, 1])**2)[:, None],
            eta[:, 2][:, None],
            eta[:, 3][:, None],
            eta[:, 4][:, None],
        ), axis=1)
        
        # jacobian matrix of the transformation from eta to theta parameters
        jacob_latentParameters = np.concatenate((
            np.concatenate([np.zeros((nEllipses, 1))]*5, axis=1)[:, None, :],
            np.concatenate([np.ones((nEllipses, 1))*2] + [np.zeros((nEllipses, 1))]*4, axis=1)[:, None, :],
            np.concatenate([2*eta[:, 0][:, None], (2*np.abs(eta[:, 1])*np.sign(eta[:, 1]))[:, None]] + [np.zeros((nEllipses, 1))]*3, axis=1)[:, None, :],
            np.concatenate([np.zeros((eta.shape[0], 1))]*2 + [np.ones((nEllipses, 1))] + [np.zeros((eta.shape[0], 1))]*2, axis=1)[:, None, :],
            np.concatenate([np.zeros((eta.shape[0], 1))]*3 + [np.ones((nEllipses, 1))] + [np.zeros((nEllipses, 1))], axis=1)[:, None, :],
            np.concatenate([np.zeros((eta.shape[0], 1))]*4 + [np.ones((nEllipses, 1))], axis=1)[:, None, :],
        ), axis=1)

        # we impose the additional constraint that theta will be unit norm
        # so we need to modify the jacobian matrix accordingly
        Pt = np.eye(6) - np.einsum('fi,fj->fij', t, t)/np.linalg.norm(t, axis=-1)**2
        jacob_latentParameters = 1/np.linalg.norm(t, axis=-1)*np.einsum(
            'fij,fjk->fik', Pt, jacob_latentParameters)
        # unit norm constraint
        t /= np.linalg.norm(t, axis=-1)

        # residuals computed on all data points
        # NOTE: loop in original, we vectorize where possible

        # transformed data point
        ux = np.concatenate((  # (nEllipses, nPts, 6)
            x[..., None]**2,
            (x*y)[..., None],
            y[..., None]**2,
            x[..., None],
            y[..., None],
            np.ones(x.shape + (1,)),
        ), axis=-1)

        # derivative of transformed data point
        dux = np.concatenate((  # (nEllipses, nPts, 6, 2)
            np.stack((2*x, y, np.zeros_like(y), np.ones_like(x), np.zeros_like(x), np.zeros_like(x)))[..., None],
            np.stack((np.zeros_like(x), x, 2*y, np.zeros_like(x), np.ones_like(x), np.zeros_like(x)))[..., None],
        ), axis=-1).transpose((1, 2, 0, 3))

        # outer products
        A = np.einsum('fpi,fpj->fpij', ux, ux)
        B = np.einsum('fpij,pfjk,fplk->fpil', dux, covList, dux)
        tBt = np.einsum('fi,fpij,fj->fp', t, B, t)
        tAt = np.einsum('fi,fpij,fj->fp', t, A, t)

        # AML cost for i'th data point (eps for safe division)
        struct.r = np.sqrt(np.abs(tAt/tBt) + np.finfo('float').eps)

        # derivative AML component
        M = A / tBt[..., None, None]
        Xbits = B * tAt[..., None, None] / tBt[..., None, None]**2
        X = M - Xbits

        # gradient for AML cost function (row vector)
        grad = np.einsum('fpij,fj->fpi', X, t)/struct.r[..., None]

        # build up jacobian matrix
        struct.jacobian_matrix = np.einsum('fpi,fij->fpj', grad, jacob_latentParameters)

        # approximate Hessian matrix
        # struct.H = struct.jacobian_matrix.T @ struct.jacobian_matrix
        struct.H = np.einsum('fji,fjk->fik', struct.jacobian_matrix, struct.jacobian_matrix)

        # sum of squares cost for the current iteration
        struct.cost[:, struct.k] = struct.r @ struct.r.T

        struct.jacob_latentParameters = jacob_latentParameters

        # use LevenbergMarquadt step to update parameters
        struct = fastLevenbergMarquardtStep(struct, 2)

        # Preparations for various stopping criteria tests

        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        eta = struct.eta[..., struct.k+1]
        t = np.concatenate((  # (nEllipses, 6)
            np.ones((nEllipses, 1)),
            2*eta[:, 0][:, None],
            (eta[:, 0]**2 + np.abs(eta[:, 1])**2)[:, None],
            eta[:, 2][:, None],
            eta[:, 3][:, None],
            eta[:, 4][:, None],
        ), axis=1)
        t /= np.linalg.norm(t, axis=-1)

        # First criterion checks to see if discriminant approaches zero
        # by using a barrier
        tIt = np.einsum('fi,ij,fj->f', t, Identity, t)
        tFt = np.einsum('fi,ij,fj->f', t, F, t)
        barrier = tIt/tFt

        # Second criterion checks to see if the determinant of conic approaches
        # zero
        M = np.concatenate((
            np.concatenate((t[:, 0][:, None], t[:, 1][:, None]/2, t[:, 3][:, None]/2), axis=1)[:, None, :],
            np.concatenate((t[:, 1][:, None]/2, t[:, 2][:, None], t[:, 4][:, None]/2), axis=1)[:, None, :],
            np.concatenate((t[:, 3][:, None]/2, t[:, 4][:, None]/2, t[:, 5][:, None]), axis=1)[:, None, :],
        ), axis=1)
        DeterminantConic = np.linalg.det(M)

        # Check for various stopping criteria to end the main loop
        # TODO: consider each ellipse index separately!
        if struct.eta_updated:
            etak = struct.eta[..., struct.k]
            etak1 = struct.eta[..., struct.k+1]

            norm_minus = np.linalg.norm(etak1 - etak, axis=-1)
            norm_plus = np.linalg.norm(etak1 + etak, axis=-1)

            costk = struct.cost[:, struct.k]
            costk1 = struct.cost[:, struct.k+1]
            if np.all(np.min(np.stack((norm_minus, norm_plus)), axis=0)) < struct.tolEta:
                keep_going = False
            elif np.all(np.abs(costk - costk1) < struct.tolCost):
                keep_going = False
            elif np.all(np.linalg.norm(struct.delta[..., struct.k+1], axis=-1) < struct.tolDelta):
                keep_going = False
        elif np.all(np.linalg.norm(grad, axis=-1) < struct.tolGrad):
            keep_going = False
        elif (np.all(np.log(barrier) > struct.tolBar) or
              np.all(np.abs(DeterminantConic) < struct.tolDet)):
            keep_going = False

        struct.k += 1

    iterations = struct.k
    theta = struct.t[..., struct.k]
    theta /= np.linalg.norm(theta, axis=-1)

    return(theta, iterations)
