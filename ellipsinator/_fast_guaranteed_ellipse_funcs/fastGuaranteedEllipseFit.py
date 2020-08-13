
import numpy as np

from .fastLevenbergMarquardtStep import fastLevenbergMarquardtStep

def fastGuaranteedEllipseFit(latentParameters, dataPts, covList):
    '''

    This function implements the ellipse fitting algorithm described in
    Z.Szpak, W. Chojnacki and A. van den Hengel
    "Guaranteed Ellipse Fitting with an Uncertainty Measure for Centre,
    Axes, and Orientation"

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
    '''

    eta = latentParameters
    # convert latent variables into length-6 vector (called t) representing
    # the equation of an ellipse
    t = np.array([
        [1, 2*eta[0], eta[0]**2 + np.abs(eta[1])**2, eta[2], eta[3], eta[4]],
    ]).T
    t /= np.linalg.norm(t)

    # various variable initialisations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # primary loop variable
    keep_going = True
    # in some case a LevenbergMarquardtStep does not decrease the cost
    # function and so the parameters (eta) are not updated
    class struct:
        pass
    struct.eta_updated = False
    # damping parameter in LevenbergMarquadtStep
    struct.lamda = 0.01
    # loop counter (matlab arrays start at index 1, not index 0)
    struct.k = 1
    # used to modify the tradeoff between gradient descent and hessian based
    # descent in LevenbergMarquadtStep
    struct.damping_multiplier = 15
    # used to modify the tradeoff between gradient descent and hessian based
    # descent in LevenbergMarquadtStep
    struct.damping_divisor = 1.2
    # number of data points
    struct.numberOfPoints = dataPts.shape[1]
    # data points that we are going to fit an ellipse to
    struct.data_points = dataPts
    # a list of 2x2 covariance matrices representing the uncertainty
    # in the coordinates of the data points
    struct.covList = covList

    # various parameters that determine stopping criteria
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # maximum loop iterations
    maxIter = 200
    # step-size tolerance
    struct.tolDelta = 1e-7
    # cost tolerance
    struct.tolCost = 1e-7
    # parameter tolerance
    struct.tolEta = 1e-7
    # gradient tolerance
    struct.tolGrad = 1e-7
    # barrier tolerance (prevent ellipse from converging on parabola)
    struct.tolBar = 15.5
    # minimum allowable magnitude of conic determinant (prevent ellipse from
    # convering on degenerate parabola (eg. two parallel lines)
    struct.tolDet = 1e-5

    Fprim = np.array([
        [0, 0,  2],
        [0, -1, 0],
        [2, 0,  0],
    ])
    F = np.concatenate([
        np.concatenate([Fprim, np.zeros((3, 3))], axis=1),
        np.concatenate([np.zeros((3, 3)), np.zeros((3, 3))], axis=1),
    ], axis=0)
    I = np.eye(6)

    # various initial memory allocations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # allocate space for cost of each iteration
    struct.cost = np.zeros(maxIter)
    # allocate space for the latent parameters of each iteration
    struct.eta = np.zeros((5, maxIter))
    # and for the parameters representing the ellipse equation
    struct.t = np.zeros((6, maxIter))
    # allocate space for the parameter direction of each iteration
    struct.delta = np.zeros((5, maxIter))
    # make parameter vector a unit norm vector for numerical stability
    #t = t / norm(t);
    # store the parameters associated with the first iteration
    struct.t[:, struct.k-1] = t.squeeze()
    struct.eta[:, struct.k-1] = eta
    # start with some random search direction (here we choose all 1's)
    # we can initialise with anything we want, so long as the norm of the
    # vector is not smaller than tolDeta. The initial search direction
    # is not used in any way in the algorithm.
    struct.delta[:, struct.k-1] = np.ones(5)
    # main estimation loop
    while keep_going and struct.k < maxIter:

        # allocate space for residuals
        struct.r = np.zeros(struct.numberOfPoints)
        # allocate space for the jacobian matrix based on AML component
        struct.jacobian_matrix = np.zeros((struct.numberOfPoints, 5))
        # grab the current latent parameter estimates
        eta = struct.eta[:, struct.k-1]
        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        t = np.array([
            [1, 2*eta[0], eta[0]**2 + np.abs(eta[1])**2, eta[2], eta[3], eta[4]],
        ]).T

        # jacobian matrix of the transformation from eta to theta parameters
        jacob_latentParameters = np.array([
            [0,                     0,                          0, 0, 0],
            [2,                     0,                          0, 0, 0],
            [2*eta[0], 2*np.abs(eta[1])*np.sign(eta[1]),        0, 0, 0],
            [0,                     0,                          1, 0, 0],
            [0,                     0,                          0, 1, 0],
            [0,                     0,                          0, 0, 1],
        ])

        # we impose the additional constraint that theta will be unit norm
        # so we need to modify the jacobian matrix accordingly
        Pt = np.eye(6) - ((t @ t.T)/(np.linalg.norm(t)**2))
        jacob_latentParameters = (1/np.linalg.norm(t))*Pt @ jacob_latentParameters
        # unit norm constraint
        t /= np.linalg.norm(t)

        # residuals computed on data points
        for ii in range(struct.numberOfPoints):
            m = dataPts[:, ii]
            # transformed data point
            ux_i = np.array([
                [m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1],
            ]).T
            # derivative of transformed data point
            dux_i = np.array([
                [2*m[0], m[1], 0,      1, 0, 0],
                [0,      m[0], 2*m[1], 0, 1, 0],
            ]).T

            # outer product
            A = ux_i @ ux_i.T

            # covariance matrix of the ith data pont
            covX_i = covList[ii]

            B = dux_i @ covX_i @ dux_i.T

            tBt = (t.T @ B @ t).squeeze() # scalar
            tAt = (t.T @ A @ t).squeeze() # scalar

            # AML cost for i'th data point
            struct.r[ii] = np.sqrt(np.abs(tAt/tBt))

            # derivative AML component
            M = A / tBt
            Xbits = B * tAt / tBt**2
            X = M - Xbits

            # gradient for AML cost function (row vector)
            grad = ((X @ t) / np.sqrt((np.abs(tAt/tBt) + 10*np.finfo('float').eps))).T

            # build up jacobian matrix
            struct.jacobian_matrix[ii, :] = grad @ jacob_latentParameters

        # approximate Hessian matrix
        struct.H =  struct.jacobian_matrix.T @ struct.jacobian_matrix

        # sum of squares cost for the current iteration
        struct.cost[struct.k-1] = struct.r.T @ struct.r

        struct.jacob_latentParameters = jacob_latentParameters

        # use LevenbergMarquadt step to update parameters
        struct = fastLevenbergMarquardtStep(struct, 2)

        # Preparations for various stopping criteria tests


        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        eta = struct.eta[:, struct.k]
        t = np.array([
            [1, 2*eta[0], eta[0]**2 + np.abs(eta[1])**2, eta[2], eta[3], eta[4]],
        ]).T
        t /= np.linalg.norm(t)

        # First criterion checks to see if discriminant approaches zero by using
        # a barrier
        tIt = (t.T @ I @ t).squeeze()
        tFt = (t.T @ F @ t).squeeze()
        barrier = tIt/tFt

        # Second criterion checks to see if the determinant of conic approaches
        # zero
        M = np.array([
            [t[0], t[1]/2, t[3]/2],
            [t[1]/2, t[2], t[4]/2],
            [t[3]/2, t[4]/2, t[5]],
        ]).squeeze()
        DeterminantConic = np.linalg.det(M)

        # Check for various stopping criteria to end the main loop
        if np.min([np.linalg.norm(struct.eta[:, struct.k] - struct.eta[:, struct.k-1]), np.linalg.norm(struct.eta[:, struct.k] + struct.eta[:, struct.k-1])]) < struct.tolEta and struct.eta_updated:
            keep_going = False
        elif np.abs(struct.cost[struct.k-1] - struct.cost[struct.k]) < struct.tolCost and struct.eta_updated:
            keep_going = False
        elif np.linalg.norm(struct.delta[:, struct.k]) < struct.tolDelta and struct.eta_updated:
            keep_going = False
        elif np.linalg.norm(grad) < struct.tolGrad:
            keep_going = False
        elif np.log(barrier) > struct.tolBar or np.abs(DeterminantConic) < struct.tolDet:
            keep_going = False

        struct.k += 1

    iterations = struct.k
    theta = struct.t[:, struct.k-1]
    theta /= np.linalg.norm(theta)

    return(theta, iterations)
