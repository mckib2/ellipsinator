
import numpy as np


class StructT:
    """simply hold parameters."""
    def __init__(self, nEllipses, keep_going, nPts, x, y, cov, t, eta):
        self.nEllipses = nEllipses
        self.keep_going = keep_going
        self.eta_updated = np.zeros(nEllipses, dtype=bool)
        # damping parameter in LevenbergMarquadtStep
        self.lamda = np.ones(nEllipses ) *0.01
        self.iters = np.zeros(nEllipses, dtype=int)  # loop counter
        # used to modify the tradeoff between gradient descent and hessian
        # based descent in LevenbergMarquadtStep
        self.damping_multiplier = 15
        # used to modify the tradeoff between gradient descent and hessian
        # based descent in LevenbergMarquadtStep
        self.damping_divisor = 1.2
        # number of data points
        self.numberOfPoints = nPts
        # data points that we are going to fit an ellipse to
        self.x = x
        self.y = y
        # a list of 2x2 covariance matrices representing the uncertainty
        # in the coordinates of the data points
        self.cov = cov

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
        # minimum allowable magnitude of conic determinant (prevent ellipse
        # from convering on degenerate parabola (eg. two parallel lines)
        self.tolDet = 1e-5

        # various initial memory allocations
        ####################################
        # allocate space for cost of each iteration
        # Don't allocate maxiter -- only need 2 (current/next)!
        self.cost = np.empty((nEllipses, 2))
        # allocate space for the latent parameters of each iteration
        self.eta = np.empty((nEllipses, 5, 2))
        # and for the parameters representing the ellipse equation
        self.t = np.empty((nEllipses, 6, 2))
        # allocate space for the parameter direction of each iteration
        self.delta = np.empty((nEllipses, 5, 2))
        # make parameter vector a unit norm vector for numerical stability
        # t = t / norm(t);
        # store the parameters associated with the first iteration
        self.t[..., 0] = t
        self.eta[..., 0] = eta
        # start with some random search direction (here we choose all 1's)
        # we can initialise with anything we want, so long as the norm of
        # the vector is not smaller than tolDeta. The initial search
        # direction is not used in any way in the algorithm.
        self.delta[..., 0] = np.ones(5)

        # More params
        self.jacobian_matrix = np.empty((nEllipses, nPts, 5))
        self.H = np.empty((nEllipses, 5, 5))
        self.r = np.empty((nEllipses, nPts))
        self.jacob_latentParameters = np.empty((nEllipses, 6, 5))

        # tmps
        self.grad = np.empty((nEllipses, nPts, 6))
        self.barrier = np.empty(nEllipses)
        self.DeterminantConic = np.empty(nEllipses)
