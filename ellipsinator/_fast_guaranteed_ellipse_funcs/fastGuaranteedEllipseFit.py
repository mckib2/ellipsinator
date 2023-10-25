"""Main fitting loop."""

from typing import Tuple

import numpy as np

from ellipsinator._fast_guaranteed_ellipse_funcs.fastLevenbergMarquardtStep import fastLevenbergMarquardtStep
from ellipsinator._fast_guaranteed_ellipse_funcs.struct_t import StructT


def fastGuaranteedEllipseFit(latentParameters: np.ndarray, x: np.ndarray, y: np.ndarray, cov: np.ndarray,
                             maxiter: int=200) -> Tuple[np.ndarray, int]:
    """

    A vectorized implementation of the ellipse fitting algorithm described
    in [1]_ modified to fit M ellipses simultaneously.

    Parameters
    ----------
    latentParameters : array_like (M, 5)
        Initial seeds for M latent parameters [p q r s t] which through a
        transformation are related to parameters  [a b c d e f] associated
        with the conic equation

            a x**2 + b x y + c y**2 + d x + e y + f = 0

        associated with the M ellipses to be fit.
    x, y : array_like (M, N)
        Matrices where M is the number of ellipses to be fit and N is the
        number of data points for each ellipse.
    cov : array_like (N, M, 2, 2)
        NxM 2x2 covariance matrices representing the uncertainty of the
        coordinates of each data point.
    maxiter : int, optional
        Maximum number of iterations to try fitting.

    Returns
    -------
    res : array_like (M, 6)
        M length-6 vectors [a b c d e f] representing the parameters of the
        equation

            a x**2 + b x y + c y**2 + d x + e y + f = 0

        with the additional result that b**2 - 4 a c < 0.

    iters : int
        Number of iterations actually run.

    Notes
    -----
    Original MATLAB implementation by Zygmunt L. Szpak.

    References
    ----------
    .. [1] Z.Szpak, W. Chojnacki and A. van den Hengel "Guaranteed Ellipse
           Fitting with an Uncertainty Measure for Centre, Axes, and
           Orientation"
    """

    nEllipses = x.shape[0]
    nPts = x.shape[1]

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
    t /= np.linalg.norm(t, axis=-1, keepdims=True)

    # various variable initialisations
    ##################################
    # primary loop variable
    keep_going = np.ones(nEllipses, dtype=bool)
    # in some case a LevenbergMarquardtStep does not decrease the cost
    # function and so the parameters (eta) are not updated

    struct = StructT(nEllipses=nEllipses, keep_going=keep_going, nPts=nPts, x=x, y=y, cov=cov, t=t, eta=eta)

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

    # main estimation loop -- only operate on ellipses that haven't
    # been fit yet
    while np.any(keep_going):

        # How many ellipses need to keep_going:
        nEllipses_keep_going = np.sum(keep_going)

        # allocate space for residuals
        # struct.r = np.zeros(struct.numberOfPoints)
        # allocate space for the jacobian matrix based on AML component
        # struct.jacobian_matrix = np.zeros((struct.numberOfPoints, 5))
        # grab the current latent parameter estimates
        eta = struct.eta[keep_going, :, 0]
        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        t = np.concatenate((  # (nEllipses_keep_going, 6)
            np.ones((nEllipses_keep_going, 1)),
            2*eta[:, 0][:, None],
            (eta[:, 0]**2 + np.abs(eta[:, 1])**2)[:, None],
            eta[:, 2][:, None],
            eta[:, 3][:, None],
            eta[:, 4][:, None],
        ), axis=1)

        # jacobian matrix of the transformation from eta to theta parameters
        zeros = np.zeros((nEllipses_keep_going, 1))
        ones = np.ones((nEllipses_keep_going, 1))
        row3 = np.concatenate(
            [2*eta[:, 0][:, None],
             (2*np.abs(eta[:, 1])*np.sign(eta[:, 1]))[:, None]] + [zeros]*3,
            axis=1)
        jacob_latentParameters = np.concatenate((
            np.concatenate([zeros]*5, axis=1)[:, None, :],
            np.concatenate([ones*2] + [zeros]*4, axis=1)[:, None, :],
            row3[:, None, :],
            np.concatenate([zeros]*2 + [ones] + [zeros]*2, axis=1)[:, None, :],
            np.concatenate([zeros]*3 + [ones] + [zeros], axis=1)[:, None, :],
            np.concatenate([zeros]*4 + [ones], axis=1)[:, None, :],
        ), axis=1)

        # we impose the additional constraint that theta will be unit norm
        # so we need to modify the jacobian matrix accordingly
        Pt = np.eye(6) - np.einsum(
            'fi,fj->fij', t, t)/np.linalg.norm(t, axis=-1)[:, None, None]**2
        jacob_latentParameters = np.einsum(
            'fij,fjk->fik', Pt,
            jacob_latentParameters)*1/np.linalg.norm(t, axis=-1)[:, None, None]
        # unit norm constraint
        t /= np.linalg.norm(t, axis=-1, keepdims=True)

        # residuals computed on all data points
        # NOTE: loop in original, we vectorize where possible

        # transformed data point
        ux = np.concatenate((  # (nEllipses_keep_going, nPts, 6)
            x[keep_going, ..., None]**2,
            (x*y)[keep_going, ..., None],
            y[keep_going, ..., None]**2,
            x[keep_going, ..., None],
            y[keep_going, ..., None],
            np.ones((nEllipses_keep_going,) + x.shape[1:] + (1,)),
        ), axis=-1)

        # derivative of transformed data point
        zeros = np.zeros_like(y[keep_going, ...])
        ones = np.ones_like(x[keep_going, ...])
        dux = np.concatenate((  # (nEllipses_keep_going, nPts, 6, 2)
            np.stack((
                2*x[keep_going, ...],
                y[keep_going, ...],
                zeros, ones, zeros, zeros))[..., None],
            np.stack((
                zeros,
                x[keep_going, ...],
                2*y[keep_going, ...],
                zeros, ones, zeros))[..., None],
        ), axis=-1).transpose((1, 2, 0, 3))

        # outer products
        A = np.einsum('fpi,fpj->fpij', ux, ux)
        B = np.einsum(
            'fpij,pfjk,fplk->fpil',
            dux, cov[:, keep_going, ...], dux)
        tBt = np.einsum('fi,fpij,fj->fp', t, B, t)
        tAt = np.einsum('fi,fpij,fj->fp', t, A, t)

        # AML cost over all data point (eps for safe division)
        struct.r[keep_going, :] = np.sqrt(
            np.abs(tAt/tBt) + np.finfo('float').eps)

        # derivative AML component
        M = A / tBt[..., None, None]
        Xbits = B * tAt[..., None, None] / tBt[..., None, None]**2
        X = M - Xbits

        # gradient for AML cost function (row vector)
        struct.grad[keep_going, ...] = np.einsum(
            'fpij,fj->fpi', X, t)/struct.r[keep_going, :, None]

        # build up jacobian matrix
        struct.jacobian_matrix[keep_going, ...] = np.einsum(
            'fpi,fij->fpj',
            struct.grad[keep_going, ...], jacob_latentParameters)

        # approximate Hessian matrix
        # struct.H = struct.jacobian_matrix.T @ struct.jacobian_matrix
        struct.H[keep_going, ...] = np.einsum(
            'fji,fjk->fik',
            struct.jacobian_matrix[keep_going, ...],
            struct.jacobian_matrix[keep_going, ...])

        # sum of squares cost for the current iteration
        struct.cost[keep_going, 0] = np.einsum(
            'fi,fi->f',
            struct.r[keep_going, :],
            struct.r[keep_going, :])

        struct.jacob_latentParameters[keep_going, ...] = jacob_latentParameters

        # use LevenbergMarquadt step to update parameters
        struct = fastLevenbergMarquardtStep(struct, 2)

        # Preparations for various stopping criteria tests

        # convert latent variables into length-6 vector (called t) representing
        # the equation of an ellipse
        eta = struct.eta[keep_going, :, 1]
        t = np.concatenate((  # (nEllipses_keep_going, 6)
            np.ones((np.sum(keep_going), 1)),
            2*eta[:, 0][:, None],
            (eta[:, 0]**2 + np.abs(eta[:, 1])**2)[:, None],
            eta[:, 2][:, None],
            eta[:, 3][:, None],
            eta[:, 4][:, None],
        ), axis=1)
        t /= np.linalg.norm(t, axis=-1, keepdims=True)

        # First criterion checks to see if discriminant approaches zero
        # by using a barrier
        tIt = np.einsum('fi,ij,fj->f', t, Identity, t)
        tFt = np.einsum('fi,ij,fj->f', t, F, t)
        struct.barrier[keep_going] = tIt/tFt

        # Second criterion checks to see if the determinant of conic approaches
        # zero
        M = np.concatenate((
            np.concatenate((
                t[:, 0][:, None],
                t[:, 1][:, None]/2,
                t[:, 3][:, None]/2), axis=1)[:, None, :],
            np.concatenate((
                t[:, 1][:, None]/2,
                t[:, 2][:, None],
                t[:, 4][:, None]/2), axis=1)[:, None, :],
            np.concatenate((
                t[:, 3][:, None]/2,
                t[:, 4][:, None]/2,
                t[:, 5][:, None]), axis=1)[:, None, :],
        ), axis=1)
        struct.DeterminantConic[keep_going] = np.linalg.det(M)

        # Check for various stopping criteria to end the main loop
        # TODO: clean this up -- it's a little messy...

        # Count iterations before we  potentially change keep_going
        struct.iters[keep_going] += 1

        # For each ellipse that has eta_updated
        eta_updated_idx = struct.eta_updated
        if np.any(eta_updated_idx):
            etak = struct.eta[eta_updated_idx, :, 0]
            etak1 = struct.eta[eta_updated_idx, :, 1]
            norm_minus = np.linalg.norm(etak1 - etak, axis=-1, keepdims=True)
            norm_plus = np.linalg.norm(etak1 + etak, axis=-1, keepdims=True)
            stop_local_idx = np.min(np.stack((
                norm_minus, norm_plus)), axis=0) < struct.tolEta
            stop_local_idx = stop_local_idx.squeeze()
            stop_idx = np.atleast_1d(
                np.argwhere(eta_updated_idx).squeeze())[stop_local_idx]
            keep_going[stop_idx] = False

            costk = struct.cost[eta_updated_idx, 0]
            costk1 = struct.cost[eta_updated_idx, 1]
            stop_local_idx = np.abs(costk - costk1) < struct.tolCost
            stop_idx = np.atleast_1d(
                np.argwhere(eta_updated_idx).squeeze())[stop_local_idx]
            keep_going[stop_idx] = False

            stop_local_idx = (np.linalg.norm(
                struct.delta[eta_updated_idx, :, 1],
                axis=-1, keepdims=True) < struct.tolDelta).squeeze()
            stop_idx = np.atleast_1d(
                np.argwhere(eta_updated_idx).squeeze())[stop_local_idx]
            keep_going[stop_idx] = False

        eta_not_updated_idx = np.logical_not(eta_updated_idx)
        if np.any(eta_not_updated_idx):
            stop_local_idx = (np.linalg.norm(
                struct.grad[eta_not_updated_idx, ...],
                axis=(1, 2), ord=2) < struct.tolGrad).squeeze()
            stop_idx = np.atleast_1d(
                np.argwhere(eta_not_updated_idx).squeeze())[stop_local_idx]
            keep_going[stop_idx] = False

            stop_local_idx = np.logical_or(
                np.log(struct.barrier[eta_not_updated_idx]) > struct.tolBar,
                np.abs(struct.DeterminantConic[eta_not_updated_idx])
                < struct.tolDet)
            stop_idx = np.atleast_1d(
                np.argwhere(eta_not_updated_idx).squeeze())[stop_local_idx]
            keep_going[stop_idx] = False

        # Timeout on iters
        keep_going[struct.iters >= maxiter] = False
        struct.keep_going = keep_going

        # shuffle current/future values
        struct.cost[:, 0] = struct.cost[:, 1]
        struct.eta[..., 0] = struct.eta[..., 1]
        struct.t[..., 0] = struct.t[..., 1]
        struct.delta[..., 0] = struct.delta[..., 1]

    theta = struct.t[..., 0]
    theta /= np.linalg.norm(theta, axis=-1, keepdims=True)

    return theta, struct.iters
