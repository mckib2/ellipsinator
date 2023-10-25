"""LevMar implementation."""

import numpy as np

from ellipsinator._fast_guaranteed_ellipse_funcs.struct_t import StructT


def fastLevenbergMarquardtStep(struct: StructT, rho: float=2):
    """Minimize maximum likelihood cost function of an ellipse.

    This function is used in the main loop of guaranteedEllipseFit in the
    process of minimizing an approximate maximum likelihood cost function
    of an ellipse fit to data.  It computes an update for the parameters
    representing the ellipse, using the method of Levenberg-Marquardt for
    non-linear optimisation.  See [1]_.

    However, unlike the traditional LevenbergMarquardt step, we do not
    add a multiple of the identity matrix to the approximate Hessian,
    but instead a different positive semi-definite matrix. Our choice
    particular choice of the different matrix corresponds to the
    gradient descent direction in the theta coordinate system,
    transformed to the eta coordinate system. We found empirically
    that taking steps according to the theta coordinate system
    instead of the eta coordinate system lead to faster convergence.

    Parameters
    ----------
    struct : StructT
        A data structure containing various parameters needed for the
        optimisation process.
    rho : float, optional

    Returns
    -------
    struct : StructT
        The same data structure 'struct', except that relevant fields
        have been updated

    Notes
    -----
    Original MATLAB implementation by Zygmunt L. Szpak, 18/3/2014.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    """

    # extract variables from data structure
    #######################################

    # Only modify ellipse estimates that are going!
    keep_going = struct.keep_going
    jacobian_matrix = struct.jacobian_matrix[keep_going, ...]
    r = struct.r[keep_going, :]
    lamda = struct.lamda[keep_going]
    delta = struct.delta[keep_going, :, 0]
    damping_multiplier = struct.damping_multiplier
    damping_divisor = struct.damping_divisor
    current_cost = struct.cost[keep_going, 0]
    cov = struct.cov[:, keep_going, ...]
    H = struct.H[keep_going, ...]
    jlp = struct.jacob_latentParameters[keep_going, ...]
    eta = struct.eta[keep_going, :, 0]
    nEllipses = np.sum(keep_going)
    x, y = struct.x[keep_going, ...], struct.y[keep_going, ...]

    # convert latent variables into length-6 vector (called t) representing
    # the equation of an ellipse
    t = np.concatenate((  # (nEllipses, 6)
        np.ones((nEllipses, 1)),
        2*eta[:, 0][:, None],
        (eta[:, 0]**2 + np.abs(eta[:, 1])**rho)[:, None],
        eta[:, 2][:, None],
        eta[:, 3][:, None],
        eta[:, 4][:, None],
    ), axis=1)
    # we impose unit norm constraint on theta
    t /= np.linalg.norm(t, axis=-1, keepdims=True)

    ############################################################
    # compute two potential updates for theta based on different
    # weightings of the identity matrix.
    ############################################################

    jacob = np.einsum('fji,fj->fi', jacobian_matrix, r)
    DMP = np.einsum('fji,fjk->fik', jlp, jlp)*lamda[:, None, None]
    # update_a = -1*(H + DMP)\jacob
    update_a = np.linalg.solve(-1*(H + DMP), jacob)

    # In a similar fashion, the second potential search direction
    # is computed

    # DMP = (jlp.T @ jlp)*lamda/damping_divisor
    DMP /= damping_divisor
    # update_b = - (H+DMP)\jacob;
    update_b = np.linalg.solve(-1*(H + DMP), jacob)

    # the potential new parameters are then
    eta_potential_a = eta + update_a
    eta_potential_b = eta + update_b

    # we need to convert from eta to theta and impose unit norm constraint
    row3 = eta_potential_a[:, 0]**2 + np.abs(eta_potential_a[:, 1])**rho
    t_potential_a = np.concatenate((
        np.ones((nEllipses, 1)),
        2*eta_potential_a[:, 0][:, None],
        row3[:, None],
        eta_potential_a[:, 2][:, None],
        eta_potential_a[:, 3][:, None],
        eta_potential_a[:, 4][:, None],
    ), axis=1)
    t_potential_a /= np.linalg.norm(t_potential_a, axis=-1, keepdims=True)

    row3 = eta_potential_b[:, 0]**2 + np.abs(eta_potential_b[:, 1])**rho
    t_potential_b = np.concatenate((
        np.ones((nEllipses, 1)),
        2*eta_potential_b[:, 0][:, None],
        row3[:, None],
        eta_potential_b[:, 2][:, None],
        eta_potential_b[:, 3][:, None],
        eta_potential_b[:, 4][:, None],
    ), axis=1)
    t_potential_b /= np.linalg.norm(t_potential_b, axis=-1, keepdims=True)

    ########################################################
    # compute new residuals and costs based on these updates
    ########################################################

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
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)
    dux = np.concatenate((  # (nEllipses, nPts, 6, 2)
        np.stack((2*x, y, zeros, ones, zeros, zeros))[..., None],
        np.stack((zeros, x, 2*y, zeros, ones, zeros))[..., None],
    ), axis=-1).transpose((1, 2, 0, 3))

    # outer products
    A = np.einsum('fpi,fpj->fpij', ux, ux)
    B = np.einsum('fpij,pfjk,fplk->fpil', dux, cov, dux)
    t_aBt_a = np.einsum('fi,fpij,fj->fp', t_potential_a, B, t_potential_a)
    t_aAt_a = np.einsum('fi,fpij,fj->fp', t_potential_a, A, t_potential_a)
    t_bBt_b = np.einsum('fi,fpij,fj->fp', t_potential_b, B, t_potential_b)
    t_bAt_b = np.einsum('fi,fpij,fj->fp', t_potential_b, A, t_potential_b)

    # AML cost across all data points
    cost_a = np.sum(np.abs(t_aAt_a/t_aBt_a), axis=-1)
    cost_b = np.sum(np.abs(t_bAt_b/t_bBt_b), axis=-1)

    ################################################################
    # determine appropriate damping and if possible select an update
    ################################################################

    # Global indices of all ellipses that are still being fit
    idx = np.atleast_1d(np.argwhere(keep_going).squeeze())

    # neither update reduced the cost
    no_update_idx = np.logical_and(
        cost_a >= current_cost, cost_b >= current_cost)
    struct.eta_updated[idx[no_update_idx]] = False
    # no change in the cost
    struct.cost[idx[no_update_idx], 1] = current_cost[no_update_idx]
    # no change in parameters
    struct.eta[idx[no_update_idx], :, 1] = eta[no_update_idx, :]
    struct.t[idx[no_update_idx], :, 1] = t[no_update_idx, :]
    # no changes in step direction
    struct.delta[idx[no_update_idx], :, 1] = delta[no_update_idx, :]
    # next iteration add more Identity matrix
    struct.lamda[idx[no_update_idx]] *= damping_multiplier

    # update 'b' reduced the cost function
    cost_b_idx = cost_b < current_cost
    struct.eta_updated[idx[cost_b_idx]] = True
    # store the new cost
    struct.cost[idx[cost_b_idx], 1] = cost_b[cost_b_idx]
    # choose update 'b'
    struct.eta[idx[cost_b_idx], :, 1] = eta_potential_b[cost_b_idx, :]
    struct.t[idx[cost_b_idx], :, 1] = t_potential_b[cost_b_idx, :]
    # store the step direction
    struct.delta[idx[cost_b_idx], :, 1] = update_b[cost_b_idx, :]
    # next iteration add less Identity matrix
    struct.lamda[idx[cost_b_idx]] /= damping_divisor

    # update 'a' reduced the cost function
    cost_a_idx = np.logical_and(
        cost_a < current_cost, np.logical_not(cost_b_idx))
    struct.eta_updated[idx[cost_a_idx]] = True
    # store the new cost
    struct.cost[idx[cost_a_idx], 1] = cost_a[cost_a_idx]
    # choose update 'a'
    struct.eta[idx[cost_a_idx], :, 1] = eta_potential_a[cost_a_idx, :]
    struct.t[idx[cost_a_idx], :, 1] = t_potential_a[cost_a_idx, :]
    # store the step direction
    struct.delta[idx[cost_a_idx], :, 1] = update_a[cost_a_idx, :]
    # keep the same damping for the next iteration
    struct.lamda[idx[cost_a_idx]] = lamda[cost_a_idx]

    # return a data structure containing all the updates
    return struct
