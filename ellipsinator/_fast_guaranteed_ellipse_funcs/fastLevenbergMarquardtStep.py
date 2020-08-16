'''LevMar implementation.'''

import numpy as np


def fastLevenbergMarquardtStep(struct, rho):
    '''

    This function is used in the main loop of guaranteedEllipseFit in the
    process of minimizing an approximate maximum likelihood cost function
    of an ellipse fit to data.  It computes an update for the parameters
    representing the ellipse, using the method of Levenberg-Marquardt for
    non-linear optimisation.
    See: http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

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
    struct
        a data structure containing various parameters
        needed for the optimisation process.

    Returns
    -------
        the same data structure 'struct', except that relevant fields
        have been updated


    Notes
    -----
    Zygmunt L. Szpak (c) 2014
    Last modified 18/3/2014
    '''

    # extract variables from data structure
    #######################################

    jacobian_matrix = struct.jacobian_matrix
    r = struct.r
    lamda = struct.lamda
    delta = struct.delta[..., struct.k]
    damping_multiplier = struct.damping_multiplier
    damping_divisor = struct.damping_divisor
    current_cost = struct.cost[:, struct.k]
    x, y = struct.x, struct.y
    covList = struct.covList
    H = struct.H
    jlp = struct.jacob_latentParameters
    eta = struct.eta[..., struct.k]
    nEllipses = struct.nEllipses

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
    t /= np.linalg.norm(t)

    ############################################################
    # compute two potential updates for theta based on different
    # weightings of the identity matrix.
    ############################################################

    jacob = np.einsum('fji,fj->fi', jacobian_matrix, r)
    DMP = np.einsum('fji,fjk->fik', jlp, jlp)*lamda
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
    t_potential_a = np.concatenate((
        np.ones((nEllipses, 1)),
        2*eta_potential_a[:, 0][:, None],
        (eta_potential_a[:, 0]**2 + np.abs(eta_potential_a[:, 1])**rho)[:, None],
        eta_potential_a[:, 2][:, None],
        eta_potential_a[:, 3][:, None],
        eta_potential_a[:, 4][:, None],
    ), axis=1)
    t_potential_a /= np.linalg.norm(t_potential_a, axis=-1)

    t_potential_b = np.concatenate((
        np.ones((nEllipses, 1)),
        2*eta_potential_b[:, 0][:, None],
        (eta_potential_b[:, 0]**2 + np.abs(eta_potential_b[:, 1])**rho)[:, None],
        eta_potential_b[:, 2][:, None],
        eta_potential_b[:, 3][:, None],
        eta_potential_b[:, 4][:, None],
    ), axis=1)
    t_potential_b /= np.linalg.norm(t_potential_b, axis=-1)

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
    dux = np.concatenate((  # (nEllipses, nPts, 6, 2)
        np.stack((2*x, y, np.zeros_like(y), np.ones_like(x), np.zeros_like(x), np.zeros_like(x)))[..., None],
        np.stack((np.zeros_like(x), x, 2*y, np.zeros_like(x), np.ones_like(x), np.zeros_like(x)))[..., None],
    ), axis=-1).transpose((1, 2, 0, 3))

    # outer products
    A = np.einsum('fpi,fpj->fpij', ux, ux)
    B = np.einsum('fpij,pfjk,fplk->fpil', dux, covList, dux)
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

    # TODO: consider each ellipse index separately!
    if cost_a >= current_cost and cost_b >= current_cost:
        # neither update reduced the cost
        struct.eta_updated = False
        # no change in the cost
        struct.cost[:, struct.k+1] = current_cost
        # no change in parameters
        struct.eta[..., struct.k+1] = eta
        struct.t[..., struct.k+1] = t
        # no changes in step direction
        struct.delta[..., struct.k+1] = delta
        # next iteration add more Identity matrix
        struct.lamda *= damping_multiplier
    elif cost_b < current_cost:
        # update 'b' reduced the cost function
        struct.eta_updated = True
        # store the new cost
        struct.cost[:, struct.k+1] = cost_b
        # choose update 'b'
        struct.eta[..., struct.k+1] = eta_potential_b
        struct.t[..., struct.k+1] = t_potential_b.squeeze()
        # store the step direction
        struct.delta[..., struct.k+1] = update_b
        # next iteration add less Identity matrix
        struct.lamda /= damping_divisor
    else:
        # update 'a' reduced the cost function
        struct.eta_updated = True
        # store the new cost
        struct.cost[:, struct.k+1] = cost_a
        # choose update 'a'
        struct.eta[..., struct.k+1] = eta_potential_a
        struct.t[..., struct.k+1] = t_potential_a
        # store the step direction
        struct.delta[..., struct.k+1] = update_a
        # keep the same damping for the next iteration
        struct.lamda = lamda

    # return a data structure containing all the updates
    return struct
