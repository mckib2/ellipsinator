
import numpy as np

def fastLevenbergMarquardtStep(struct, rho):
    '''
    Function: fastLevenbergMarquardtStep

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

    Parameters:

      struct     - a data structure containing various parameters
                   needed for the optimisation process.

    Returns:

     the same data structure 'struct', except that relevant fields have
     been updated

    See Also:

    fastGuaranteedEllipseFit

    Zygmunt L. Szpak (c) 2014
    Last modified 18/3/2014
    '''

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # extract variables from data structure                              %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    jacobian_matrix = struct.jacobian_matrix
    r = struct.r
    lamda = struct.lamda
    delta = struct.delta[struct.k-1]
    damping_multiplier = struct.damping_multiplier
    damping_divisor = struct.damping_divisor
    current_cost = struct.cost[struct.k-1]
    data_points = struct.data_points
    covList = struct.covList
    numberOfPoints = struct.numberOfPoints
    H = struct.H
    jlp = struct.jacob_latentParameters
    eta = struct.eta[:, struct.k-1]

    # convert latent variables into length-6 vector (called t) representing
    # the equation of an ellipse
    t = np.array([
        [1, 2*eta[0], eta[0]**2 + np.abs(eta[1])**rho, eta[2], eta[3], eta[4]],
    ]).T
    # we impose unit norm constraint on theta
    t /= np.linalg.norm(t)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # compute two potential updates for theta based on different         %
    # weightings of the identity matrix.                                 %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    jacob = jacobian_matrix.T @ r
    DMP = (jlp.T @ jlp) * lamda
    #update_a = -1*(H + DMP)\jacob
    update_a = np.linalg.solve(-1*(H + DMP), jacob)

    # In a similar fashion, the second potential search direction
    # is computed

    DMP = (jlp.T @ jlp)*lamda/damping_divisor
    #update_b = - (H+DMP)\jacob;
    update_b = np.linalg.solve(-1*(H + DMP), jacob)

    # the potential new parameters are then
    eta_potential_a = eta + update_a.squeeze()
    eta_potential_b = eta + update_b.squeeze()

    # we need to convert from eta to theta and impose unit norm constraint
    t_potential_a = np.array([
        [1, 2*eta_potential_a[0], eta_potential_a[0]**2 + np.abs(eta_potential_a[1])**rho, eta_potential_a[2], eta_potential_a[3], eta_potential_a[4]],
    ]).T
    t_potential_a /= np.linalg.norm(t_potential_a)

    t_potential_b = np.array([
        [1, 2*eta_potential_b[0], eta_potential_b[0]**2 + np.abs(eta_potential_b[1])**rho, eta_potential_b[2], eta_potential_b[3], eta_potential_b[4]],
    ]).T
    t_potential_b /= np.linalg.norm(t_potential_b)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # compute new residuals and costs based on these updates             %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # residuals computed on data points
    cost_a = 0
    cost_b = 0
    for ii in range(numberOfPoints):
        m = data_points[:, ii]
        # transformed data point
        ux_i = np.array([
            [m[0]**2, m[0]*m[1], m[1]**2, m[0], m[1], 1],
        ]).T
        # derivative of transformed data point
        dux_i = np.array([
            [2*m[0], m[1], 0, 1, 0, 0],
            [0, m[0], 2*m[1], 0, 1, 0],
        ]).T

        # outer product
        A = ux_i @ ux_i.T

        # covariance matrix of the ith data pont
        covX_i = covList[ii]

        B = dux_i @ covX_i @ dux_i.T

        t_aBt_a = (t_potential_a.T @ B @ t_potential_a).squeeze()
        t_aAt_a = (t_potential_a.T  @ A @ t_potential_a).squeeze()

        t_bBt_b = (t_potential_b.T @ B @ t_potential_b).squeeze()
        t_bAt_b = (t_potential_b.T  @ A @ t_potential_b).squeeze()

        # AML cost for i'th data point
        cost_a += np.abs(t_aAt_a/t_aBt_a)
        cost_b += np.abs(t_bAt_b/t_bBt_b)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # determine appropriate damping and if possible select an update     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if cost_a >= current_cost and cost_b >= current_cost:
        # neither update reduced the cost
        struct.eta_updated = False
        # no change in the cost
        struct.cost[struct.k] = current_cost
        # no change in parameters
        struct.eta[:, struct.k] = eta
        struct.t[:, struct.k] = t
        # no changes in step direction
        struct.delta[:, struct.k] = delta
        # next iteration add more Identity matrix
        struct.lamda *= damping_multiplier
    elif cost_b < current_cost:
        # update 'b' reduced the cost function
        struct.eta_updated = True
        # store the new cost
        struct.cost[struct.k] = cost_b
        # choose update 'b'
        struct.eta[:, struct.k] = eta_potential_b
        struct.t[:, struct.k] = t_potential_b.squeeze()
        # store the step direction
        struct.delta[:, struct.k] = update_b.T
        # next iteration add less Identity matrix
        struct.lamda /= damping_divisor
    else:
        # update 'a' reduced the cost function
        struct.eta_updated = True
        # store the new cost
        struct.cost[struct.k] = cost_a
        # choose update 'a'
        struct.eta[:, struct.k] = eta_potential_a
        struct.t[:, struct.k] = t_potential_a
        # store the step direction
        struct.delta[:, struct.k] = update_a.T
        # keep the same damping for the next iteration
        struct.lamda = lamda

    # return a data structure containing all the updates
    return struct
