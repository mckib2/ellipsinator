'''
WARNING: This is not yet as polished as fit_ellipse_halir!
'''

import logging

import numpy as np

def fit_ellipse_fitzgibon(x, y):
    '''Python port of direct ellipse fitting algorithm by Fitzgibon et. al.

    Parameters
    ----------
    x : array_like
        y coordinates assumed to be on ellipse.
    y : array_like
        y coordinates assumed to be on ellipse.

    Returns
    -------
    res : array_like (6,)
        Ellipse coefficients.

    Notes
    -----
    See Figure 1 from [1]_.
    Also see previous python port:
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    References
    ==========
    .. [1] Halır, Radim, and Jan Flusser. "Numerically stable direct least
           squares fitting of ellipses." Proc. 6th International Conference in
           Central Europe on Computer Graphics and Visualization. WSCG. Vol.
           98. 1998.
    '''

    # Like a pancake...
    x = x.flatten()
    y = y.flatten()

    # Make sure we have at least 6 points (6 unknowns...)
    if x.size < 6 and y.size < 6:
        logging.warning('We need at least 6 sample points for a good fit!')

    # Do the thing
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x))) # Design matrix
    S = np.dot(D.T, D) # Scatter matrix
    C = np.zeros([6, 6]) # Constraint matrix
    C[(0, 2), (0, 2)] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C)) # solve eigensystem
    n = np.argmax(np.abs(E)) # find positive eigenvalue
    a = V[:, n].squeeze() # corresponding eigenvector
    return a
