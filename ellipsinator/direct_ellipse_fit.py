
import numpy as np


def direct_ellipse_fit(data: np.ndarray) -> np.ndarray:
    """
    This code is an implementation of the following paper

    R. Halif and J. Flusser
    Numerically stable direct least squares fitting of ellipses
    Proc. 6th International Conference in Central Europe on Computer Graphics
    and Visualization. WSCG '98
    Czech Republic,125--132, feb, 1998
    """
    x = data[0, :]
    y = data[1, :]

    D1 = np.stack((x**2, x*y, y**2)).T # quadratic part of the design matrix
    D2 = np.stack((x, y, np.ones_like(x))).T    # linear part of the design matrix
    S1 = D1.T @ D1                 # quadratic part of the scatter matrix
    S2 = D1.T @ D2                 # combined part of the scatter matrix
    S3 = D2.T @ D2                 # linear part of the scatter matrix
    T = -1*np.linalg.inv(S3) @ S2.T           # for getting a2 from a1
    M = S1 + S2 @ T               # reduce scatter matrix
    M = np.array([
        M[2, :]/2,
        -1*M[1, :],
        M[0, :]/2,
    ])  # premultiply by inv(C1)
    evalue, evec = np.linalg.eig(M)       # solve eigensystem
    cond = 4*evec[0, :]*evec[2, :] - evec[1, :]**2  # evaluate a'Ca
    al = evec[:, cond > 0]   # eigenvector for min. pos. eigenvalue
    a = np.concatenate((al, T @ al))              # ellipse coefficients
    a /= np.linalg.norm(a)
    return a.squeeze()
