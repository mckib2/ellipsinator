
import numpy as np

def get_center(c):
    '''Compute center of ellipse from implicit function coefficients.
    Parameters
    ----------
    c : array_like (6,)
        Coefficients of general quadratic polynomial function for
        conic funs.

    Returns
    -------
    (xc, yc) : tuple of float
        (x,y) coordinate of center.
    '''
    A, B, C, D, E, _F = c[:]
    den = B**2 - 4*A*C
    xc = (2*C*D - B*E)/den
    yc = (2*A*E - B*D)/den
    return(xc, yc)

def rotate_points(x, y, phi, p=(0, 0)):
    '''Rotate points x, y through angle phi w.r.t. point p.

    Parameters
    ----------
    x : array_like (N,)
        x coordinates of points to be rotated.
    y : array_like (N,)
        y coordinates of points to be rotated.
    phi : float
        Angle in radians to rotate points.
    p : tuple (2,), optional
        Point to rotate around.

    Returns
    -------
    (xr, yr) : tuple (2,) of array_like (N,)
        (x,y) coordinates of rotated points.
    '''
    x = x.flatten()
    y = y.flatten()
    xr = (x - p[0])*np.cos(phi) - (y - p[0])*np.sin(phi) + p[0]
    yr = (y - p[1])*np.cos(phi) + (x - p[1])*np.sin(phi) + p[1]
    return(xr, yr)

def get_semiaxes(c):
    '''Solve for semi-axes of the cartesian form of ellipse equation.
    Parameters
    ----------
    c : array_like (6,)
        Coefficients of general quadratic polynomial function for
        conic functions.

    Returns
    -------
    tuple (2,) of float
        Semi-major/minor axes

    Notes
    -----
    https://en.wikipedia.org/wiki/Ellipse
    '''
    A, B, C, D, E, F = c[:]
    B2 = B**2
    den = B2 - 4*A*C
    num = 2*(A*E**2 + C*D**2 - B*D*E + den*F)
    num *= (A + C + np.array([1, -1])*np.sqrt((A - C)**2 + B2))
    AB = -1*np.sqrt(num)/den

    return(AB[0], AB[1])

def rotate_coefficients(c, phi):
    '''Rotate coefficients of implicit equations through angle phi.

    Parameters
    ----------
    c : array_like (6,)
        Coefficients of general quadratic polynomial function for conic funs.
    phi : float
        Angle in radians to rotate ellipse.

    Returns
    -------
    res : array_like (6,)
        Coefficients of rotated ellipse.

    References
    ----------
    .. [1] http://www.mathamazement.com/Lessons/Pre-Calculus/09_Conic-Sections-and-Analytic-Geometry/rotation-of-axes.html
    '''
    cp, c2p = np.cos(phi), np.cos(2*phi)
    sp, s2p = np.sin(phi), np.sin(2*phi)
    A, B, C, D, E, F = c[:]
    Ar = (A + C + (A - C)*c2p - B*s2p)/2
    Br = (A - C)*s2p + B*c2p
    Cr = (A + C + (C - A)*c2p + B*s2p)/2
    Dr = D*cp - E*sp
    Er = D*sp + E*cp
    return np.array([Ar, Br, Cr, Dr, Er, F])

def check_fit(C, x, y):
    '''General quadratic polynomial function.

    Parameters
    ----------
    C : array_like (6,)
        coefficients.
    x : array_like (N,)
        x coordinates assumed to be on ellipse.
    y : array_like (M,)
        y coordinates assumed to be on ellipse.

    Returns
    -------
    res : float
        Measure of how well the ellipse fits the points (x, y).

    Notes
    -----
    We want this to equal 0 for a good ellipse fit.   This polynomial is called
    the algebraic distance of the point (x, y) to the given conic.
    This equation is referenced in [1]_ and [2]_.

    References
    ----------
    .. [1] Shcherbakova, Yulia, et al. "PLANET: an ellipse fitting approach for
           simultaneous T1 and T2 mapping using phase‐cycled balanced
           steady‐state free precession." Magnetic resonance in medicine 79.2
           (2018): 711-722.
    .. [2] Halır, Radim, and Jan Flusser. "Numerically stable direct least
           squares fitting of ellipses." Proc. 6th International Conference in
           Central Europe on Computer Graphics and Visualization. WSCG. Vol.
           98. 1998.
    '''
    x = x.flatten()
    y = y.flatten()
    return C[0]*x**2 + C[1]*x*y + C[2]*y**2 + C[3]*x + C[4]*y + C[5]
