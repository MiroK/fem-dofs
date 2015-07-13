import polynomials.legendre_basis as leg
from points.points import chebyshev_points
from fem.lagrange_element import LagrangeElement
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def cg_optimal_dofs_restricted(deg, vary=0):
    '''
    What are optimal dofs that give smallest condition number in L^2 norm.
    By CG I mean that I constraint two dofs to be at (-1, 1). The remaining
    deg-1 points are to be determined. For even degrees symmetry is dof at 0 is
    also forced
    
    For vary None: the symmetry is used and `all` the points are used for
    search, i.e. this is multi-d problem.

    For vary=int: that guy and its mirror are changed, i.e. this is
    1-d optimization problem.
    '''
    poly_set = leg.basis_functions(deg)
    M = leg.mass_matrix(deg)

    # Only allow negative (make life easier for mirroring) and not midpoint or -1
    if vary: assert 0 < vary < (deg/2 + 1 if deg % 2 else deg/2)

    # Want to minimize this
    def cond_number(x):
        # 2: [-1, y0, -1]
        # (3, 4): [-1, y0, -y0, -1], [-1, -y0, 0, y0, 1],
        # (5, 6): [-1, y0, y1, -y1, -y0, -1], [-1, y0, y1, 0, -y1, -y0, -1]
        # One-d optimzation
        if vary:
            x_ = chebyshev_points(deg)
            x_[vary] = x
            x_[deg-vary] = -x
            dof_set = x_
        # Multi-d optimization
        else:
            if deg == 2:
                dof_set = np.r_[-1, x, 1]
            # Combine no zero
            elif deg % 2 == 1:
                dof_set = np.r_[-1, x, -x[::-1], 1]
            # Combine w/ zero
            else:
                dof_set = np.r_[-1, x, 0, -x[::-1], 1]

        element = LagrangeElement(poly_set, dof_set)
        alpha = element.alpha
        M_  = alpha.dot(M.dot(alpha.T))
        return np.linalg.cond(M_)

    # Initial guess
    # 1d optim
    x0 = chebyshev_points(deg)
    if vary:
        x0 = x0[vary]
    else:
        # Multi-d optim
        if deg == 2:
            x0 = x0[1]
        # Combine no zero
        elif deg % 2 == 1:
            x0 = x0[1:deg/2+1]
        else:
            x0 = x0[1:deg/2]
    
    # Optimize
    res = minimize(cond_number, x0)

    return res, x0, cond_number(x0)

    # TODO add unrestricted to (-1, 1) but keep 0 + symmetries

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    res, x0, fx0 = cg_optimal_dofs_restricted(deg=5, vary=2)

    print 'x0', x0, 'f(x0)', fx0
    print res
