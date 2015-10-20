from __future__ import division
import sys
sys.path.append('../')
from mesh import IntervalMesh, Cells
from function_space import FunctionSpace
from function import Function, Expression, Constant
from hermite_element import HermiteElement
from assembler import assemble_matrix
from plotting import plot, show
from bcs import DirichletBC
from utils import colors

from lagrange_element import LagrangeElement
from polynomials import legendre_basis as leg
from points import chebyshev_points

from math import log, sqrt
import numpy as np
from sympy import sin, pi, Symbol, exp, integrate, Matrix
from scipy.sparse.linalg import spsolve

from plotting import plot, show
import matplotlib.pyplot as plt
import random


def solve(n_cells, degree=3, with_plot=False):
    # Problem
    x = Symbol('x')
    u = -0.0625*x**3/pi**3 + 0.0625*x/pi**3 + sin(2*pi*x)/(16*pi**4)
    f = sin(2*pi*x)

    # As Expr
    u = Expression(u)
    f = Expression(f)

    # Space
    element = HermiteElement(degree)

    mesh = IntervalMesh(a=-1, b=1, n_cells=n_cells)
    V = FunctionSpace(mesh, element)
    bc = DirichletBC(V, u)

    # Need mass matrix to intefrate the rhs
    M = assemble_matrix(V, 'mass', get_geom_tensor=None, timer=0)
    # NOTE We cannot you apply the alpha transform idea because the functions
    # are mapped with this selective weight on 2nd, 3rd functions. So some rows
    # of alpha would have to be multiplied by weights which are cell specific.
    # And then on top of this there would be a dx = J*dy term. Better just to
    # use the qudrature representations
    # Mpoly_matrix = leg.mass_matrix(degree)
    # M_ = assemble_matrix(V, Mpoly_matrix, Mget_geom_tensor, timer=0)
    
    # Stiffness matrix for Laplacian
    A = assemble_matrix(V, 'bending', get_geom_tensor=None, timer=0)
    # NOTE the above
    # Apoly_matrix = leg.stiffness_matrix(degree)
    # A_ = assemble_matrix(V, Apoly_matrix, Aget_geom_tensor, timer=0)
  
    # Interpolant of source
    fV = V.interpolate(f)
    # Integrate in L2 to get the vector
    b = M.dot(fV.vector)
    
    # Apply boundary conditions
    bc.apply(A, b, True)
    x = spsolve(A, b)

    print '>>>>', np.linalg.norm(x - V.interpolate(u).vector)

    # As function
    uh = Function(V, x)

    # This is a (slow) way of plotting the high order
    if with_plot:
        fig = plt.figure()
        ax = fig.gca()
        uV = V.interpolate(u)
        
        for cell in Cells(mesh):
            a, b = cell.vertices[0, 0], cell.vertices[1, 0]
            x = np.linspace(a, b, 100)

            y = uh.eval_cell(x, cell)
            ax.plot(x, y, color=random.choice(['b', 'g', 'm', 'c']))
            
            y = uV.eval_cell(x, cell)
            ax.plot(x, y, color='r')

            y = u.eval_cell(x, cell)
            ax.plot(x, y, color='k')

        plt.show()

    # Error norm
    fine_degree = degree + 3
    element = HermiteElement(fine_degree)
    
    V_fine = FunctionSpace(mesh, element)
    # Interpolate exact solution to fine
    u_fine = V_fine.interpolate(u)
    # Interpolate approx solution fine
    uh_fine = V_fine.interpolate(uh)

    # Difference vector
    e = u_fine.vector - uh_fine.vector

    # Integrate the error
    A_fine = assemble_matrix(V_fine, 'bending', get_geom_tensor=None, timer=0)
    e = sqrt(np.sum(e*A_fine.dot(e)))
    # Mesh size
    hmin = mesh.hmin()

    # Add the cond number
    kappa = np.linalg.cond(A.toarray())

    return hmin, e, kappa, A.shape[0]


def get_problem(f):
    '''Compute solution of [1] from f.'''
    x = Symbol('x')
    # u4 = f
    u3 = integrate(f, x)
    u2 = integrate(u3, x)
    u1 = integrate(u2, x)
    u = integrate(u1, x) # + ax^3/6 + bx^/2 + cx + d

    mat = Matrix([[-1/6, 1/2, -1, 1],
                 [1/6, 1/2, 1, 1],
                 [1/2, -1, 1, 0],
                 [1/2, 1, 1, 0]])

    vec = Matrix([-u.subs(x, -1),
                 -u.subs(x, 1),
                 -u1.subs(x, -1),
                 -u1.subs(x, 1)])

    a, b, c, d = mat.LUsolve(vec)
    u += a*x**3/6 + b*x**2/2 + c*x + d

    # Check that it is the solution
    assert abs(u.evalf(subs={x: -1})) < 1E-15
    assert abs(u.evalf(subs={x: 1})) < 1E-15
    assert abs(u.diff(x, 1).evalf(subs={x: -1})) < 1E-15
    assert abs(u.diff(x, 1).evalf(subs={x: 1})) < 1E-15
    assert integrate((u.diff(x, 4) - f)**2, (x, -1, 1)) < 1E-15

    return u, f

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Compute the analytical solution
    x = Symbol('x')
    f = sin(2*pi*x)
    u, f = get_problem(f)
    print u, f
  
    # Test
    if True:
        degree = 3
        print '-'*79
        print degree
        print '-'*79

        iis =  [2**i for i in range(4, 10)]
        data = np.zeros((4, 1+len(iis))) # rows
        h0, e0, kappa0, N0 = solve(n_cells=8, degree=degree, with_plot=False)
        data[:, 0] = [N0, e0, np.nan, kappa0]

        for col, n_cells in enumerate(iis, 1):
            h, e, kappa, N = solve(n_cells=n_cells, degree=degree)
            # Rate
            r = log(e/e0)/log(h/h0)
            r_k = log(kappa0/kappa)/log(h/h0)

            msg0 = 'h = %.2E\te = %.4E\tr = (%.2f)' % (h, e, r)
            msg1 = 'kappa[%d] = %.4E(%.2f)' % (N, kappa, r_k)
            print colors['green'] % msg0, colors['red'] % msg1
            # Next
            h0, e0, kappa0 = h, e, kappa

            data[:, col] = [N, e, r, kappa]


    print data
    print ' & '.join(map(lambda x: '%d' % x, data[0, :])) + r'\\'
    print ' & '.join(map(lambda x: '%.2e' % x, data[1, :])) + r'\\'
    print ' & '.join(map(lambda x: '%.2f' % x, data[2, :])) + r'\\'
    print ' & '.join(map(lambda x: '%.2E' % x, data[3, :])) + r'\\'
