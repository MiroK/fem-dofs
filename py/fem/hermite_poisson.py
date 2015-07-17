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
from sympy import sin, pi, Symbol, exp
from scipy.sparse.linalg import spsolve

from plotting import plot, show
import matplotlib.pyplot as plt
import random


def solve(n_cells, degree=3, with_plot=False):
    # Problem
    w = 3*np.pi
    x = Symbol('x')
    u = sin(w*x)
    f = -u.diff(x, 2)

    # As Expr
    u = Expression(u)
    f = Expression(f)

    # Space
    element = HermiteElement(degree)
    # poly_set = leg.basis_functions(degree)
    # dof_set = chebyshev_points(degree)
    # element = LagrangeElement(poly_set, dof_set)

    mesh = IntervalMesh(a=0, b=2, n_cells=n_cells)
    V = FunctionSpace(mesh, element)
    bc = DirichletBC(V, u)

    # Need mass matrix to intefrate the rhs
    Mget_geom_tensor = lambda cell: 1./cell.Jac
    M = assemble_matrix(V, 'mass', Mget_geom_tensor, timer=0)
    # NOTE We cannot you apply the alpha transform idea because the functions
    # are mapped with this selective weight on 2nd, 3rd functions. So some rows
    # of alpha would have to be multiplied by weights which are cell specific.
    # And then on top of this there would be a dx = J*dy term. Better just to
    # use the qudrature representations
    # Mpoly_matrix = leg.mass_matrix(degree)
    # M_ = assemble_matrix(V, Mpoly_matrix, Mget_geom_tensor, timer=0)
    
    # Stiffness matrix for Laplacian
    Aget_geom_tensor = lambda cell: cell.Jac
    A = assemble_matrix(V, 'stiff', Aget_geom_tensor, timer=0)
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

    # Error norm in CG high order
    fine_degree = degree + 3
    poly_set = leg.basis_functions(fine_degree)
    dof_set = chebyshev_points(fine_degree)
    element = LagrangeElement(poly_set, dof_set)
    
    V_fine = FunctionSpace(mesh, element)
    # Interpolate exact solution to fine
    u_fine = V_fine.interpolate(u)
    # Interpolate approx solution fine
    uh_fine = V_fine.interpolate(uh)

    # Difference vector
    e = u_fine.vector - uh_fine.vector

    # Need matrix for integration of H10 norm
    Apoly_matrix = leg.stiffness_matrix(fine_degree)
    A_fine = assemble_matrix(V_fine, Apoly_matrix, Aget_geom_tensor, timer=0)
    # Integrate the error
    e = sqrt(np.sum(e*A_fine.dot(e)))
    # Mesh size
    hmin = mesh.hmin()

    # Add the cond number
    kappa = np.linalg.cond(A.toarray())

    return hmin, e, kappa, A.shape[0]

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    degree = 3
    h0, e0, kappa0, N0 = solve(n_cells=8, degree=degree, with_plot=False)
    for n_cells in [2**i for i in range(4, 10)]:
        h, e, kappa, N = solve(n_cells=n_cells, degree=degree)
        # Rate
        r = log(e/e0)/log(h/h0)
        r_k = log(kappa0/kappa)/log(h/h0)

        msg0 = 'h = %.2E\te = %.4E\tr = (%.2f)' % (h, e, r)
        msg1 = 'kappa[%d] = %.4E(%.2f)' % (N, kappa, r_k)
        print colors['green'] % msg0, colors['red'] % msg1
        # Next
        h0, e0, kappa0 = h, e, kappa
