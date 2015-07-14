import sys
sys.path.append('../')
from mesh import IntervalMesh
from cg_space import FunctionSpace
from function import Function, Expression, Constant
from lagrange_element import LagrangeElement
from polynomials import legendre_basis as leg
from points import chebyshev_points, equidistant_points, gauss_legendre_points
from assembler import assemble_matrix
from plotting import plot, show
from bcs import DirichletBC
from utils import colors

from math import log, sqrt
import numpy as np
from sympy import sin, pi, Symbol, exp
from scipy.sparse.linalg import spsolve


def solve(mode, points, degree, solution, w):
    '''
    TODO
    '''
    # Pick points
    get_points = {'eq': equidistant_points,
                  'cheb': chebyshev_points,
                  'gauss': gauss_legendre_points}[points]
    
    # Analytical solution and rhs
    x = Symbol('x')
    if solution == 'sine':
        u = sin(w*x)
    elif solution == 'delta':
        u = exp(-x**2/2/w)/sqrt(2*pi*w)
    f = -u.diff(x, 2)

    # As Expr
    u = Expression(u)
    f = Expression(f)

    # Now run the loop
    data = []
    if mode == 'convergence':
        h0, e0 = _solve(mode, get_points, degree, n_cells=8, u=u, f=f)
        for n_cells in [2**i for i in range(4, 10)]:
            h, e = _solve(mode, get_points, degree, n_cells, u, f)
            # Rate
            r = log(e/e0)/log(h/h0)

            msg = 'h = %.2E\te = %.4E\tr = (%.2f)' % (h, e, r)
            print colors['green'] % msg 
            data.append([h, e, r])
            # Next
            h0, e0 = h, e

    else:
        for n_cells in [2**i for i in range(4, 10)]:
            h, A = _solve(mode, get_points, degree, n_cells, u, f)
            kappaA = np.linalg.cond(A.toarray())
            N = A.shape[0]

            msg = '\t Computing condition number of %d x %d matrix' % A.shape
            print colors['red'] % msg
            msg = 'h = %.2E\tN = %g\tkappa = %.4E' % (h, N, kappaA)
            print colors['green'] % msg 
            data.append([h, N, kappaA])
        
    data = np.array(data)
    header = msg.split('\t')
    header = [w.split('=')[0] for w in header]
    header = '\t'.join(header)

    return data, header


def _solve(mode, points, degree, n_cells, u, f):
    '''
    In mode == convergence:
    Solve -u`` = f with dirichet bcs bdry of (-1, 1) given by exact solution.
    The Vh space is CG_space of degree elements and n_cells. Return hmin, error
    for convergence computation.

    In mode == cond:
    Just return h and the matrix A.
    '''
    # Element. The polynomial space is spanned by Legendre basis
    poly_set = leg.basis_functions(degree)
    dof_set = points(degree)
    element = LagrangeElement(poly_set, dof_set)

    # Mesh
    mesh = IntervalMesh(a=-1, b=1, n_cells=n_cells)

    # Space
    V = FunctionSpace(mesh, element)
    bc = DirichletBC(V, u)

    # Need mass matrix to intefrate the rhs
    Mpoly_matrix = leg.mass_matrix(degree)
    Mget_geom_tensor = lambda cell: 1./cell.Jac
    M = assemble_matrix(V, Mpoly_matrix, Mget_geom_tensor, timer=0)
    
    # Stiffness matrix for Laplacian
    Apoly_matrix = leg.stiffness_matrix(degree)
    Aget_geom_tensor = lambda cell: cell.Jac
    A = assemble_matrix(V, Apoly_matrix, Aget_geom_tensor, timer=0)
   
    # Interpolant of source
    fV = V.interpolate(f)
    # Integrate in L2 to get the vector
    b = M.dot(fV.vector)
    
    # Apply boundary conditions
    bc.apply(A, b, True)
    x = spsolve(A, b)

    if mode == 'condition':
        return mesh.hmin(), A

    # As function
    uh = Function(V, x)
   
    # Error norm
    # Higher order DG element
    fine_degree = degree + 3
    poly_set = leg.basis_functions(fine_degree)
    dof_set = chebyshev_points(fine_degree)
    element = LagrangeElement(poly_set, dof_set)
    # THe space
    V_fine = FunctionSpace(mesh, element, 'L2')
    # Interpolate exact solution to fine
    u_fine = V_fine.interpolate(u)
    # Interpolate approx solution fine
    uh_fine = V_fine.interpolate(uh)

    # Difference vector
    e = u_fine.vector - uh_fine.vector
    # Need matrix for integration of H10 norm
    Apoly_matrix = leg.stiffness_matrix(fine_degree)
    A_fine = assemble_matrix(V_fine, Apoly_matrix, Aget_geom_tensor, timer=1)
    # Integrate the error
    e = sqrt(np.sum(e*A_fine.dot(e)))
    # Mesh size
    hmin = mesh.hmin()

    return hmin, e
