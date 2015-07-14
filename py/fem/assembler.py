from __future__ import division
from scipy.sparse import lil_matrix
from mesh import Cells
import time


colors = {'blue': '\033[1;37;34m%s\033[0m',
          'red': '\033[1;37;31m%s\033[0m',
          'green': '\033[1;37;32m%s\033[0m',
          'cyan':  '\033[1;37;36m%s\033[0m',
          'purple':  '\033[1;37;35m%s\033[0m',
          'orange': '\033[1;33;40m%s\033[0m',}


def assemble_matrix(V, poly_matrix, get_geom_tensor, timer=0):
    '''
    Reference element matrix is obtaind from polymatrix and a transformation 
    specific to an element. Element matrix are reference element matrix *
    geometric tensor specific to the element. 
    '''
    t0 = time.time()
    # Local
    r_matrix = poly_matrix
    alpha = V.element.alpha
    r_matrix = alpha.dot(r_matrix).dot(alpha.T)

    # Now global
    size = V.dim
    mesh = V.mesh
    A = lil_matrix((size, size))
    
    dofmap = V.dofmap
    for cell in Cells(mesh):
        global_dofs = dofmap.cell_dofs(cell.index)
        G_matrix = get_geom_tensor(cell)
        K_matrix = r_matrix*G_matrix
        for i, gi in enumerate(global_dofs):
            for j, gj in enumerate(global_dofs):
                A[gi, gj] += K_matrix[i, j]

    if timer:
        msg = '\tAssembled matrix %d x %d in %g s' % (size, size, time.time()- t0)
        print colors['red'] % msg
    return A.tocsr()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from mesh import IntervalMesh
    from cg_space import FunctionSpace
    from function import Function, Expression, Constant
    from lagrange_element import LagrangeElement
    from polynomials import legendre_basis as leg
    from points import chebyshev_points
    from plotting import plot, show
    from bcs import DirichletBC
    from math import log, sqrt
    import numpy as np
    import sys
    from sympy import sin, pi, Symbol, exp
    from scipy.sparse.linalg import spsolve

    def solve(n_cells):
        # Element
        degree = 7
        poly_set = leg.basis_functions(degree)
        dof_set = chebyshev_points(degree)
        element = LagrangeElement(poly_set, dof_set)

        # Mesh
        mesh = IntervalMesh(a=0, b=1, n_cells=n_cells)

        # Space
        V = FunctionSpace(mesh, element)
        bc = DirichletBC(V, Constant(0))

        # Matrix spec: for mass matrix divide by cell's Jacobian to get geometric
        # tensor
        Mpoly_matrix = leg.mass_matrix(degree)
        Mget_geom_tensor = lambda cell: 1./cell.Jac
        M = assemble_matrix(V, Mpoly_matrix, Mget_geom_tensor, timer=0)
        # For stiffness multiply by Jac
        Apoly_matrix = leg.stiffness_matrix(degree)
        Aget_geom_tensor = lambda cell: cell.Jac
        # Lhs of lin sys
        A = assemble_matrix(V, Apoly_matrix, Aget_geom_tensor, timer=0)
        
        # Poisson 
        x = Symbol('x')
        w = 8*pi
        u = sin(w*x)*sin(w*exp(x))
        f = -u.diff(x, 2)

        f = Expression(f)
        u = Expression(u)

        fV = V.interpolate(f)

        # Rhs of lin sys
        b = M.dot(fV.vector)
        # bcs
        bc.apply(A, b, True)
        x = spsolve(A, b)
        # As function
        uh = Function(V, x)
       
        # Error norm
        # Higher order space
        fine_degree = degree + 3
        poly_set = leg.basis_functions(fine_degree)
        dof_set = chebyshev_points(fine_degree)
        element = LagrangeElement(poly_set, dof_set)
        V_fine = FunctionSpace(mesh, element)
        # Interpolate exact solution to fine
        u_fine = V_fine.interpolate(u)
        # Interpolate approx solution fine
        uh_fine = V_fine.interpolate(uh)

        # Now make error (vector) in V_fine
        e = u_fine.vector - uh_fine.vector
        # Matrices for integration of H10 norm
        # And H10 norm
        Apoly_matrix = leg.stiffness_matrix(fine_degree)
        A_fine = assemble_matrix(V_fine, Apoly_matrix, Aget_geom_tensor, timer=0)
        # Error
        e = sqrt(np.sum(e*A_fine.dot(e)))
        # Mesh size
        hmin = mesh.hmin()

        # Visualize
        # uh_fine.vector -= u_fine.vector
        # uh_fine.vector = np.abs(uh_fine.vector)
        # fig = plot(uh_fine, color='b')
        # show()

        return hmin, e

    h0, e0 = solve(8)
    for n_cells in [2**i for i in range(4, 10)]:
        h, e = solve(n_cells)
        r = log(e/e0)/log(h/h0)

        msg = 'h = %.2E, e = %.4E r = (%.2f)' % (h, e, r)
        print colors['green'] % msg 

        h0, e0 = h, e

    # FIXME
    # 1) Add boundary conditions <- tabulate_facets?
    # 2) Test that the solution is correct - comuting l2, H1, L2 error without
    # supperconvergence. DG space? Mesh.size
    # 3) How would the assembly modify for qudrature representation
    # 4) Make interface for testing node position and matrix effects ...
    
    # 5) READ JOHN'S PAPER AND SET UP PLAN!!!
