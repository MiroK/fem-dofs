from __future__ import division
from scipy.sparse import lil_matrix
from mesh import Cells
import time

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
        print 'Assembled matrix %d x %d in %g s' % (size, size, time.time()- t0)
    return A.tocsr()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from mesh import IntervalMesh
    from cg_space import FunctionSpace
    from function import Function, Expression
    from lagrange_element import LagrangeElement
    from polynomials import legendre_basis as leg
    from points import chebyshev_points
    from plotting import plot, show
    from math import log
    import numpy as np
    import sys

    def solve(n_cells):
        # Element
        degree = 2
        poly_set = leg.basis_functions(degree)
        dof_set = chebyshev_points(degree)
        element = LagrangeElement(poly_set, dof_set)

        # Mesh
        mesh = IntervalMesh(a=0, b=1, n_cells=n_cells)

        # Space
        V = FunctionSpace(mesh, element)

        # Matrix spec: for mass matrix divide by cell's Jacobian to get geometric
        # tensor
        poly_matrix = leg.mass_matrix(degree)
        get_geom_tensor = lambda cell: 1./cell.Jac
        M = assemble_matrix(V, poly_matrix, get_geom_tensor, timer=1)
        # For stiffness multiply by Jac
        poly_matrix = leg.stiffness_matrix(degree)
        get_geom_tensor = lambda cell: cell.Jac
        A = assemble_matrix(V, poly_matrix, get_geom_tensor, timer=1)
        
        # FIXME bcs
        A = A.toarray()
        A[0, :] = 0; A[:, 0] = 0; A[0, 0] = 1
        A[-2, :] = 0; A[:, -2] = 0; A[-2, -2] = 1

        # Poisson 
        from sympy import sin, pi, Symbol

        x = Symbol('x')
        # Rhs
        w = 5*pi
        f = Expression(sin(w*x))
        # Exact solution
        u = Expression(sin(w*x)/(w**2))

        fV = V.interpolate(f)

        b = M.dot(fV.vector)
        # FIXME bcs
        b[0] = 0; b[-2] = 0
        
        from scipy.sparse.linalg import spsolve
        x = spsolve(A, b)

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

        # Error l^2. We can do L^2 and H^1 with matrices
        el2 = np.sqrt(np.linalg.norm(u_fine.vector - uh_fine.vector))/V_fine.dim
        N = V.dim
   
        # Visualize
        uh_fine.vector -= u_fine.vector
        uh_fine.vector = np.abs(uh_fine.vector)
        fig = plot(uh_fine, color='b')
        show()

        return N, el2

    N0, e0 = solve(8)
    for n_cells in [2**i for i in range(4, 10)]:
        N, e = solve(n_cells)
        r = log(e/e0)/log(N0/N)
        
        print 'N=%d, e=%g, r=%.2f' % (N, e, r)

        N0, e0 = N, e

    # FIXME
    # 1) Add boundary conditions <- tabulate_facets?
    # 2) Test that the solution is correct - comuting l2, H1, L2 error without
    # supperconvergence. DG space? Mesh.size
    # 3) How would the assembly modify for qudrature representation
    # 4) Make interface for testing node position and matrix effects ...
    
    # 5) READ JOHN'S PAPER AND SET UP PLAN!!!
