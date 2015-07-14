from __future__ import division
from scipy.sparse import lil_matrix
from numpy.polynomial.legendre import leggauss
from mesh import Cells
from utils import colors
import time


def assemble_matrix(V, poly_matrix, get_geom_tensor, timer=0):
    '''
    Reference element matrix is obtaind from polymatrix and a transformation 
    specific to an element. Element matrix are reference element matrix *
    geometric tensor specific to the element. 

    If polymatrix is array this is like a geometry representation except that 
    the reference element is computed differently. With string swith to
    quadrature tensor.
    '''
    t0 = time.time()
    if isinstance(poly_matrix, str):
        mat = quadrature_assamble_matrix(V, poly_matrix)
    else:
        mat = tensor_assamble_matrix(V, poly_matrix, get_geom_tensor)
    
    if timer:
        msg = '\tAssembled matrix %d x %d' % mat.shape
        msg = ' '.join([msg, 'in %g s' % (time.time()-t0)])
        print colors['red'] % msg
    return mat.tocsr()


def tensor_assamble_matrix(V, poly_matrix, get_geom_tensor):
    '''It is what it is.'''
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

    return A


def quadrature_assamble_matrix(V, poly_matrix):
    '''It is what it is.'''
    # Local stuff
    finite_element = V.element
    fe_dim = finite_element.dim 
    poly_degree = fe_dim - 1
    # Assume here that this is mass matrix 2*p is max needed p+1
    xq, wq = leggauss(poly_degree+1)
    quad_degree = len(xq)
    assert quad_degree == poly_degree + 1

    if poly_matrix == 'mass':
        def element_matrix(finite_element, cell):
            # Map the points to cell
            points = cell.map_from_reference(xq)
            K_matrix = finite_element.eval_basis_all(points, cell)
            # Remember dx
            weights = wq/cell.Jac
            K_matrix = (weights*K_matrix).dot(K_matrix.T)
            return K_matrix
    else:
        # Higher degree than necessary
        def element_matrix(finite_element, cell):
            # Map the points to cell
            points = cell.map_from_reference(xq)
            K_matrix = finite_element.eval_basis_derivative_all(1, points, cell)
            # Remember dx
            weights = wq/cell.Jac
            K_matrix = (weights*K_matrix).dot(K_matrix.T)
            return K_matrix

    # Global assembly
    size = V.dim
    mesh = V.mesh
    A = lil_matrix((size, size))
    dofmap = V.dofmap
    for cell in Cells(mesh):
        # Compute element matrix
        K_matrix = element_matrix(finite_element, cell)

        global_dofs = dofmap.cell_dofs(cell.index)
        for i, gi in enumerate(global_dofs):
            for j, gj in enumerate(global_dofs):
                A[gi, gj] += K_matrix[i, j]

    return A

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

    representation = 'tensor' # 'quadrature' # 'tensor'
    degree = 5
    msg = '>>>>>>>>>>>>>> DEGREE %d <<<<<<<<<<<<<<<' % degree
    print colors['blue'] % msg
    def solve(n_cells):
        # Element
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
        if representation == 'tensor':
            Mpoly_matrix = leg.mass_matrix(degree)
        else:
            Mpoly_matrix = 'mass'
        Mget_geom_tensor = lambda cell: 1./cell.Jac
        M = assemble_matrix(V, Mpoly_matrix, Mget_geom_tensor, timer=0)
        
        # M_ = assemble_matrix(V, 'mass', Mget_geom_tensor, timer=0)
        # assert np.linalg.norm(M.toarray() - M_.toarray()) < 1E-13

        # For stiffness multiply by Jac
        if representation == 'tensor':
            Apoly_matrix = leg.stiffness_matrix(degree)
        else:
            Apoly_matrix = 'stiff'
        Aget_geom_tensor = lambda cell: cell.Jac
        # Lhs of lin sys
        A = assemble_matrix(V, Apoly_matrix, Aget_geom_tensor, timer=1)
        
        # A_ = assemble_matrix(V, 'stiff', Mget_geom_tensor, timer=0)
        # assert np.linalg.norm(A.toarray() - A_.toarray()) < 1E-13
        
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
        if representation == 'tensor':
            Apoly_matrix = leg.stiffness_matrix(fine_degree)
        else:
            Apoly_matrix = 'stiff'
        A_fine = assemble_matrix(V_fine, Apoly_matrix, Aget_geom_tensor, timer=0)
        # Error
        e = sqrt(np.sum(e*A_fine.dot(e)))
        # Mesh size
        hmin = mesh.hmin()

        # Visualize
        # uh_fine.vector -= u_fine.vector
        # uh_fine.vector = np.abs(uh_fine.vector)
        # fig = plot(uh_fine, color='b')
        # fig = plot(A)
        # show()

        return hmin, e

    h0, e0 = solve(1)
    for n_cells in [2**i for i in range(4, 10)]:
        h, e = solve(n_cells)
        r = log(e/e0)/log(h/h0)

        msg = 'h = %.2E, e = %.4E r = (%.2f)' % (h, e, r)
        print colors['green'] % msg 

        h0, e0 = h, e

    # FIXME
    # 2) Test that the solution is correct - comuting l2, H1, L2 error without
    # supperconvergence. DG space? Mesh.size
    # 3) How would the assembly modify for qudrature representation
    # 4) Make interface for testing node position and matrix effects ...
    
    # 5) READ JOHN'S PAPER AND SET UP PLAN!!!
