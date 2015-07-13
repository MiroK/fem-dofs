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

    # Element
    degree = 1
    poly_set = leg.basis_functions(degree)
    dof_set = chebyshev_points(degree)
    element = LagrangeElement(poly_set, dof_set)

    # Mesh
    mesh = IntervalMesh(a=0, b=1, n_cells=100)

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
    A[-1, :] = 0; A[:, -1] = 0; A[-1, -1] = 1

    # Poisson with nullspace
    from sympy import sin, pi, Symbol
    x = Symbol('x')
    f = Expression(sin(pi*x))
    fV = V.interpolate(f)

    b = M.dot(fV.vector)
    # FIXME bcs
    b[0] = 0; b[-1] = 0
    
    from scipy.sparse.linalg import spsolve
    x = spsolve(A, b)

    u = Function(V, x)
    plot(u)
    show()

    # Add boundary conditions
    # Test that the solution is correct
    # How would the assembly modify for qudrature representation
