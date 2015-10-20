from __future__ import division
from scipy.sparse import lil_matrix
from numpy.polynomial.legendre import leggauss
from utils import colors
from mesh import Cells
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

    elif poly_matrix == 'stiffness':
        # Higher degree than necessary
        def element_matrix(finite_element, cell):
            # Map the points to cell
            points = cell.map_from_reference(xq)
            K_matrix = finite_element.eval_basis_derivative_all(1, points, cell)
            # Remember dx
            weights = wq/cell.Jac
            K_matrix = (weights*K_matrix).dot(K_matrix.T)
            return K_matrix

    elif poly_matrix == 'bending':
        # Higher degree than necessary
        def element_matrix(finite_element, cell):
            # Map the points to cell
            points = cell.map_from_reference(xq)
            K_matrix = finite_element.eval_basis_derivative_all(2, points, cell)
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
