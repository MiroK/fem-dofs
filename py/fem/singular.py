from __future__ import division
import sys
sys.path.append('../')
from mesh import IntervalMesh, Cells
from function_space import FunctionSpace
from function import Function, Expression, Constant
from hermite_element import HermiteElement
from assembler import assemble_matrix

from lagrange_element import LagrangeElement
from polynomials import legendre_basis as leg
from points import chebyshev_points, equidistant_points
from scipy.linalg import eigvalsh, eigh, eig
from scipy.sparse import lil_matrix
import numpy as np


def is_symmetric(mat):
    n, m = mat.shape
    if n != m:
        return False
    else:
        x = np.random.rand(n)
        return np.linalg.norm(mat.dot(x)-mat.__rmul__(x))/n < 1E-13

def get_G(V):
    # Global assembly
    mesh = V.mesh
    finite_element = V.element

    facet_to_cell = mesh.connectivity[(0, 1)]
    # Get the cells on the boundary and their facets
    bdry_cells, bdry_facets = [], []
    for facet_index, cells_of_facet in enumerate(facet_to_cell):
        # Bdry facet connected to only one cell
        if len(cells_of_facet) == 1:
            cell_index = cells_of_facet[0]
            bdry_facets.append(facet_index)
            # Physical call needs to be constructed here
            cell = mesh.cell(cell_index)
            bdry_cells.append(cell)

    size = V.dim
    A = lil_matrix((size, size))

    dofmap = V.dofmap
    for cell, xq in zip(bdry_cells, (-1, 1)):
        # Get element matrix
        points = cell.map_from_reference(xq)
        v_mat = finite_element.eval_basis_all(points, cell)
        g_mat = finite_element.eval_basis_derivative_all(1, points, cell).T
        K_matrix = (v_mat*xq).dot(g_mat)  # Also use as sign
        # To Global
        global_dofs = dofmap.cell_dofs(cell.index)
        for i, gi in enumerate(global_dofs):
            for j, gj in enumerate(global_dofs):
                A[gi, gj] += K_matrix[i, j]

    
    return A


def solve(n_cells, degree=3, with_plot=False):
    # Space
    # element = HermiteElement(degree)
    poly_set = leg.basis_functions(degree)
    dof_set = equidistant_points(degree)
    element = LagrangeElement(poly_set, dof_set)

    a, b = -1., 1.
    mesh = IntervalMesh(a=a, b=b, n_cells=n_cells)
    V = FunctionSpace(mesh, element)

    # Need mass matrix to intefrate the rhs
    Mget_geom_tensor = lambda cell: 1./cell.Jac
    M = assemble_matrix(V, 'mass', Mget_geom_tensor, timer=0)
    
    # Stiffness matrix for Laplacian
    Aget_geom_tensor = lambda cell: cell.Jac
    A = assemble_matrix(V, 'stiff', Aget_geom_tensor, timer=0)

    print 'M sym', is_symmetric(M)
    print 'A sym', is_symmetric(A)

    ew, ev = eigh(A.toarray(), M.toarray())
    ew = np.abs(ew)

    # Now add inner(dot(grad(u), n), v)*ds
    G = get_G(V)
    # A = A - G

    # print 'A sym', is_symmetric(A)
    # ew, ev = eig(A.toarray())#, M.toarray())
    # ew = np.sort(np.abs(ew))[:3]
    # print ew

    # from sympy import symbols, sqrt, S

    # x = symbols('x')
    # nullspace = map(Expression, [S(1/2.), sqrt(3/2.)*x])

    # Z = [V.interpolate(f).vector for f in nullspace]

    # print [np.inner(pi_z, A.dot(pi_z)) for pi_z in Z]

    print 'xxx'
    print M.toarray()
    print
    print A.toarray()
    print
    print G.toarray()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
   
    degree = 4
    solve(n_cells=1, degree=degree, with_plot=False)
