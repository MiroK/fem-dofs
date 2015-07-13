from copy import deepcopy

class DirichletBC(object):
    '''Boundary conditions to be applied to function space V.'''
    def __init__(self, V, g, gamma=None):
        '''Prescribe Dofs(g) of V at gamma.'''
        assert gamma is None, 'For now bcs everywhere'

        # Compute where to apply bcs
        # In general tdim-1 -> tdim connectivity
        facet_to_cell = V.mesh.connectivity[(0, 1)]
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
        
        # Now get dofs at bdry facet
        # This in general is tabulate_facet_entities
        # FIXME: this is global need local
        print V.dim
        print bdry_facets
        bdry_dofs = [V.vertex_to_dof_map[facet] for facet in bdry_facets]
        print bdry_dofs
        print bdry_cells
        
        # Evaluate dofs to get the boundary value
        # Build this as a map dof -> value
        # Construct cell to get the eval right
        bdry_map = {dof: V.element.eval_dof(i=dof, f=g, cell=cell) 
                    for dof, cell in zip(bdry_dofs, bdry_cells)}

        print bdry_map


    def apply(A, b=None, symmetry=True):
        '''
        Apply boundary conditionts to matrix A, vector b possibly in a symmetric
        way.
        '''
        pass

 # ----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from mesh import IntervalMesh
    from cg_space import FunctionSpace
    from function import Constant
    from lagrange_element import LagrangeElement
    from polynomials import legendre_basis as leg
    from points import chebyshev_points

    # Element
    degree = 3
    poly_set = leg.basis_functions(degree)
    dof_set = chebyshev_points(degree)
    element = LagrangeElement(poly_set, dof_set)

    # Mesh
    n_cells = 10
    mesh = IntervalMesh(a=0, b=1, n_cells=n_cells)

    # Space
    V = FunctionSpace(mesh, element)

    bc = DirichletBC(V, Constant(2))
