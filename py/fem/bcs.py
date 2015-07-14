from copy import deepcopy
import numpy as np

class DirichletBC(object):
    '''Boundary conditions to be applied to function space V.'''
    def __init__(self, V, g, gamma=None):
        '''Prescribe Dofs(g) of V at gamma.'''
        assert gamma is None, 'For now bcs everywhere'
        # Compute where to apply bcs
        # In general tdim-1 -> tdim connectivity
        mesh = V.mesh
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
        
        # Now get dofs at bdry facet, local!
        dofmap = V.dofmap
        bdry_dofs = [dofmap.tabulate_facet_dofs(facet)
                     for facet in bdry_facets]
        
        # Build this as a map dof -> value
        # The key is the global dof
        bdry_map = {dofmap.cell_dofs(cell.index)[dof]:
                    # The value is computed from local and cell
                    V.element.eval_dof(i=dof, f=g, cell=cell) 
                    for dofs, cell in zip(bdry_dofs, bdry_cells)
                    # Local to cell dofs of the cell
                    for dof in dofs}
        self.bdry_map = bdry_map
        # Remeber the space
        self.function_space = V

    def zero_rows(self, A):
        '''Zero the rows corresponding to boundary condition.'''
        assert A.shape[0] == self.function_space.dim

        # Get the data of matrix. Exprected a csr_matrix
        rows = A.indptr
        cols = A.indices
        vals = A.data

        # We could just as well set all the values for the row
        # to zero, but that would not be very efficient
        for bc_row in self.bdry_map.keys():
            first_col = rows[bc_row]
            last_col = rows[bc_row+1]
            row_nnz = last_col - first_col

            # Remove column indices of nnz entries of this row
            cols = np.r_[cols[:first_col], cols[last_col:]]
            # Remove nnz values of this row from all values
            vals = np.r_[vals[:first_col], vals[last_col:]]
            # Decrease the count of all subsequent rows
            rows[bc_row+1:] -= row_nnz

        # Set the new data
        A.indptr = rows
        A.indices = cols
        A.data = vals
        return 1

    def apply(self, A, b=None, symmetry=True):
        '''
        Apply boundary conditionts to matrix A, vector b possibly in a symmetric
        way.
        '''
        assert A.shape[0] == A.shape[1] and A.shape[0] == len(b)
        assert len(b) == self.function_space.dim

        if not symmetry:
            # dof index --> value to be prescribed to b
            # If we have b, set its values
            if b is not None:
                for bc_row, bc_value in self.bdry_map.iteritems():
                    b[bc_row] = bc_value
    
            # Get data of A
            rows = A.indptr
            cols = A.indices
            vals = A.data
            
            for bc_row in self.bdry_map.keys():
                first_col = rows[bc_row]
                last_col = rows[bc_row+1]
                row_nnz = last_col - first_col
                
                # Replace rows columsn with single entry for the row. This is typically
                # a diagonal
                cols = np.r_[cols[:first_col], np.array([bc_row], dtype='int32'), cols[last_col:]]
                # The value of the new entry is 1
                vals = np.r_[vals[:first_col], np.array([1.]), vals[last_col:]]
                # We have removed all but one nnz so shift the subsequent rows
                rows[bc_row+1:] -= row_nnz - 1

            # Set new data
            A.indptr = rows
            A.indices = cols
            A.data = vals
            return 1

        else:
            assert b is not None
            return self.zero_columns(A, b)


    def zero_columns(self, A, b, diag_val=1):
        '''
        Zero the columns of A that correspond to boundary conditions and modify b
        to reflect this changes.
        '''
        # This mirrors DirichletBC::zero_column
        n_rows, n_cols = A.shape
        # First learn what cols are boundary cols and record the value
        bc_cols = [False]*n_cols
        bc_values = np.zeros(n_cols)
        for dof, value in self.bdry_map.iteritems():
            bc_cols[dof] = True
            bc_values[dof] = value

        # Get data of A
        rows = A.indptr
        cols = A.indices
        vals = A.data

        for row in range(n_rows):
            # Diagonal blocks of mixed systems
            if diag_val and bc_cols[row]:
                # Modify so that diag_value*U[row]=diag_value*bc_values[row]
                b[row] = bc_values[row]*diag_val

                # Set the row to ...diag_val....
                first_col = rows[row]
                last_col = rows[row+1]
                row_nnz = last_col - first_col
                
                # Replace rows columsn with single entry for the row. This is typically
                # a diagonal
                cols = np.r_[cols[:first_col], np.array([row], dtype='int32'), cols[last_col:]]
                # The value of the new entry is 1
                vals = np.r_[vals[:first_col], np.array([diag_val]), vals[last_col:]]
                # We have removed all but one nnz so shift the subsequent rows
                rows[row+1:] -= row_nnz - 1
            
            # Off diagonal blocks
            else:
                # For column which is boundary one we need to zero that nonzero entry
                # and modify the right hande side
                first_col = rows[row]
                last_col = rows[row+1]
                row_nnz = last_col - first_col

                row_columns = cols[first_col:last_col]
                row_values = vals[first_col:last_col]
               
                j_not_removed = []
                for j, col in enumerate(row_columns):
                    # Zero or not boundary
                    if not bc_cols[col] or abs(row_values[j]) < 1E-15:
                        j_not_removed.append(j)
                        continue

                    # Reflect zeroeing of the column value on the right hand size
                    b[row] -= bc_values[col]*row_values[j]
                    # The column value is zeroed by throwing it out from columns
                    # and values
                
                # If there has been change in the row, modify the matrix and rhs
                if len(j_not_removed) != row_nnz:
                    # New row of A
                    # Zeroed columns not included
                    cols = np.r_[cols[:first_col], row_columns[j_not_removed], cols[last_col:]]
                    # Neither their values
                    vals = np.r_[vals[:first_col], row_values[j_not_removed], vals[last_col:]]
                    # Update how many rows removed
                    rows[row+1:] -= row_nnz - len(j_not_removed)

        # Set new data
        A.indptr = rows
        A.indices = cols
        A.data = vals
        return 1

 # ----------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from mesh import IntervalMesh
    from cg_space import FunctionSpace
    from function import Constant, Expression
    from lagrange_element import LagrangeElement
    from polynomials import legendre_basis as leg
    from points import chebyshev_points
    from sympy import Symbol
    from scipy.sparse import csr_matrix
    import numpy as np

    # Element
    degree = 2
    poly_set = leg.basis_functions(degree)
    dof_set = chebyshev_points(degree)
    element = LagrangeElement(poly_set, dof_set)

    # Mesh
    n_cells = 2
    mesh = IntervalMesh(a=-1, b=4, n_cells=n_cells)

    # Space
    V = FunctionSpace(mesh, element)

    x = Symbol('x')
    bc = DirichletBC(V, Expression(x))
    # Hope that the map is okay. Check if applied correctly
    
    # No symmetry
    A = csr_matrix(np.random.rand(V.dim, V.dim))
    A = A + A.T
    b = np.random.rand(V.dim)
    x = np.random.rand(len(b))
    bc.apply(A, b, False)
    for row, value in bc.bdry_map.items():
        assert abs(b[row] - value) < 1E-13
        assert abs(A.dot(x)[row] - x[row]) < 1E-13
    
    # Now symmetry
    A = csr_matrix(np.random.rand(V.dim, V.dim))
    A = A + A.T
    assert np.linalg.norm((A - A.T).toarray()) < 1E-13
    b = np.random.rand(V.dim)
    x = np.random.rand(len(b))
    bc.apply(A, b, True)
    for row, value in bc.bdry_map.items():
        assert abs(b[row] - value) < 1E-13
        assert abs(A.dot(x)[row] - x[row]) < 1E-13 
    assert np.linalg.norm((A - A.T).toarray()) < 1E-13
