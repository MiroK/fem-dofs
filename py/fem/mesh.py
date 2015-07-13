from __future__ import division
import numpy as np


class IntervalCell(object):
    '''1D interval finite element cell.'''
    def __init__(self, vertices, index=-1):
        '''1D interval in 1D Euclidean space.'''
        assert vertices.shape == (2, 1)
        assert vertices[0] < vertices[1]
        self.vertices = vertices.copy()
        self.index = index

    def contains(self, x, tol=1E-15):
        '''Is x \in [a, b].'''
        assert x.shape == (1, )
        v0, v1 = self.vertices[0], self.vertices[1]
        return np.all(v0 - tol < x < v1 + tol)

    def map_to_reference(self, x):
        '''
        Find point x_hat in [-1, 1](reference cell) such that F_k(x_hat) = x,
        with F_k the affine map.
        '''
        v0, v1 = self.vertices
        return (2*x - v0 - v1)/(v1 - v0)

    def map_from_reference(self, x_hat):
        '''
        Find point x = F_k(x_hat). Point x_hat is assemed to be in the
        reference cell [-1, 1].
        '''
        v0, v1 = self.vertices
        return 0.5*(v1 - v0)*x_hat + 0.5*(v0 + v1)

    @property
    def Jac(self):
        '''Jacobian of the inv(F_k): x\in this cell --> reference cell.'''
        # Note that theat is d(x_hat) = Jac*d(x)!
        v0, v1 = self.vertices
        return 2./(v1 - v0)

    @property
    def volume(self):
        '''Cell size'''
        return float(np.hypot(*self.vertices))


class ReferenceIntervalCell(IntervalCell):
    '''Cell where the 1D finite elements are defined.'''
    def __init__(self):
        IntervalCell.__init__(self, np.array([[-1.], [1.]]))


class IntervalMesh(object):
    '''Interval partitioned into interval cells.'''
    def __init__(self, **kwargs):
        '''Initialized from [a, b] and n_cells or list of vertices.'''
        # Vertices given
        if 'vertex_coordinates' in kwargs:
            vertex_coordinates = kwargs['vertex_coordinates']
            # Check that geoemetry dim is okay
            assert vertex_coordinates.shape[1] == 1
            # Check increasing sequences
            assert np.all((vertex_coordinates[0:-1] - vertex_coordinates[1:]) < 0)

            n_vertices = vertex_coordinates.shape[0]
            n_cells = n_vertices - 1

            # Cells from index vertices
            cells = [(i, i+1) for i in range(n_cells)]

            # Connectivities
            # cell/vertex with index is connected to cells/vertices with indices
            cell_vertex = cells
            # Reverse
            vertex_cell = [(0, )] +\
                          [(i-1, i) for i in range(1, n_vertices-1)] +\
                          [(n_cells-1, )]
            # Tdim - Tdim
            cell_cell = [(c-1, c, c+ 1) for c in range(1, n_cells-1)]
            cell_cell = [(0, 1)] + cell_cell + [(n_cells-2, n_cells-1)]

            # Let's remember stuff
            self.vertex_coordinates = vertex_coordinates
            self.n_vertices = n_vertices
            self.n_cells = n_cells

            # Cells will be used to create iterator
            self.cells = cells

            # This would be in topology?
            self.connectivity = {(1, 1): cell_cell,
                                 (1, 0): cell_vertex,
                                 (0, 1): vertex_cell}

        # Create vertices for [a, b] with n_cells
        elif 'a' in kwargs and 'b' in kwargs and 'n_cells' in kwargs:
            a, b, n_cells = kwargs['a'], kwargs['b'], kwargs['n_cells']
            vertex_coordinates = np.linspace(a, b, n_cells+1).reshape((-1, 1))
            IntervalMesh.__init__(self, vertex_coordinates=vertex_coordinates)

        # Anything else fails
        else:
            raise ValueError

    def hmin(self):
        '''Return smallest cell volume'''
        return min(cell.volume for cell in Cells(self)) 

    def cell(self, index):
        '''Get the cell of mesh.'''
        vertices = self.vertex_coordinates[self.connectivity[(1, 0)][index], :]
        return IntervalCell(vertices, index)


class Cells(object):
    '''Iterator over cells of mesh.'''
    def __init__(self, mesh):
        self.mesh = mesh

    def __iter__(self):
        mesh_vertices = self.mesh.vertex_coordinates
        # Connect cells -> vertices
        c2v = self.mesh.connectivity[(1, 0)]
         
        dummy = IntervalCell(np.array([[0.], [1.]]))
        for cell_index in range(self.mesh.n_cells):
            cell_vertices = mesh_vertices[c2v[cell_index], :]
            dummy.vertices = cell_vertices
            dummy.index = cell_index
            yield dummy

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    mesh = IntervalMesh(a=-2, b=1, n_cells=11)
    for cell in Cells(mesh):
        v0, v1 = cell.vertices
        assert np.allclose(v0, cell.map_from_reference(cell.map_to_reference(v0)))

    print mesh.cell(0).vertices

    cell = IntervalCell(np.array([[0.], [2.]]))
    assert int(cell.volume) == 2
    assert cell.contains(np.array([0.5]))
    assert not cell.contains(np.array([2.5]))

    xs = np.array([[0.], [0.5], [1.]])
    x_hats = cell.map_to_reference(xs)
    xs_ = cell.map_from_reference(x_hats)
    assert np.linalg.norm(xs - xs_) < 1E-14
