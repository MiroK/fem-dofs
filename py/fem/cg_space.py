from collections import defaultdict
from lagrange_element import LagrangeElement
from mesh import IntervalMesh, Cells
from function import Function
import numpy as np

class CGDofMap(object):
    '''
    Glue the Lagrange elements together to ensure continuity.
    This class knows how to map cell(index) to its dofs(global numbering).
    '''
    def __init__(self, mesh, element):
        '''Build the dofmap for mesh of elements.'''
        assert isinstance(mesh, IntervalMesh)
        assert isinstance(element, LagrangeElement)

        c2v = mesh.connectivity[(1, 0)]
        n_cells = len(c2v)
        dim = element.dim
        # This map takes the cell and returs the list of its global dofs 
        local_to_global = defaultdict(list)
        # The build idea is based on local dof ordering at reference cell (-1, 1).
        # The local dof ordering is 0 at -1, 1 at 1 and then there are interior dofs
        # In exterior dofs are the only thing shared between the cell and it can be
        # checked whether that dof was given a label be seeing if its vertex has
        # been seen
        global_dof = 0
        seen_global_dofs = {}
        for cell, vs in enumerate(c2v):
            # Local interor dofs are always 0, 1
            for loc, v in enumerate(vs):
                # If the the vertex was seen reuse the dof
                if v in seen_global_dofs:
                    local_to_global[cell].append(seen_global_dofs[v])
                # Otherwise the dof gets new tag - global dof
                else:
                    local_to_global[cell].append(global_dof)
                    # That tagged is also remembered for future vertices
                    seen_global_dofs[v] = global_dof
                    # Increse
                    global_dof += 1
            # Now go over the interior guys
            for loc in range(2, dim):
                local_to_global[cell].append(global_dof)
                global_dof += 1

        # Remember
        self.dofmap = local_to_global
        # Nice by product of this is vertex_to_dofmap
        self._vertex_to_dofmap = [seen_global_dofs[vertex]
                                  for vertex in sorted(seen_global_dofs.keys())]

    def cell_dofs(self, cell):
        '''Return local to global map of dofs of this cell.'''
        return self.dofmap[cell]


class FunctionSpace(object):
    '''Finite element function space ever mesh.'''
    def __init__(self, mesh, element):
        self.dofmap = CGDofMap(mesh, element)
        self.element = element
        self.mesh = mesh
        self.dim = len(set(sum((dofs for dofs in self.dofmap.dofmap.values()), [])))

    def interpolate(self, f):
        '''Interpolate f to V creating a new function.'''
        vector = np.zeros(self.dim)
        # Fill the vector L_global(f)
        for cell in Cells(self.mesh):
            global_dofs = self.dofmap.cell_dofs(cell.index)
            dof_values = self.element.eval_dofs(f, cell)
            vector[global_dofs] = dof_values
        # Now make the function
        return Function(self, vector)

    @property
    def vertex_to_dof_map(self):
        '''Item i of this map is the dof index of dof at vertex i.'''
        return self.dofmap._vertex_to_dofmap
