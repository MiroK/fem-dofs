from lagrange_element import LagrangeElement
from hermite_element import HermiteElement
from dofmap import CGDofMap, DGDofMap, HermiteDofMap
from function import Function
from mesh import Cells
import numpy as np


class FunctionSpace(object):
    '''Finite element function space ever mesh.'''
    def __init__(self, mesh, element, continuity='C0'):
        if isinstance(element, HermiteElement):
            self.dofmap = HermiteDofMap(mesh, element)

        elif isinstance(element, LagrangeElement):
            assert continuity
            if continuity == 'L2':
                self.dofmap = DGDofMap(mesh, element)
            else:
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
