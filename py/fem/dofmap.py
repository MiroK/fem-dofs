from collections import defaultdict
from lagrange_element import LagrangeElement
from hermite_element import HermiteElement
from mesh import IntervalMesh, Cells
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
        # Keesh for later computing
        self.mesh = mesh

    def cell_dofs(self, cell):
        '''Return local to global map of dofs of this cell.'''
        return self.dofmap[cell]
        
    def tabulate_facet_dofs(self, facet):
        '''
        A map from vertex to index such that cell_dofs[index] are the dofs at 
        facet.
        '''
        f2c = self.mesh.connectivity[(0, 1)]
        # Grab the first cell connected to facet
        cell = f2c[facet][0]
        dofs = self.cell_dofs(cell)
        # Find the index of gloval. This is not the right way...
        return [dofs.index(self._vertex_to_dofmap[facet])]


class DGDofMap(object):
    '''
    This class knows how to map cell(index) to its dofs(global numbering).
    The space if Discontinuous Lagrange
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
        
        global_dof = 0
        seen_global_dofs = {}
        for cell, vs in enumerate(c2v):
            # Local interor dofs are always 0, 1
            for loc, v in enumerate(vs):
                local_to_global[cell].append(global_dof)
                # Collects vertex -> 2 dofs
                seen_global_dofs[v] = global_dof
                global_dof += 1
            # Now go over the interior guys
            for loc in range(2, dim):
                local_to_global[cell].append(global_dof)
                global_dof += 1

        # Remember
        self.dofmap = local_to_global
        # Nice by product of this is vertex_to_dofmap. This is not 1-1. Just
        # keep it here for later ...
        self._vertex_to_dofmap = [seen_global_dofs[vertex]
                                  for vertex in sorted(seen_global_dofs.keys())]
        # Keesh for later computing
        self.mesh = mesh

    def cell_dofs(self, cell):
        '''Return local to global map of dofs of this cell.'''
        return self.dofmap[cell]
        
    def tabulate_facet_dofs(self, facet):
        '''
        A map from vertex to index such that cell_dofs[index] are the dofs at 
        facet.
        '''
        # This is not well defined?
        return []


class HermiteDofMap(object):
    '''
    Glue the Hermite elements together to ensure C^1 continuity.
    This class knows how to map cell(index) to its dofs(global numbering).
    '''
    def __init__(self, mesh, element):
        '''Build the dofmap for mesh of elements.'''
        assert isinstance(mesh, IntervalMesh)
        assert isinstance(element, HermiteElement)

        c2v = mesh.connectivity[(1, 0)]
        n_cells = len(c2v)
        dim = element.dim
        # This map takes the cell and returs the list of its global dofs 
        local_to_global = defaultdict(list)
        global_dof = 0
        seen_global_dofs0 = {}   # Values 
        seen_global_dofs1 = {}   # Derivatives
        for cell, vs in enumerate(c2v):
            # Local interor dofs for values are always 0, 1
            for loc, v in enumerate(vs):
                # If the the vertex was seen reuse the dof
                if v in seen_global_dofs0:
                    local_to_global[cell].append(seen_global_dofs0[v])
                # Otherwise the dof gets new tag - global dof
                else:
                    local_to_global[cell].append(global_dof)
                    # That tagged is also remembered for future vertices
                    seen_global_dofs0[v] = global_dof
                    # Increse
                    global_dof += 1

            # Local interor dofs for derivate values are always 2, 3
            for loc, v in enumerate(vs, 2):
                # If the the vertex was seen reuse the dof
                if v in seen_global_dofs1:
                    local_to_global[cell].append(seen_global_dofs1[v])
                # Otherwise the dof gets new tag - global dof
                else:
                    local_to_global[cell].append(global_dof)
                    # That tagged is also remembered for future vertices
                    seen_global_dofs1[v] = global_dof
                    # Increse
                    global_dof += 1
                
            # Now go over the interior guys if there are some
            for loc in range(4, dim):
                local_to_global[cell].append(global_dof)
                global_dof += 1

        # Remember
        self.dofmap = local_to_global
        # Nice by product of this is vertex_to_dofmap for value @ functionals
        self._vertex_to_dofmap0 = [seen_global_dofs0[vertex]
                                   for vertex in sorted(seen_global_dofs0.keys())]
        # And derivatives
        self._vertex_to_dofmap1 = [seen_global_dofs1[vertex]
                                   for vertex in sorted(seen_global_dofs1.keys())]
        # Keesh for later computing
        self.mesh = mesh

    def cell_dofs(self, cell):
        '''Return local to global map of dofs of this cell.'''
        return self.dofmap[cell]
        
    def tabulate_facet_dofs(self, facet):
        '''
        A map from vertex to index such that cell_dofs[index] are the dofs at 
        facet.
        '''
        f2c = self.mesh.connectivity[(0, 1)]
        # Grab the first cell connected to facet
        cell = f2c[facet][0]
        dofs = self.cell_dofs(cell)
        # Find the index of global. This is not the right way...
        # HACK to set bcs in Poisson. I only need @ value functionals
        # return [dofs.index(self._vertex_to_dofmap0[facet])]

        return [dofs.index(self._vertex_to_dofmap0[facet])] + \
               [dofs.index(self._vertex_to_dofmap1[facet])]
