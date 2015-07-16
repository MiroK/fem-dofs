from sympy import integrate, S, Symbol, sin
from function import Expression, Function
from mesh import IntervalMesh, Cells
from dofmap import CGDofMap, DGDofMap, HermiteDofMap
from function_space import FunctionSpace
from lagrange_element import LagrangeElement
from hermite_element import HermiteElement
import numpy as np

# CG ---------
# Element
x = Symbol('x')
poly_set = [S(1), x, x**2, x**3]
dof_set = np.array([-1, -0.5, 0.5, 1])
element = LagrangeElement(poly_set, dof_set)
# Mesh
mesh = IntervalMesh(a=-1, b=1, n_cells=2)
# Dofmap
dofmap = CGDofMap(mesh, element)
assert dofmap.dofmap[0] == [0, 1, 2, 3]
assert dofmap.dofmap[1] == [1, 4, 5, 6]
# Space
V = FunctionSpace(mesh, element)
assert V.dim == 7
assert dofmap.tabulate_facet_dofs(1) == [1]
# Interpolation
mesh = IntervalMesh(a=-1, b=1, n_cells=10)
V = FunctionSpace(mesh, element)
# Function: expr
x = Symbol('x')
f = Expression(x)
f = V.interpolate(f)
# As pure function
g = V.interpolate(f)
assert np.linalg.norm(f.vector - g.vector) < 1E-13

# DG ------
# Dofmap
mesh = IntervalMesh(a=-1, b=1, n_cells=2)
dofmap = DGDofMap(mesh, element)
assert dofmap.dofmap[0] == [0, 1, 2, 3]
assert dofmap.dofmap[1] == [4, 5, 6, 7]

# Space
V = FunctionSpace(mesh, element, continuity='L2')
assert V.dim == 8
assert dofmap.tabulate_facet_dofs(1) == []

mesh = IntervalMesh(a=-1, b=1, n_cells=10)
V = FunctionSpace(mesh, element, continuity='L2')
# Function: expr
x = Symbol('x')
f = Expression(x)
f = V.interpolate(f)
# As pure function
g = V.interpolate(f)
assert np.linalg.norm(f.vector - g.vector) < 1E-13

# Hermite
# Dofmap
element = HermiteElement(3)
mesh = IntervalMesh(a=-1, b=1, n_cells=3)
dofmap = HermiteDofMap(mesh, element)
assert dofmap.dofmap[0] == [0, 1, 2, 3]
assert dofmap.dofmap[1] == [1, 4, 3, 5]
 
# Space
V = FunctionSpace(mesh, element)
assert V.dim == 8
assert dofmap.tabulate_facet_dofs(3) == [1]

mesh = IntervalMesh(a=-1, b=1, n_cells=10)
V = FunctionSpace(mesh, element)
# Function: expr
x = Symbol('x')
f = Expression(x**2)
f = V.interpolate(f)
# As pure function
g = V.interpolate(f)

assert np.linalg.norm(f.vector - g.vector, np.inf) < 1E-13
