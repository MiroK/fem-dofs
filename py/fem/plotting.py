import matplotlib.pyplot as plt
import numpy as np


def plot(f, mesh=None, fig=None, **kwargs):
    '''Plot functions/expression in 1D.'''
    # Decide if function or expression
    try:
        V = f.function_space
        # Not really Scattered
        x = V.mesh.vertex_coordinates
        # Scattered in the same way
        y = f.vertex_values()
        isort = np.argsort(x[:, 0])
        x = x[isort]
        y = y[isort]
    # Expression
    except AttributeError:
        # Evaluate at mesh vertices
        x = np.sort(mesh.vertex_coordinates)
        y = f.eval(x)

    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, **kwargs)

    return fig

def show():
    '''Save the import.'''
    plt.show()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import Symbol, sin, S
    from function import Expression, Function
    from mesh import IntervalMesh
    from cg_space import FunctionSpace
    from lagrange_element import LagrangeElement
    import numpy as np

    x = Symbol('x')
    f = Expression(sin(2*x))
    meshf = IntervalMesh(a=-1, b=1, n_cells=4)
    # Just f
    fig = plot(f, mesh=meshf, label='f')
    # Two functions
    mesh = IntervalMesh(a=-1, b=1, n_cells=100)
    poly_set = [S(1), x, x**2, x**3]
    dof_set = np.array([-1, -0.5, 0.5, 1])
    element = LagrangeElement(poly_set, dof_set)
    V = FunctionSpace(mesh, element)
    # Here's one
    g = V.interpolate(f)
    # Fake the other
    gg = Function(V) 
    np.copyto(gg.vector, g.vector)
    gg.vector += 1
    # Add 
    fig = plot(g, fig=fig, label='g', color='r')
    fig = plot(gg, fig=fig, label='gg', color='g')

    plt.legend(loc='best')
    plt.show()
