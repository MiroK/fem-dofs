from sympy import Expr, Symbol, lambdify
from mesh import Cells
import numpy as np


class GenericFunction(object):
    '''This is something that can be evaluated at point x.'''
    def eval(self, x):
        raise NotImplementedError('I am a template!')

    def eval_cell(self, x, cell):
        raise NotImplementedError('I am a template!')


class Function(GenericFunction):
    '''Function from function space.'''
    def __init__(self, V, values=None):
        self.function_space = V
        self.vector = np.zeros((V.dim))
        if values is not None:
            np.copyto(self.vector, values)

    def eval(self, x):
        '''
        Evaluate at points x. This might be slow because cell must be found
        first before evaluation.
        '''
        for cell in Cells(self.function_space.mesh):
            if cell.contains(x):
                return self.eval_cell(x, cell)
        # Yell if point not found
        raise ValueError('Point %s not in mesh!' % str(x))
   
    def eval_cell(self, x, cell):
        '''Evaluate at point inside cell.'''
        V = self.function_space
        # First need values of basis functions at x
        phi_x = V.element.eval_basis_all(x, cell)
        # Now I need to reach to the vector and get the dof_values
        coef = self.vector[V.dofmap.cell_dofs(cell.index)]
        # And do the dot product
        return coef.dot(phi_x)


    def interpolate(self, f):
        '''Interpolate f into the function.'''
        interpolant = self.function_space.interpolate(f)
        np.copyto(self.vector, interpolant.vector)


    def vertex_values(self):
        '''
        Values of those dofs that are associated with vertices sorted
        according to order of vertices in mesh. Note that the values are 
        extracted there is no eval call.
        '''
        return self.vector[self.function_space.vertex_to_dof_map]


class Expression(GenericFunction):
    '''Expression is a not a finite element function.'''
    def __init__(self, expr):
        assert isinstance(expr, Expr)
        x = Symbol('x')
        assert x in expr.atoms()
        self.expr = lambdify(x, expr, 'numpy')
    
    def eval(self, x):
        '''Eval at x.'''
        if isinstance(x, (float, int)): x = np.array([x])
        return self.expr(x)

    def eval_cell(self, x, cell):
        '''Eval at x.'''
        return self.eval(x)


class Constant(GenericFunction):
    '''It is what it is.'''
    def __init__(self, value):
        self.value = float(value)

    def eval(self, x):
        '''Eval at x.'''
        return np.array([self.value])

    def eval_cell(self, x, cell):
        '''Eval at x.'''
        return np.array([self.value])

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Test Expression
    from sympy import sin
    import numpy as np

    x = Symbol('x')
    f = sin(2*x)

    f = Expression(f)
    print f.eval(0.)
