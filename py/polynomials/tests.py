from sympy import lambdify, Symbol
from sympy.mpmath import quad


x = Symbol('x')
_TOL = 1E-13


def test_matrix(basis, mat, form):
    '''Is mat the matrix of form in the basis?'''
    for i, v in enumerate(basis):
        for j, u in enumerate(basis):
            val = form(u, v)
            val_ = mat[i, j]
            if abs(val - val_) > _TOL:
                print '\t mismatch @ [%d, %d] %g' % (i, j, abs(val - val_))
                return False

    return True


def test_M(basis, M):
    '''Test mass matrix of the basis.'''
    # The linear form is the L^2 inner product over (-1, 1)
    a_form = lambda u, v: quad(lambdify(x, u*v), [-1, 1])
    return test_matrix(basis, M, a_form)


def test_A(basis, A):
    '''Test stiffness matrix of the basis.'''
    # The linear form is the H^1_0 inner product over (-1, 1)
    a_form = lambda u, v: quad(lambdify(x, u.diff(x, 1)*v.diff(x, 1)), [-1, 1])
    return test_matrix(basis, A, a_form)


def test_C(basis, C):
    '''Test gradient matrix of the basis.'''
    # The linear form is the Trial`*Test integrated over (-1, 1)
    a_form = lambda u, v: quad(lambdify(x, u.diff(x, 1)*v), [-1, 1])
    return test_matrix(basis, C, a_form)
