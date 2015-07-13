from __future__ import division
from sympy import Symbol
from scipy.sparse import diags
import numpy as np


def basis_functions(deg):
    '''Polynomial basis of degree P^deg[-1, 1] of monomials.'''
    return [Symbol('x')**k for k in range(deg+1)]


def mass_matrix(deg):
    '''Mass matrix of monomials of degree up to deg.'''
    n = deg+1
    return np.array([[(1-(-1.)**(i+j+1))/(i+j+1) for j in range(n)]
                      for i in range(n)])


def stiffness_matrix(deg):
    '''Stiffness matrix of monomials of degree up to deg.'''
    n = deg+1
    mat = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            mat[i, j] = (1-(-1.)**(i+j-1))*i*j/(i+j-1)
    return mat


def derivative_matrix(deg):
    '''Matrix of \int_{-1}^{1}(P_i, P_j`) of monomials of degree up to deg.'''
    n = deg + 1
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(1, n):
            mat[i, j] = (1 - (-1.)**(i + j))*j/(i + j)
    return mat

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from tests import test_M, test_A, test_C
    from sympy.plotting import plot
    from sympy import Symbol

    deg = 5
    basis_fs = basis_functions(deg=deg)
    x = Symbol('x')
    ps = plot(basis_fs[0], (x, -1, 1), show=False)
    [ps.append(plot(f, (x, -1, 1), show=False)[0]) for f in basis_fs[1:]]
    ps.show()

    assert test_M(basis_fs, mass_matrix(deg))
    assert test_A(basis_fs, stiffness_matrix(deg))
    assert test_C(basis_fs, derivative_matrix(deg))
