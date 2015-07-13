from __future__ import division
from sympy import legendre, Symbol
from scipy.sparse import diags
import numpy as np


def basis_functions(deg):
    '''Polynomial basis of degree P^deg[-1, 1] of Lengendre polynomials.'''
    return [legendre(k, Symbol('x')) for k in range(deg+1)]


def mass_matrix(deg):
    '''Mass matrix of Legendre polynomials of degree up to deg.'''
    n = deg + 1
    return diags(np.array([2./(2*i+1) for i in range(n)]), 0).toarray()


def stiffness_matrix(deg):
    '''Stiffness matrix of Legendre polynomials of degree up to deg.'''
    n = deg + 1
    # The matrix has part of the main diagonal on every second side diagonal
    main_diag = np.array([sum(2*(2*k+1) for k in range(0 if i%2 else 1, i, 2))
                          for i in range(n)])
    # Upper diagonals
    offsets = range(0, n, 2)
    diagonals = [main_diag[:n-offset] for offset in offsets]
    # All diagonal
    all_offsets = [-offset for offset in offsets[:0:-1]] + offsets
    all_diagonals = [diagonal for diagonal in diagonals[:0:-1]] + diagonals

    return diags(all_diagonals, all_offsets, shape=(n, n)).toarray()


def derivative_matrix(deg):
    '''
    Matrix of \int_{-1}^{1}(L_i, L_j`) of Legendre polynomials of degree up
    to deg.
    '''
    n = deg + 1
    offsets = []
    diagonals = []
    for k in range(1, deg+1, 2):
        offsets.append(k)
        diagonals.append(2*np.ones(deg+1-k))
    
    return diags(diagonals, offsets, shape=(n, n)).toarray()

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
