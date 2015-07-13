from __future__ import division
from sympy import legendre, Symbol, S
from scipy.linalg import eigh
import numpy as np
import common


def basis_functions(deg):
    '''
    Polynomial basis of degree P^deg[-1, 1] from Chebyshev polynomials of the
    first kind.
    '''
    x = Symbol('x')
    basis = [S(1), x]
    k = 1
    while k < deg:
        basis.append(2*x*basis[-1] - basis[-2])
        k += 1
    return basis

# Now all the matrices are given by numerical quadratues
# IDEA would be really nice to have the matrices analytically. Okay they are
# half dense (so I am glass half full :) but stil ...

def mass_matrix(deg, unroll=True):
    '''Mass matrix of Chebyshev polynomials of first kind.'''
    return common.mass_matrix(deg, basis_functions, unroll)


def stiffness_matrix(deg, unroll=True):
    '''Stiffness matrix of Chebyshev polynomials of first kind.'''
    return common.stiffness_matrix(deg, basis_functions, unroll)


def derivative_matrix(deg, unroll=True):
    '''
    Matrix of \int_{-1}^{1}(L_i, L_j`) from Chebyshev polynomials of first kind.
    '''
    return common.derivative_matrix(deg, basis_functions, unroll)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from tests import test_M, test_A, test_C
    from sympy.plotting import plot
    from sympy import simplify

    deg = 5
    basis_fs = basis_functions(deg=deg)
    x = Symbol('x')
    # Need to check the definition
    reference = [1, x, 2*x**2-1, 4*x**3-3*x, 8*x**4-8*x**2+1]
    assert not any(map(simplify, (f-f_ for f, f_ in zip(basis_fs, reference))))

    ps = plot(basis_fs[0], (x, -1, 1), show=False)
    [ps.append(plot(f, (x, -1, 1), show=False)[0]) for f in basis_fs[1:]]
    # ps.show()

    assert test_M(basis_fs, mass_matrix(deg, True))
    assert test_A(basis_fs, stiffness_matrix(deg, True))
    assert test_C(basis_fs, derivative_matrix(deg, True))
