from __future__ import division
from sympy import Symbol, binomial, Rational
import numpy as np
import common

from scipy.sparse import diags
from scipy.special import binom, hyp2f1


def basis_functions(deg):
    '''Polynomial basis of degree P^deg[-1, 1] of Bernstein polynomials.'''
    x = Symbol('x')
    return [binomial(deg, k)*((1+x)**k)*((1-x)**(deg-k))*(Rational(1, 2)**deg)
            for k in range(deg+1)]


# FIXME until then things are numeric
def mass_matrix(deg, unroll=True):
    '''Mass matrix of Bernstein basis.'''
    return common.mass_matrix(deg, basis_functions, unroll)


def stiffness_matrix(deg, unroll=True):
    '''Stiffness matrix of Bernstein basis.'''
    return common.stiffness_matrix(deg, basis_functions, unroll)


def derivative_matrix(deg, unroll=True):
    '''
    Matrix of \int_{-1}^{1}(L_i, L_j`) Bernstein basis.
    '''
    return common.derivative_matrix(deg, basis_functions, unroll)

# def _integrate(a, b):
#     ''' 
#     int_{-1}^{1} (1-x)^a (1+x)^b
#     '''
#     feval = lambda x: 2**a*(x + 1)**(b+1)*hyp2f1(-a, b+1, b+2, 0.5*(x+1))/(b+1)
#     return feval(1) - feval(-1)
# 
# 
# def mass_matrix(deg):
#     '''Mass matrix of Bernstein polynomials of degree up to deg.'''
#     n = deg+1
#     mat = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             mat[i, j] = binom(deg, i)*binom(deg, j)*_integrate(2*deg-i-j, i+j)
#     mat *= 0.5**(2*deg)
#     return mat
# 
# 
# def stiffness_matrix(deg):
#     '''Stiffness matrix of Bernstein polynomials of degree up to deg.'''
#     n = deg+1
#     mat = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i == j and (i == 0 or i == deg):
#                 mat[i, j] = deg**2/2/(2*deg-1)
#             elif ((i + j) == 1) or ((i+j) == 2*deg-1):
#                 mat[i, j] = deg*_integrate(2*deg-2, 0)
#                 mat[i, j] = -deg*(deg-1)*_integrate(2*deg-3, 1)
#                 #mat[i, j] += (deg-i-1)*(deg-j-1)*_integrate(2*deg-i-j-2, i+j)
# 
#                 mat[i, j] *= binom(deg, i)*binom(deg, j)
#                 mat[i, j] *= 0.5**(2*deg)
#             else:
#                 mat[i, j] = i*j*_integrate(2*deg-i-j, i+j-2)
#                 mat[i, j] -= i*(deg-j-1)*_integrate(2*deg-i-j-1, i+j-1)
#                 mat[i, j] -= j*(deg-i-1)*_integrate(2*deg-i-j-1, i+j-1)
#                 mat[i, j] += (deg-i-1)*(deg-j-1)*_integrate(2*deg-i-j-2, i+j)
# 
#                 mat[i, j] *= binom(deg, i)*binom(deg, j)
#                 mat[i, j] *= 0.5**(2*deg)
#     return mat
# 
# 
# def derivative_matrix(deg):
#     '''
#     Matrix of \int_{-1}^{1}(B_i, B_j`) of Bernstein polynomials of degree up
#     to deg.
#     '''
#     n = deg+1
#     mat = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i == j and (i == 0 or i == deg):
#                 mat[i, j] = -0.5 if i == 0 else 0.5
#             else:
#                 mat[i, j] = j*_integrate(2*deg-i-j, i+j-1)
#                 mat[i, j] -= (deg-j)*_integrate(2*deg-i-j-1, i+j)
#                 mat[i, j] *= binom(deg, i)*binom(deg, j)
#                 mat[i, j] *= 0.5**(2*deg)
#     return mat

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from tests import test_M, test_A, test_C
    from sympy.plotting import plot
    from sympy import Symbol, simplify, S

    deg = 5
    basis_fs = basis_functions(deg=deg)

    # Check partition of unity property
    assert simplify(sum(basis_fs)) == S(1)

    x = Symbol('x')
    ps = plot(basis_fs[0], (x, -1, 1), show=False)
    [ps.append(plot(f, (x, -1, 1), show=False)[0]) for f in basis_fs[1:]]
    # ps.show()

    assert test_M(basis_fs, mass_matrix(deg))
    assert test_A(basis_fs, stiffness_matrix(deg))
    assert test_C(basis_fs, derivative_matrix(deg))
