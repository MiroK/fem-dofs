from __future__ import division
from sympy import legendre, Symbol, S
from scipy.linalg import eigh
import numpy as np
import legendre_basis as leg
import common

# IDEA some modif of QR to get orthogonality. I am fine having the matrix
# scale e.g. linearly as the mass of Legendre
# IDEA would be really nice to have the coefs analytically

def basis_functions(deg):
    '''
    Polynomial basis of degree P^deg[-1, 1] of polynomials orthonormal w.r.t
    to H^1 inner product.
    '''
    A = leg.mass_matrix(deg) + leg.stiffness_matrix(deg)
    # IDEA This can also be done with sympy.mpmath.matrix and eigsym routine
    # Consider for highter deg (precision?)
    lmbda, U = eigh(A)
    # In the new basis H1 inner = alpha.A.alpha.T
    # But from eigenvalues alpha.[U.Lmbda.U.T].alpha.T so making
    # alpha = Lmbda**{-0.5}.U.T
    Lmbda = np.diag(lmbda**(-0.5))
    alpha = Lmbda.dot(U.T)
    assert np.allclose(alpha.dot(A.dot(alpha.T)), np.eye(deg+1))

    # Now alha has in each row the new basis represented in Legendre polynomials
    leg_basis = leg.basis_functions(deg)
    basis = [sum([c*f for c, f in zip(row, leg_basis)], S(0)) for row in alpha]
    return basis

# Now all the matrices are given by numerical quadratues
# IDEA would be really nice to have the matrices analytically. Okay they are
# half dense (so I am glass half full :) but stil ...

def mass_matrix(deg, unroll=True):
    '''Mass matrix of the H1 orthonormal basis.'''
    return common.mass_matrix(deg, basis_functions, unroll)


def stiffness_matrix(deg, unroll=True):
    '''Stiffness matrix of the H1 orthonormal basis.'''
    return common.stiffness_matrix(deg, basis_functions, unroll)


def derivative_matrix(deg, unroll=True):
    '''
    Matrix of \int_{-1}^{1}(L_i, L_j`) of the H1 orthonormal basis.
    '''
    return common.derivative_matrix(deg, basis_functions, unroll)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from tests import test_M, test_A, test_C
    from sympy.plotting import plot

    deg = 5
    basis_fs = basis_functions(deg=deg)
    x = Symbol('x')
    ps = plot(basis_fs[0], (x, -1, 1), show=False)
    [ps.append(plot(f, (x, -1, 1), show=False)[0]) for f in basis_fs[1:]]
    # ps.show()

    assert test_M(basis_fs, mass_matrix(deg, True))
    assert test_M(basis_fs, mass_matrix(deg, False))
    assert test_A(basis_fs, stiffness_matrix(deg, True))
    assert test_A(basis_fs, stiffness_matrix(deg, False))
    assert test_C(basis_fs, derivative_matrix(deg, True))
    assert test_C(basis_fs, derivative_matrix(deg, False))
    
    # NOTE there is very little difference in speed between unroll True and False
