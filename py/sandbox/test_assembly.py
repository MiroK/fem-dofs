#
#
#
#

from numpy.polynomial.legendre import legval, legder, leggauss
import numpy as np


def get_basis(deg, dofset=None):
    # Legendre polynomials are used as the basis of polynomials. In the basis of
    # Legendre polynomials is row of eye
    polyset = np.eye(deg+1)
   
    if dofset is None: dofset = np.linspace(-1, 1, deg+1)
    # Reatange dofs to have external first
    dofset = np.r_[dofset[0], dofset[-1], dofset[1:-1]]
    # Compute the nodal matrix
    A = np.array([legval(dofset, base) for base in polyset])
    # New coefficients
    B = np.linalg.inv(A) 

    # Combine the basis according to new weights
    basis = [lambda x, c=c: legval(x, c) for c in B]
    # Check that the basis is nodal
    assert np.allclose(np.array([f(dofset) for f in basis]), polyset)
    # First deriv
    dbasis = [lambda x, c=legder(c): legval(x, c) for c in B]
    
    return basis, dbasis


def mass_matrix(basis):
    deg = len(basis) + 1
    xq, wq = leggauss(deg)

    Ujq = np.array([uj(xq) for uj in basis])
    Viq = np.array([ui(xq) for ui in basis])
    M = (Viq*wq).dot(Ujq.T)
    return M

def stiffness_matrix(basis):
    deg = len(basis)

    xq, wq = leggauss(deg)

    Ujq = np.array([uj(xq) for uj in basis])
    Viq = np.array([ui(xq) for ui in basis])
    A = (Viq*wq).dot(Ujq.T)
    return A

def G_matrix(basis, d_basis):
    xq, wq = np.array([-1, 1]), np.array([-1, 1])

    Ujq = np.array([uj(xq) for uj in dbasis])
    Viq = np.array([ui(xq) for ui in basis])
    A = (Viq*wq).dot(Ujq.T)
    return A

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    deg = 4

    basis, dbasis = get_basis(deg)

    print mass_matrix(basis)
    print

    print stiffness_matrix(dbasis)
    print

    print G_matrix(basis, dbasis)


