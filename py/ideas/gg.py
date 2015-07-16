from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.sparse import diags
import numpy as np
import sys

deg = int(sys.argv[1])

for deg in range(deg, deg+1):
    
    # Understand why: the presentation is good
    k = np.arange(1, deg)
    diag = k/np.sqrt(4*k**2-1)
    # Would be nice to have the whole Jacobi thing awilable ->
    # tri, tet quad

    J = diags([diag, diag], [-1, 1], shape=(deg, deg))
    lmbda, U = eigh(J.toarray())

    points = lmbda
    weights = 2*U[0, :]**2

    xq, wq = leggauss(deg)

    print xq, wq
    print np.linalg.norm(xq-points), np.linalg.norm(wq-weights)
