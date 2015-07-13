from scipy.linalg import toeplitz
import sympy.mpmath as mp
import numpy as np

n = 25
A = toeplitz(np.r_[2, 1, np.zeros(n)])

# Numpy ---
lmbda, U = np.linalg.eig(A)
Lmbda = np.diag(lmbda)
print np.linalg.norm(A - U.dot(Lmbda.dot(U.T)))

# Mpmath
def diag(diagonal):
    mat = mp.matrix(np.eye(len(diagonal)))
    for i, value in enumerate(diagonal):
        mat[i, i] = value
    return mat

A = mp.matrix(A.tolist())
E, Q = mp.eigsy(A)
E = diag(E)

# Sort for comparison
Lmbda = np.diag(np.sort(np.diagonal(Lmbda)))
print np.linalg.norm(Lmbda - np.array(E.tolist(), dtype=float))
