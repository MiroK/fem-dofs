from __future__ import division
from scipy.sparse import triu, tril, diags
from scipy.sparse.linalg import spsolve
import numpy as np

# Iterative solvers for Ax = b of the type
#   
#   M*x_new = N*x_old + b
#
# for M, N the splits of A such that A = M - N

def richardson(A, b, x0=None, w=1., maxiter=200, tol=1E-6):
    '''(Weighted) Richardson iteration has M = I, N = I - A.'''
    # Start from 0 initial guess
    if x0 is None: x0 = np.zeros(A.shape[1])

    r = b - A.dot(x0)
    residuals = [np.linalg.norm(r)]

    count = 0
    while residuals[-1] > tol and count < maxiter:
        # Update: x = (1-w*A)x0 + w*b
        x0 = x0 - w*A.dot(x0) + w*b
        # Error 
        r = b - A.dot(x0)
        residuals.append(np.linalg.norm(r))
        # Count
        count += 1
    
    converged = residuals[-1] < tol
    n_iters = len(residuals) - 1
    data = {'status': converged, 'iter count': n_iters, 'residuals': residuals}

    return x0, data


def jacobi(A, b, x0=None, maxiter=200, tol=1E-6):
    '''Jacobi iteration has M = D, N = D - A.'''
    # Start from 0 initial guess
    if x0 is None: x0 = np.zeros(A.shape[1])

    r = b - A.dot(x0)
    residuals = [np.linalg.norm(r)]

    D = A.diagonal()
    count = 0
    while residuals[-1] > tol and count < maxiter:
        # Update: Dx = (D-A)*x0 + b
        x0 = (D*x0 - A.dot(x0) + b)/D
        # Error 
        r = b - A.dot(x0)
        residuals.append(np.linalg.norm(r))

        # Count
        count += 1
    
    converged = residuals[-1] < tol
    n_iters = len(residuals) - 1
    data = {'status': converged, 'iter count': n_iters, 'residuals': residuals}

    return x0, data


def sor(A, b, x0=None, w=1., maxiter=200, tol=1E-6, direction='forward'):
    '''
    SOR iteration has M = L + D/w, N = (1/w-1)*D - U for forward
    and M = U + D/w, N = (1/w-1)*D - L for bacward.
    '''
    L, D, U = tril(A, k=-1), diags(A.diagonal(), 0), triu(A, k=1)
    if direction == 'forward':
        M = L + D/w
        N = (1/w - 1)*D - U
    else:
        M = U + D/w
        N = (1/w - 1)*D - L

    # Start from 0 initial guess
    if x0 is None: x0 = np.zeros(A.shape[1])

    r = b - A.dot(x0)
    residuals = [np.linalg.norm(r)]

    count = 0
    while residuals[-1] > tol and count < maxiter:
        # Update
        x0 = spsolve(M, N.dot(x0) + b)
        # Error 
        r = b - A.dot(x0)
        residuals.append(np.linalg.norm(r))
        # Count
        count += 1
    
    converged = residuals[-1] < tol
    n_iters = len(residuals) - 1
    data = {'status': converged, 'iter count': n_iters, 'residuals': residuals}

    return x0, data


def gauss_seidel(A, b, x0=None, maxiter=200, tol=1E-6):
    '''Gauss-Seidel iteration has M = L + D, N = -U.'''
    return sor(A, b, x0=x0, maxiter=maxiter, tol=tol, w=1., direction='forward')


def ssor(A, b, x0=None, w=1., maxiter=200, tol=1E-6):
    '''For symmetric matrices combine forward and backward SOR.'''
    assert is_symmetric(A, tol=1E-6)

    L, D, U = tril(A, k=-1), diags(A.diagonal(), 0), triu(A, k=1)
    # Forward
    MF = L + D/w
    NF = (1/w - 1)*D - U
    # Backward
    MB = U + D/w
    NB = (1/w - 1)*D - L

    # Start from 0 initial guess
    if x0 is None: x0 = np.zeros(A.shape[1])

    r = b - A.dot(x0)
    residuals = [np.linalg.norm(r)]

    count = 0
    while residuals[-1] > tol and count < maxiter:
        # Update
        x0 = spsolve(MF, NF.dot(x0) + b)
        x0 = spsolve(MB, NB.dot(x0) + b)
        # Error 
        r = b - A.dot(x0)
        residuals.append(np.linalg.norm(r))
        # Count
        count += 1
    
    converged = residuals[-1] < tol
    n_iters = len(residuals) - 1
    data = {'status': converged, 'iter count': n_iters, 'residuals': residuals}

    return x0, data


def symmetric_gauss_seidel(A, b, x0=None, maxiter=200, tol=1E-6):
    '''Symmetrix Gauss-Seidel iterations.'''
    assert is_symmetric(A, tol=1E-6)
    return ssor(A, b, x0=x0, maxiter=maxiter, tol=tol, w=1.)


def is_symmetric(A, tol):
    '''Cehck if matrix is symmetric to within tolerance.'''
    x = np.random.rand(A.shape[1])
    return np.linalg.norm(A.dot(x) - A.transpose().dot(x)) < tol
        

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    from scipy.linalg import toeplitz, eigvalsh
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
   
    # Setup test with sym-pos def matrix
    N = 50
    A = csr_matrix(toeplitz(np.r_[2., -1., np.zeros(N)]))/N
    b = np.random.rand(A.shape[0])
    x_true = spsolve(A, b)

    # Optimal weight for richardson
    ew = eigvalsh(A.toarray())
    lmin, lmax = np.min(ew), np.max(ew)

    methods = {'richardson': (richardson, 2/(lmin+lmax), 'r'),
               'jacobi' : (jacobi, None, 'b'),
               'gs': (gauss_seidel, None, 'g'),
               'sor': (sor, 1.6, 'k'),
               'sgs': (symmetric_gauss_seidel, None, 'm'),
               'ssor': (ssor, 1.6, 'y')}

    maxiter = 10000

    # Compare all
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(range(len(x_true)), x_true, label='Exact', color='c')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('residual error')

    for method in methods:
        
        call, w, color = methods[method]
        if w is None:
            x, data = call(A, b, maxiter=maxiter)
        else:
            x, data = call(A, b, w=w)
        
        res = data['residuals']
        
        # Compare with spsolve
        ax0.plot(range(len(x)), x, color=color, label=method)
    
        # Convergence history
        ax1.loglog(range(1, len(res)+1), res, color=color)

    ax0.legend(loc='best')
    plt.show()
