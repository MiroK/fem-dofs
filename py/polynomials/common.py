from sympy import Symbol, lambdify
import numpy as np

# Numerical quadratures of some matrices

def mass_matrix(deg, basis_functions, unroll=True):
    '''Mass matrix of polynomials spanned by deg+1 basis_functions.'''
    # Numerical quadrature, max degree of integrand 2*deg -> deg+1
    xq, wq = np.polynomial.legendre.leggauss(deg+1)
    # Fast eval basis
    x = Symbol('x')
    basis = [lambdify(x, f, 'numpy') for f in basis_functions(deg)]

    M = np.zeros((len(basis), len(basis)))
    # Save calls to eval
    if unroll:
        # Basis functions evaluated in quadrature points
        for row, f in enumerate(basis[:]):
            M[row, :] = f(xq)
        # Integrate 
        M = (wq*M).dot(M.T)
    # Here there are more call to eval
    else:
        for i, v in enumerate(basis):
            v_xq = v(xq)
            M[i, i] = np.sum(wq*v_xq**2)
            for j, u in enumerate(basis[i+1:], i+1):
                M[i, j] = np.sum(wq*v(xq)*u(xq))
                M[j, i] = M[i, j]
    return M


def stiffness_matrix(deg, basis_functions, unroll=True):
    '''Stiffness matrix of polynomials spanned by deg+1 basis_functions.'''
    # Numerical quadrature, mas degree of integrand 2*deg - 2 -> deg
    xq, wq = np.polynomial.legendre.leggauss(deg)
    # Fast eval basis derivative
    x = Symbol('x')
    basis = [lambdify(x, f.diff(x, 1), 'numpy') for f in basis_functions(deg)]

    # Save calls to eval
    if unroll:
        A = np.zeros((len(basis), len(xq)))
        # Basis functions evaluated in quadrature points
        for row, f in enumerate(basis[:]):
            A[row, :] = f(xq)
        # Integrate 
        A = (wq*A).dot(A.T)
    # Here there are more call to eval
    else:
        A = np.zeros((len(basis), len(basis)))
        for i, v in enumerate(basis):
            v_xq = v(xq)
            A[i, i] = np.sum(wq*v_xq**2)
            for j, u in enumerate(basis[i+1:], i+1):
                A[i, j] = np.sum(wq*v(xq)*u(xq))
                A[j, i] = A[i, j]
    return A


def derivative_matrix(deg, basis_functions, unroll=True):
    '''
    Matrix of \int_{-1}^{1}(L_i, L_j`) for polynomials spanned by deg+1 
    basis_functions.
    '''
    # Numerical quadrature, mas degree of integrand 2*deg - 1 -> deg
    xq, wq = np.polynomial.legendre.leggauss(deg)
    # Fast eval of basis and asis derivative
    x = Symbol('x')
    basis = [lambdify(x, f, 'numpy') for f in basis_functions(deg)]
    dbasis = [lambdify(x, f.diff(x, 1), 'numpy') for f in basis_functions(deg)]
    
    # Save calls to eval
    if unroll:
        V = np.zeros((len(basis), len(xq)))
        # Basis functions evaluated in quadrature points
        for row, f in enumerate(basis): V[row, :] = f(xq)

        dV = np.zeros((len(xq), len(basis)))
        # Basis derivatives evaluated in quadrature points
        for col, df in enumerate(dbasis): dV[:, col] = df(xq)
        
        # Integrate 
        C = (wq*V).dot(dV)

    # Here there are more calls to eval
    else:
        C = np.zeros((len(basis), len(basis)))
        for i, v in enumerate(basis):
            for j, du in enumerate(dbasis):
                C[i, j] = np.sum(wq*v(xq)*du(xq))
    
    return C
