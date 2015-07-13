from __future__ import division
import numpy as np
from sympy import lambdify, Symbol


class LagrangeElement(object):
    '''
    Lagrange element over 1d element (-1, 1)

    The basis of polynomial space is specified by poly_set.
    The dofs are point-evaluation at points from nodal_set.
    '''
    def __init__(self, poly_set, dof_set):
        # Need to construct basis of polynomial space nodal with dofs
        # Poly_set is not the right one. The right on is linear combination
        x = Symbol('x')
        n = len(poly_set)
        # The matrix dof_i(f_j)
        B = np.zeros((n, n))
        for col, f in enumerate(poly_set):
            B[:, col] = lambdify(x, f)(dof_set)

        # Invert to get the coefficients of the nodal basis
        self.alpha = np.linalg.inv(B).T
        # Having coeffcient comes in handy if some matrix M is given w.r.t
        # poly_set basis then alpha.M.alpha.T is the matrix represented in
        # nodal basis

        # Symbolic nodal basis
        self.sym_basis = [sum(c*f for c, f in zip(row, poly_set))
                          for row in self.alpha]
        
        # Another way is to construct Lagrange interpolant of dofs and store its
        # coefficients. Then polyval...
        self.num_basis = np.zeros((n, n))
        for i in range(n):
            # 1 at i-th dof, zero everwhere else
            y = np.zeros(n)
            y[i] = 1
            self.num_basis[i] = np.polynomial.polynomial.polyfit(dof_set, y, n-1)
    

    def dim(self):
        return len(self.sym_basis)

    # Some FIAT/FFC like functionality

    def eval_basis(self, x, i):
        '''Evaluate i-th basis function at points x.'''
        basis_coefs = self.num_basis[i]
        if isinstance(x, (int, float)): x = np.array([x])
        if isinstance(x, (list, tuple)): x = np.array(x)
        # Always flatten
        x = x.flatten()
        y = np.polynomial.polynomial.polyval(x, basis_coefs)
        return y

    
    def eval_basis_all(self, x):
        '''Evaluate all basis functions at points x.'''
        # Each row corresponds to a basis function values at x
        return np.vstack([self.eval_basis(x, i) for i in range(self.dim())])


    def eval_basis_derivative(self, x, i, n):
        '''Evaluate n-th derivative of i-th basis functions at poitns x.'''
        basis_coefs = self.num_basis[i]
        # Take the derivative
        basis_coefs = np.polynomial.polynomial.polyder(basis_coefs, n)
        if isinstance(x, (int, float)): x = np.array([x])
        if isinstance(x, (list, tuple)): x = np.array(x)
        # Always flatten
        x = x.flatten()
        y = np.polynomial.polynomial.polyval(x, basis_coefs)
        return y


    def eval_basis_derivative_all(self, x, n):
        '''Evaluate n-th derivative of all basis functions at poitns x.'''
        return np.vstack([self.eval_basis_derivative(x, i, n)
                          for i in range(self.dim())])


    def eval(self, x, coefs):
        '''
        Evaluate linear combination of basis functions with coefs as weight
        in points x.
        '''
        assert len(coefs) == self.dim(), 'Need %d coeficients' % self.dim()
        if isinstance(coefs, (list, tuple)): coefs = np.array(coefs)
        # Reshape for mutliplying x
        coefs = coefs.reshape((self.dim(), 1))
        f_x = self.eval_basis_all(x)
        # Combine
        values = np.sum(coefs*f_x, axis=0)
        return values
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate
    from numpy.polynomial.legendre import leggauss

    # Sanity checks
    x = Symbol('x')
    poly_set = [1, x, x**2, x**3]
    dof_set = np.array([-1, -0.5, 0.5, 1])
    element = LagrangeElement(poly_set, dof_set)

    # Basis evaluation
    pts = [0.25, 0.5, -1.]
    v = element.eval_basis_all(pts)
    v_ = np.array([lambdify(x, element.sym_basis[i])(np.array(pts))
                   for i in range(element.dim())])
    assert np.allclose(v, v_)

    # Function evaluation
    coefs = np.random.rand(len(poly_set))
    v = element.eval(pts, coefs)
    # Make symbolic function
    f = sum(c*b for c, b in zip(coefs, element.sym_basis))
    # Lambdify and eval
    v_ = lambdify(x, f)(np.array(pts))
    assert np.allclose(v, v_)

    # Derivative evaluation
    order = 1
    v = element.eval_basis_derivative_all(pts, order)
    v_ = np.array([lambdify(x, element.sym_basis[i].diff(x, order))(np.array(pts))
                   for i in range(element.dim())])
    assert np.allclose(v, v_)

    # The matrix by transformation
    n = len(poly_set)
    # Exact poly_set matrix
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        M[i, i] = integrate(poly_set[i]*poly_set[i], (x, -1, 1))
        for j in range(i, n):
            M[i, j] = integrate(poly_set[i]*poly_set[j], (x, -1, 1))
            M[j, i] = M[i, j]

    # Nodal basis matrix
    M_nodal = np.zeros_like(M)
    points, weights = leggauss(deg=len(poly_set))
    for i in range(n):
        fi_xq = element.eval_basis(points, i)
        M_nodal[i, i] = np.sum(weights*fi_xq**2)
        for j in range(n):
            fi_xq = element.eval_basis(points, i)
            fj_xq = element.eval_basis(points, j)
            M_nodal[i, j] = np.sum(weights*fi_xq*fj_xq)
            M_nodal[j, i] = M_nodal[i, j]
    
    alpha = element.alpha
    M_nodal_ = alpha.dot(M.dot(alpha.T))
    assert np.allclose(M_nodal, M_nodal_)
