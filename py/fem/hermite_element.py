from __future__ import division
import sys
sys.path.append('../')
import polynomials.legendre_basis as leg
from points import gauss_legendre_points
from mesh import ReferenceIntervalCell
from sympy import lambdify, Symbol, integrate
import numpy as np


class HermiteElement(object):
    '''
    Hermite element over reference 1d element (-1, 1).
    '''
    def __init__(self, degree):

        b_degree = degree - 3
        x = Symbol('x')
        poly_set = leg.basis_functions(deg=degree)
        pts0 = gauss_legendre_points(b_degree+1)[1:-1]
        pts1 = np.array([-1., 1.])
        # The coefficient matrix
        B = np.zeros((degree+1, degree+1))
        B[:, 0] = [(-1)**k for k in range(degree+1)] # Values are (-1)
        B[:, 1] = np.ones(degree+1)                  # Values at (1)

        # Val at -1, val at 1, dval at -1, dval at 1, the rest of polyevaluas.
        for row, f in enumerate(poly_set):
            vals = lambdify(x, f, 'numpy')(pts0)
            if isinstance(vals, (int, float)): vals = vals*np.ones(len(pts0))
            dvals = lambdify(x, f.diff(x, 1), 'numpy')(pts1)
            if isinstance(dvals, (int, float)): dvals = dvals*np.ones(len(pts1))

            B[row, 2:4] = dvals
            B[row, 4:] = vals

        # Invert to get the coefficients of the nodal basis
        self.alpha = np.linalg.inv(B)
        # Having coeffcient comes in handy if some matrix M is given w.r.t
        # poly_set basis then alpha.M.alpha.T is the matrix represented in
        # nodal basis

        # Symbolic nodal basis
        self.sym_basis = [sum(c*f for c, f in zip(row, poly_set))
                          for row in self.alpha]
        # For numerical evaluation the row in alpha are coeffcients of Legendre
        # polynomials so we can use legval
        
        # Finally remember the dofs as coordinates val, val, dval, dval
        self.dofs = np.hstack([pts1, pts1, pts0])

        # And my finite element
        self.cell = ReferenceIntervalCell()

        # HACK computing df/dx without the derivatives. This is exact for P3 and
        # lower. Represent the L(f) = df/dx(p) = \int Riesz(L) * f dx
        # Riesz(L) = l is a polynomial of degree 3-compute its coeficients
        # They are given by mass matrix of the nodal basis
        xq, wq = np.polynomial.legendre.leggauss(degree+1)
        M = self.alpha.dot(leg.mass_matrix(degree).dot(self.alpha.T))
        beta = np.linalg.inv(M)
        # Now I have expansion w.r.t nodal. Combine with alpha to get Legendre
        # beta = beta.dot(self.alpha)
        # beta[2:4] are the coefs of df/dx eval at -1 and 1. Need them only at
        # quadrature points
        self.riesz = (np.polynomial.legendre.legval(xq, beta[2].dot(self.alpha)),
                      np.polynomial.legendre.legval(xq, beta[3].dot(self.alpha)))
        # Remember these for later evaluations
        self.quad = xq, wq
   
    @property
    def dim(self):
        return len(self.sym_basis)

    # Some FIAT/FFC like functionality
    # Other elements with cell K = Interval(a, b) are defined by mapping the
    # reference element. Stuff on reference is x_hat
    #
    # Geometry: x = F_k(x_hat) = (a/2)(1-x_hat) + (b/2)(1+x_hat)
    # (Basis) functions phi_i(x) = phi_hat(inv(F_k)(x_hat))
    # (Dofs)  eval @ x hat eval at F_k(x_hat)
    #         derivatives need to be scalled by cells Jacobian

    def eval_basis(self, i, x, cell=None):
        '''Evaluate i-th basis function at points x.'''
        basis_coefs = self.alpha[i]
        if isinstance(x, (int, float)): x = np.array([x])
        if isinstance(x, (list, tuple)): x = np.array(x)
        if cell is None: cell = self.cell
        # Map to reference domain
        x_hat = cell.map_to_reference(x)
        # Always flatten
        x_hat = x_hat.flatten()
        y = np.polynomial.legendre.legval(x_hat, basis_coefs)
        
        # Scale the basis that correspond to derivative dofs
        if i == 2 or i ==3:
            y /= cell.Jac

        return y
    
    def eval_basis_all(self, x, cell=None):
        '''Evaluate all basis functions at points x.'''
        # Each row corresponds to a basis function values at x
        return np.vstack([self.eval_basis(i, x, cell) for i in range(self.dim)])

    def eval_basis_derivative(self, i, n, x, cell=None):
        '''Evaluate n-th derivative of i-th basis functions at poitns x.'''
        basis_coefs = self.alpha[i]
        if isinstance(x, (int, float)): x = np.array([x])
        if isinstance(x, (list, tuple)): x = np.array(x)
        if cell is None: cell = self.cell
        # Map to reference domain
        x_hat = cell.map_to_reference(x)
        # Always flatten
        x_hat = x_hat.flatten()
        # Take the derivative
        dbasis_coefs = np.polynomial.legendre.legder(basis_coefs, m=n)
        y = np.polynomial.legendre.legval(x_hat, dbasis_coefs)
        y *= (cell.Jac)**n

        # Scale the basis that corrspond to derivative dofs
        if i == 2 or i == 3:
            y /= cell.Jac

        return y

    def eval_basis_derivative_all(self, n, x, cell=None):
        '''Evaluate n-th derivative of all basis functions at poitns x.'''
        return np.vstack([self.eval_basis_derivative(i, n, x, cell)
                          for i in range(self.dim)])

    def eval_dof(self, i, f, cell=None):
        '''Evaluate dof i of the cell at function f.'''
        # Point evaluation
        if cell is None: cell = self.cell
        # The derivatives are computed from Riesz by inner product for now.
        # Only requires evaluation of f and not the derivatives
        if i == 2 or i == 3:
            xq, wq = self.quad
            # Mirrow the points
            xqK = cell.map_from_reference(xq)
            f_xqK = f.eval_cell(xqK, cell)
            riesz = self.riesz[0 if i == 2 else 1]
            return np.sum(f_xqK*riesz*wq)*cell.Jac
        # The dof position is x_hat -- need to eval at x in cell
        else:
            return f.eval_cell(cell.map_from_reference(self.dofs[i]), cell)

    def eval_dofs(self, f, cell=None):
        '''Evaluate all dofs'''
        return np.vstack([self.eval_dof(i, f, cell) for i in range(self.dim)])

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate
    from numpy.polynomial.legendre import leggauss
    from function import Expression

    degree = 4
    element = HermiteElement(degree)
    poly_set = leg.basis_functions(degree)
    x = Symbol('x')

    from mesh import IntervalCell
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    # Funcions and derivatives. Check continuity
    for a, b in [(-1.4, -1), (-1, 1), (1, 2), (2, 4)]:
        x = np.linspace(a, b, 100)
        cell = IntervalCell(np.array([[a], [b]]))
        for i, color in zip(range(element.dim), ['r', 'b', 'g', 'c', 'm', 'k']):

            y = element.eval_basis(i, x, cell)
            ax0.plot(x, y, color=color)

            y = element.eval_basis_derivative(i, 1, x, cell)
            ax1.plot(x, y, color=color)
    # plt.show()

    x = Symbol('x')
    # Check reference cell
    if True:
        # pts = np.array([0.3])
        pts = np.array([[0.25], [0.5], [-1.], [0.1]])
        # Basis evaluation
        v = element.eval_basis_all(pts)
        v_ = np.array([lambdify(x, element.sym_basis[i])(pts)
                       for i in range(element.dim)]).reshape(v.shape)
        assert np.allclose(v, v_)
        
        # Derivative evaluation
        order = 1
        v = element.eval_basis_derivative_all(order, pts)
        v_ = np.array([lambdify(x, element.sym_basis[i].diff(x, order))(np.array(pts))
                       for i in range(element.dim)]).reshape(v.shape)
        assert np.allclose(v, v_)

        # The matrix by transformation
        n = len(poly_set)
        # Exact poly_set matrix
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            M[i, i] = integrate(poly_set[i]*poly_set[i], (x, -1, 1))
            for j in range(i+1, n):
                M[i, j] = integrate(poly_set[i]*poly_set[j], (x, -1, 1))
                M[j, i] = M[i, j]

        # Nodal basis matrix
        M_nodal = np.zeros_like(M)
        points, weights = leggauss(deg=len(poly_set))
        for i in range(n):
            fi_xq = element.eval_basis(i, points)
            M_nodal[i, i] = np.sum(weights*fi_xq**2)
            for j in range(i+1, n):
                fi_xq = element.eval_basis(i, points)
                fj_xq = element.eval_basis(j, points)
                M_nodal[i, j] = np.sum(weights*fi_xq*fj_xq)
                M_nodal[j, i] = M_nodal[i, j]
        
        alpha = element.alpha
        M_nodal_ = alpha.dot(M.dot(alpha.T))
        assert np.allclose(M_nodal, M_nodal_)

        # Check nodality
        n = element.dim
        I = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                I[i, j] = element.eval_dof(j, Expression(element.sym_basis[i]))
        assert np.allclose(I, np.eye(n))
