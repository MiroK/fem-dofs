from __future__ import division
import numpy as np
from sympy import lambdify, Symbol
from mesh import ReferenceIntervalCell
from function import Expression


class LagrangeElement(object):
    '''
    Lagrange element over reference 1d element (-1, 1)

    The basis of polynomial space is specified by poly_set.
    The dofs are point-evaluation at points from nodal_set.
    '''
    def __init__(self, poly_set, dof_set):
        assert len(poly_set) == len(dof_set)
        # Need to construct basis of polynomial space nodal with dofs
        # Poly_set is not the right one. The right on is its linear combination
        x = Symbol('x')
        n = len(poly_set)
        # The matrix dof_i(f_j)
        # Dofs come as [-1, interior, 1]. For dofmap construction it is easier
        # to have [-1, 1](exterior) + interior dofs.
        dof_set = np.r_[dof_set[0], dof_set[-1], dof_set[1:-1]]
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
        
        # Finally remember the dofs as coordinates
        self.dofs = dof_set.reshape((-1, 1))
        # And my finite element
        self.cell = ReferenceIntervalCell()
   
    @property
    def dim(self):
        return len(self.sym_basis)

    def riesz_basis(self, inner_product):
        '''Compute the riesz representant of dofs in the nodal basis'''
        M = np.zeros((self.dim, self.dim))
        for i, u in enumerate(self.sym_basis):
            M[i, i] = inner_product(u, u)
            for j, v in enumerate(self.sym_basis[i+1:], i+1):
                M[i, j] = inner_product(u, v)
                M[j, i] = M[i, j]

        print '>>', np.linalg.eigvals(M)
        print M
        print '>>'

        beta = np.linalg.inv(M)
        return [sum(c*f for c, f in zip(row, self.sym_basis)) for row in beta]

    # Some FIAT/FFC like functionality
    # Other elements with cell K = Interval(a, b) are defined by mapping the
    # reference element. Stuff on reference is x_hat
    #
    # Geometry: x = F_k(x_hat) = (a/2)(1-x_hat) + (b/2)(1+x_hat)
    # (Basis) functions phi_i(x) = phi_hat(inv(F_k)(x_hat))
    # (Dofs)  eval @ x hat eval at F_k(x_hat)

    def eval_basis(self, i, x, cell=None):
        '''Evaluate i-th basis function at points x.'''
        basis_coefs = self.num_basis[i]
        if isinstance(x, (int, float)): x = np.array([x])
        if isinstance(x, (list, tuple)): x = np.array(x)
        if cell is None: cell = self.cell
        # Map to reference domain
        x_hat = cell.map_to_reference(x)
        # Always flatten
        x_hat = x_hat.flatten()
        y = np.polynomial.polynomial.polyval(x_hat, basis_coefs)
        return y
    
    def eval_basis_all(self, x, cell=None):
        '''Evaluate all basis functions at points x.'''
        # Each row corresponds to a basis function values at x
        return np.vstack([self.eval_basis(i, x, cell) for i in range(self.dim)])

    def eval_basis_derivative(self, i, n, x, cell=None):
        '''Evaluate n-th derivative of i-th basis functions at poitns x.'''
        basis_coefs = self.num_basis[i]
        # Take the derivative
        basis_coefs = np.polynomial.polynomial.polyder(basis_coefs, n)
        if isinstance(x, (int, float)): x = np.array([x])
        if isinstance(x, (list, tuple)): x = np.array(x)
        if cell is None: cell = self.cell
        # Map to reference domain
        x_hat = cell.map_to_reference(x)
        # Always flatten
        x_hat = x_hat.flatten()
        y = np.polynomial.polynomial.polyval(x_hat, basis_coefs)
        y *= (cell.Jac)**n
        return y

    def eval_basis_derivative_all(self, n, x, cell=None):
        '''Evaluate n-th derivative of all basis functions at poitns x.'''
        return np.vstack([self.eval_basis_derivative(i, n, x, cell)
                          for i in range(self.dim)])

    def eval_dof(self, i, f, cell=None):
        '''Evaluate dof i of the cell at function f.'''
        # Point evaluation
        if cell is None: cell = self.cell
        # The dof position is x_hat -- need to eval at x in cell
        return f.eval_cell(cell.map_from_reference(self.dofs[i]), cell)

    def eval_dofs(self, f, cell=None):
        '''Evaluat all dofs'''
        return np.vstack([self.eval_dof(i, f, cell) for i in range(self.dim)])
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy import integrate, S
    from numpy.polynomial.legendre import leggauss
    from mesh import IntervalCell

    # Sanity checks
    x = Symbol('x')
    poly_set = [S(1), x, x**2, x**3]
    dof_set = np.array([-1, -0.5, 0.5, 1])
    element = LagrangeElement(poly_set, dof_set)

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

        # Check riesz
        I = np.zeros((n, n))
        L2_ip = lambda u, v: integrate(u*v, (x, -1, 1))
        H1_ip = lambda u, v: integrate(u.diff(x, 1)*v.diff(x, 1) + u*v, (x, -1, 1))
        H10_ip = lambda u, v: integrate(u.diff(x, 1)*v.diff(x, 1), (x, -1, 1))

        inner_product = L2_ip
        basis = element.sym_basis
        riesz_basis = element.riesz_basis(inner_product)
        for i, L in enumerate(riesz_basis):
            for j, f in enumerate(basis):
                I[i, j] = inner_product(L, f)
        assert np.allclose(I, np.eye(n))

        M = np.zeros((self.dim, self.dim))
        for i, u in enumerate(riesz_basis):
            M[i, i] = inner_product(u, u)
            for j, v in enumerate(riesz_basis[i+1:], i+1):
                M[i, j] = inner_product(u, v)
                M[j, i] = M[i, j]
        print M
        # from sympy.plotting import plot
        # x = Symbol('x')
        # ps = plot(basis[0], (x, -1, 1), show=False, color='b')
        # [ps.append(plot(f, (x, -1, 1), show=False, color='b')[0]) for f in basis[1:]]
        # ps = plot(riesz_basis[0], (x, -1, 1), show=False)
        # [ps.append(plot(f, (x, -1, 1), show=False)[0]) for f in riesz_basis[1:]]
        # ps.show()

    # Check other cell
    a, b, = 0, 1.1
    cell = IntervalCell(np.array([[a], [b]]))
    if True:
        # pts = np.array([0.3])
        pts = np.array([[0.25], [0.5], [0.1]])
        # Basis evaluation
        v = element.eval_basis_all(pts, cell=cell)
        # Convert the basis of element to [a, b]
        v_ = np.array([lambdify(x, element.sym_basis[i].subs(x, (2*x-a-b)/(b-a)))(pts)
                       for i in range(element.dim)]).reshape(v.shape)
        assert np.allclose(v, v_)

        # Derivative evaluation
        order = 1
        v = element.eval_basis_derivative_all(order, pts, cell=cell)
        # Convert the basis of element to [a, b]
        v_ = np.array([lambdify(x, element.sym_basis[i].subs(x, (2*x-a-b)/(b-a)).diff(x, order))(np.array(pts))
                       for i in range(element.dim)]).reshape(v.shape)
        assert np.allclose(v, v_)

        # The matrix on this element, exact symbolic
        # Map the reference element basis to element K
        basis_K = [f.subs(x, (2*x-a-b)/(b-a)).diff(x, 1) for f in element.sym_basis]
        A_K = np.zeros((n, n), dtype=float)
        # Perform exact integration
        for i in range(n):
            A_K[i, i] = integrate(basis_K[i]*basis_K[i], (x, a, b))
            for j in range(i+1, n):
                A_K[i, j] = integrate(basis_K[i]*basis_K[j], (x, a, b))
                A_K[j, i] = A_K[i, j]

        # See if I understand the integration
        A_K_ = np.zeros_like(A_K)
        points, weights = leggauss(deg=len(poly_set))
        # Map the points to cell
        points = cell.map_from_reference(points)
        for i in range(n):
            fi_xq = element.eval_basis_derivative(i, 1, points, cell)
            A_K_[i, i] = np.sum(weights*fi_xq**2)
            for j in range(i+1, n):
                fj_xq = element.eval_basis_derivative(j, 1, points, cell)
                A_K_[i, j] = np.sum(weights*fi_xq*fj_xq)
                A_K_[j, i] = A_K_[i, j]

        # Volume measure Jacobian
        A_K_ =A_K_/cell.Jac
        assert np.allclose(A_K, A_K_)

        # Can do the same by mapping the reference element matrix. It is
        # integrated numerically on reference cell
        A_ref = np.zeros_like(A_K)
        points, weights = leggauss(deg=len(poly_set))
        for i in range(n):
            fi_xq = element.eval_basis_derivative(i, 1, points)
            A_ref[i, i] = np.sum(weights*fi_xq**2)
            for j in range(i+1, n):
                fj_xq = element.eval_basis_derivative(j, 1, points)
                A_ref[i, j] = np.sum(weights*fi_xq*fj_xq)
                A_ref[j, i] = A_ref[i, j]
        
        # J*J from derivs and 1./J from volume <-- this is the geoemtry tensor 
        A_ref = A_ref*cell.Jac
        assert np.allclose(A_K, A_ref)
