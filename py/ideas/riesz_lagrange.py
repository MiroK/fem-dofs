# Every functional represent by integral...
from sympy import Symbol, lambdify, integrate, S
import numpy as np

x = Symbol('x')

class LagrangeElement(object):
    def __init__(self, degree, a, b):
        self.vertices = np.linspace(a, b, degree+1)
        poly_set = [S(1)] + [x**k for k in range(1, degree+1)]

        V = np.zeros((len(poly_set), len(poly_set)))
        for row, f in enumerate(poly_set):
            V[row, :] = lambdify(x, f, 'numpy')(self.vertices)
        alpha = np.linalg.inv(V)

        self.sym_basis = [sum(c*f for c, f in zip(row, poly_set)) for row in alpha]
        self.num_basis = [lambdify(x, f) for f in self.sym_basis]

        # Riez representation of functionals
        # Mass matrix of nodal basis
        dim = len(self.sym_basis)
        M = np.zeros((dim, dim))
        xq, wq = np.polynomial.legendre.leggauss(degree+1)
        X = np.zeros((dim, len(xq)))
        for row, f in enumerate(self.num_basis):
            X[row, :] = f(xq)
        M = wq*(X.dot(X.T))

        M = np.zeros((dim, dim))
        for i, u in enumerate(self.sym_basis):
            for j, v in enumerate(self.sym_basis):
                M[i, j] = integrate(u*v, (x, a, b))

        # Functional eval at nodal basis
        L = np.zeros((dim, dim))
        for col, f in enumerate(self.num_basis):
            L[:, col] = f(self.vertices)

        print L
        # Compute expansion coefs
        beta = np.linalg.inv(M).dot(L)
        self.riesz_basis = [sum(c*phi for c, phi in zip(row, self.sym_basis))
                            for row in beta]

    def eval_basis(self, i, x):
        return self.num_basis[i](x)

    def eval_dof(self, i, f):
        return f(self.vertices[i])

    def eval_dof_riesz(self, i, f):
        a, b = self.vertices[0], self.vertices[-1]
        return integrate(self.riesz_basis[i]*f, (x, a, b))

    def eval_dof_riesz_cell(self, i, f, cell):
        # ONLY (-1, 1)
        a, b = cell
        J = (b-a)/2.
        f_xhat = f.subs(x, 0.5*a*(1-x)+0.5*b*(1+x))*J
        return integrate(self.riesz_basis[i]*f_xhat, (x, -1, 1))

    @property
    def dim(self):
        return len(self.num_basis)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from sympy.plotting import plot
    from sympy.mpmath import quad

   
    degree = 4
    ref_element = LagrangeElement(degree=degree, a=-1, b=1)
    
    a, b = 0, 2
    phys_element = LagrangeElement(degree=degree, a=a, b=b)

    element = phys_element

    x = Symbol('x')
    basis_fs = element.sym_basis
    ps = plot(basis_fs[0], (x, a, b), show=False)
    [ps.append(plot(f, (x, a, b), show=False)[0]) for f in basis_fs[1:]]
    # ps.show()

    for element in [ref_element, phys_element]:
        dim = element.dim
        N = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
               N[i, j] = element.eval_dof(j, element.num_basis[i])
        assert np.linalg.norm(N - np.eye(dim)) < 1E-10
        print 'OK'

        R = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
               R[i, j] = element.eval_dof_riesz(j, element.sym_basis[i])
        diff = np.linalg.norm(R - np.eye(dim))
        print diff
        assert diff < 1E-10
        print 'OK'

    # print ref_element.riesz_basis
    # print phys_element.riesz_basis

    # F : x_hat --> x = a(1-x)/2 + b(1+x)/2
    Finv = (2*x - b - a)/(b-a)
    J = (b-a)/2.

    mapped_riesz = [f.subs(x, Finv)/J
                    for f in ref_element.riesz_basis]
    # print mapped_riesz
    for rmap, r in zip(mapped_riesz, phys_element.riesz_basis):
        print '>', quad(lambdify(x, (rmap-r)**2), [a, b])**0.5
    
    mapped_basis = [f.subs(x, Finv) for f in ref_element.sym_basis]
    # print mapped_basis
    for rmap, r in zip(mapped_basis, phys_element.sym_basis):
        print '>>', quad(lambdify(x, (rmap-r)**2), [a, b])**0.5
    
    coef = np.random.rand(degree+1)
    f = sum(c*x**i for i, c in enumerate(coef))
    f_lambda = lambdify(x, f)
    for element, cell in [(ref_element, (-1, 1)), (phys_element, (a, b))]:
        for i in range(dim):
           print element.eval_dof(j, f_lambda),\
                 element.eval_dof_riesz(j, f),\
                 ref_element.eval_dof_riesz_cell(j, f, cell)
