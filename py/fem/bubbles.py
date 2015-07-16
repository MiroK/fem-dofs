from __future__ import division
import sys
sys.path.append('../')
import polynomials.legendre_basis as leg
from points import chebyshev_points
from sympy import lambdify, Symbol, integrate
from sympy.plotting import plot
import numpy as np

deg = 10
# Suppose I want to find some polynomial of degree 2 over -1, 1 that has
# 0 boundary values

x = Symbol('x')
poly_set = leg.basis_functions(deg=deg)

# f = a*l0 + b*l1 + c*l2
# Suppose we reure that the L-2 norm is 1
# [f(-1)=0] = [l0(-1), l1(-1), l2(-1) ] [a]
# [f(1)=0 ] = [l0(1), l1(1), l2(1) ] [b]
# [norm=1 ] = [(l0, l0), ...            [c]    
# suppose instead nodality at 0
# [f(0) = 1] = .....

A = np.zeros((deg+1, deg+1))

pts = np.r_[-1, 1, chebyshev_points(deg)[1:-1]]

for col, f in enumerate(poly_set):
    A[:, col] = lambdify(x, poly_set[col], 'numpy')(pts)

# norm
# A[2] = [float(integrate(f*f, (x, -1, 1))) for f in poly_set]
b = np.zeros(deg+1)
b[-1 if deg % 2 else -deg/2] = 1


coefs = np.linalg.solve(A, b)
f = sum(c*f for c, f in zip(coefs, poly_set))
#plot(f, (x, -1, 1))


def bubble_basis(deg):
    assert deg > 0
    if deg == 1:
        return np.array([[0.5, -0.5], [0.5, 0.5]])
    else:
        # Compute the bubble
        # IDEA the shen polynomials are bubble. What 'principle' defines them
        x = Symbol('x')
        poly_set = leg.basis_functions(deg=deg)
        pts = np.r_[-1, 1, chebyshev_points(deg)[1:-1]]
        
        A = np.zeros((deg+1, deg+1))
        for col, f in enumerate(poly_set):
            A[:, col] = lambdify(x, poly_set[col], 'numpy')(pts)
        b = np.zeros(deg+1)
        b[-1 if deg % 2 else -deg/2] = 1

        bubble = np.linalg.solve(A, b)
        # The whole hierarchy
        basis_coefs = np.zeros((len(bubble), len(bubble)))
        basis_coefs[-1] = bubble
        basis_coefs[:deg, :deg] = bubble_basis(deg-1)
        return basis_coefs

deg = 20
alpha = bubble_basis(deg)
poly_basis = leg.basis_functions(len(alpha))
basis = [sum(c*f for c, f in zip(row, poly_basis)) for row in alpha]

ps = plot(basis[0], (x, -1, 1), show=False)
[ps.append(plot(f, (x, -1, 1), show=False)[0]) for f in basis[1:]]
ps.show()

# Plot is nice, but let's make sure that this is really a basis of polynomials
# Any polynomial must be expreesible in it or equivantyl Ax = 0 -> x = 0
pts = chebyshev_points(deg)
A = np.zeros((len(pts), len(basis))) 
for col, f in enumerate(basis):
    A[:, col] = lambdify(x, f, 'numpy')(pts)

print np.linalg.solve(A, np.zeros(len(pts)))
