# Let's see how the (-1, 1)[reference] element matrices of nodal basis scale
# Nodal basis <- some polyset + some dofs

import polynomials
import fem.lagrange
import numpy as np

def scaling(basis, get_points, matrix, deg_range, bc=None):
    # Select matrix
    matrices = {'mass': basis.mass_matrix,
                'stiff': basis.stiffness_matrix,
                'H1': lambda deg: basis.mass_matrix(deg) + basis.stiffness_matrix(deg)
                }
    matrix = matrices[matrix]

    conds = []
    for deg in deg_range:
        # Matrix of polynomials
        mat = matrix(deg)
        # Get the elemenent ...
        dof_set = get_points(deg)
        poly_set = basis.basis_functions(deg)
        element = fem.lagrange.LagrangeElement(poly_set, dof_set)
        # Tranform
        alpha = element.alpha
        mat = alpha.dot(mat.dot(alpha.T))

        # Apply boundary conditions row, col to -1, 1
        if bc == 'LA':
            # -1
            mat[:, 0] = 0; mat[0, :] = 0; mat[0, 0] = 1
            # 1
            mat[:, -1] = 0; mat[-1, :] = 0; mat[-1, -1] = 1
            cond = np.linalg.cond(mat)
        # Compute the condition number from nonzero eigenvalus
        elif bc == 'eigs':
            eigs = np.abs(np.linalg.eigvalsh(mat))
            eigs = eigs[eigs > 1E-13]
            print eigs
            cond = np.max(eigs)/np.min(eigs) if len(eigs) else np.nan
        # No special treatment
        else:
            cond = np.linalg.cond(mat)

        conds.append(cond)

    return conds

# -----------------------------------------------------------------------------

if __name__:
    import matplotlib.pyplot as plt
    from points.points import equidistant_points, chebyshev_points
    from points.points import gauss_legendre_points

    # Basis with color
    basis = {'leg': (polynomials.legendre_basis, 'red'),
             'monom': (polynomials.monomial_basis, 'blue')}
    deg_range = range(1, 21)
    # Matrices with bcs and markers
    matrices = {'mass': None,
                'stiff_LA': 'LA',
                'stiff_eigs': 'eigs',
                'H1': None}
    # Pick one matrix
    matrix = 'stiff_LA'
    bc = matrices[matrix]

    pts = {'eq': (equidistant_points, 'x'),
           'cheb': (chebyshev_points, 'o'),
           'gauss': (gauss_legendre_points, 's')}

    fig, axarr = plt.subplots(1, len(basis))

    for i, base in enumerate(basis):
        base, color = basis[base]
        # Get matrices for all points
        lines = []
        for p in pts:
            points, marker = pts[p]
            cond = scaling(base, points, matrix.split('_')[0], deg_range, bc=bc)
            line, = axarr[i].loglog(deg_range, cond, marker=marker, color=color,
                                    linestyle='--')
            lines.append(line)
        axarr[i].legend(lines, pts.keys(), loc='best')

    plt.show()
