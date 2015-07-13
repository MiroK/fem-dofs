# Let's see how the (-1, 1)[reference] element matrices of different polynomial 
# basis scale 

import polynomials
import numpy as np

def scaling(basis, matrix, deg_range, bc=None):
    # Select matrix
    matrices = {'mass': basis.mass_matrix,
                'stiff': basis.stiffness_matrix,
                'H1': lambda deg: basis.mass_matrix(deg) + basis.stiffness_matrix(deg)
                }
    matrix = matrices[matrix]

    conds = []
    for deg in deg_range:
        mat = matrix(deg)

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
            cond = np.max(eigs)/np.min(eigs) if len(eigs) else np.nan
        # No special treatment
        else:
            cond = np.linalg.cond(mat)

        conds.append(cond)

    return conds

# -----------------------------------------------------------------------------

if __name__:
    import matplotlib.pyplot as plt
    # Basis with color
    basis = {'leg': (polynomials.legendre_basis, 'red'),
             'monom': (polynomials.monomial_basis, 'blue'),
             'bernstein': (polynomials.bernstein_basis, 'green'),
             'H1_ON': (polynomials.h1ON_basis, 'black'),
             'chebyshev': (polynomials.chebyshev_basis, 'magenta')
            }
    deg_range = range(1, 15)
    # Matrices with bcs and markers
    matrices = {'mass': (None, 'o'),
                'stiff_LA': ('LA', 'x'),
                'stiff_eigs': ('eigs', '*'),
                'H1': (None, 's')}

    fig, axarr = plt.subplots(3, len(basis)/2)

    for i, base_key in enumerate(basis):
        base, color = basis[base_key]
        # Get all matrices for base
        lines = []
        row, col = i/2, i%2
        ax = axarr[row][col]
        for matrix in matrices:
            bc, marker = matrices[matrix]
            cond = scaling(base, matrix.split('_')[0], deg_range, bc=bc)
            line, = ax.loglog(deg_range, cond, marker=marker, color=color,
                              linestyle='--')
            lines.append(line)
        ax.legend(lines, matrices.keys(), loc='best')
        ax.set_title(base_key)

    plt.show()
