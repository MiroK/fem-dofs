from __future__ import division
from numpy.polynomial.legendre import leggauss
import numpy as np

#
# Deg+1 points in [-1, 1] to define polynomials of degree deg.
# For all the points generators we require that the endpoints are -1, 1
#

def equidistant_points(deg):
    '''Equally spaced points that define polynomial of degree deg.'''
    assert deg > 0
    return np.linspace(-1, 1, deg+1)


def chebyshev_points(deg):
    '''Chebyshev points that define polynomial of degree deg.'''
    assert deg > 0
    if deg == 1:
        return np.array([-1., 1.])
    else:
        cheb = np.array([np.cos(np.pi*(2*k-1)/(2*(deg+1))) for k in range(2, deg+1)])
        return np.r_[-1., -cheb, 1.]


def gauss_legendre_points(deg):
    '''points of GLL quadrature that define polynomial of degree deg.'''
    assert deg > 0
    if deg == 1:
        return np.array([-1., 1.])
    else:
        gl = leggauss(deg-1)[0]
        return np.r_[-1., gl, 1.]


def map_to_ufc(points, cell=[0, 1]):
    '''Map the points from [-1, 1] to UFCInterval [0, 1]'''
    return np.array([0.5*cell[0]*(1-p) + 0.5*cell[1]*(1+p) for p in points])


def radiate_points(points):
    '''
    Points above are always symmetric around 0. For fun draw some circles...
    '''
    # Points closes to the center 0 but not 0
    l = len(points)
    m = l//2 - 1
    # Compute radius of circles that intersect the line at symmetric points
    radii = []
    for i in reversed(range(m)):
        j = l - 1 - i
        rad = points[j] - points[i]
        radii.append(rad)
    return radii

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pts = gauss_legendre_points(4)
    ufc_pts = map_to_ufc(pts)
    assert np.allclose([ufc_pts[0], ufc_pts[-1]], [0., 1.])

    import matplotlib.pyplot as plt
   
    plt.figure()
    deg = 100
    radii = radiate_points(equidistant_points(deg))
    x = range(1, len(radii)+1)
    plt.plot(x, radii, marker='o', linestyle='--', label='Eqdist')

    radii = radiate_points(gauss_legendre_points(deg))
    plt.plot(x, radii, marker='x', linestyle='--', label='Gauss')

    radii = radiate_points(chebyshev_points(deg))
    plt.plot(x, radii, marker='s', linestyle='--', label='Cheb')

    plt.legend(loc='best')
    plt.show()

