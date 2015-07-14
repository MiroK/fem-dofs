# See how choice of nodes effects error of solution and condition number of
# system

import argparse
from fem.fem_conditioning import solve
from numpy import savetxt
from fem.utils import colors 
import os

parser = argparse.ArgumentParser()
parser.add_argument('-mode', nargs=1, choices=('convergence', 'condition'),
                    required=True)
parser.add_argument('-points', nargs=1, choices=('eq', 'cheb', 'gauss', 'all'),
                    required=True)
parser.add_argument('-degree', nargs=1, type=int,
                    required=True)
parser.add_argument('-solution', nargs=1, choices=('sine', 'delta'),
                    default=['sine'])
parser.add_argument('-solution_param', nargs=1, type=float, default=[5],
                    help='This is sine(pi*w) or exp(-x**2/2/w)/sqrt(2*pi*w)')

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    args = parser.parse_args()
    mode = args.mode[0]
    points = ['eq', 'cheb', 'gauss'] if args.points[0] == 'all' else args.points
    degree = args.degree[0]
    solution = args.solution[0]
    w = args.solution_param[0]
   
    args = vars(args)
    for _points in points:
        args['points'] = _points
        # Save here
        f_name = '_'.join(map(lambda kv: ':'.join([kv[0], str(kv[1][0])]),
                              args.iteritems()))
        root = './results'
        if not os.path.exists(root): os.mkdir(root)
        f_name = os.path.join(root, f_name)
        
        print  colors['blue'] % ('Computing %s' % f_name)
        data, header = solve(mode=mode, points=_points, degree=degree, solution=solution, w=w)
        savetxt(f_name, data, header=header)
        print
