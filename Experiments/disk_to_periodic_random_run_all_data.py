# Using unconstrained optimization instead of constrained by transforming the target function
# This experiment runs the comparison between unconstrained optimization that ignores the constraints completely,
# COBYLA method, and Broyden-Fletcher-Goldfarb-Ghanno method on the transformed target function.
#
# The otput calculated is:
#   x coordinate of the minimum found by the unconstrained optimization;
#   y coordinate of for the minimum found by the unconstrained optimization;
#   T value for the minimum found by the unconstrained optimization;
#
#   x coordinate of the minimum found by the COBYLA optimization;
#   y coordinate of for the minimum found by the COBYLA optimization;
#   T value for the minimum found by the COBYLA optimization;
#   A number of function evaluations needed for the COBYLA optimization;
#
#   x coordinate of the minimum found by the BFGS optimization;
#   y coordinate of for the minimum found by the BFGS optimization;
#   T value for the minimum found by the BFGS optimization;
#   A number of function evaluations needed for the BFGS optimization.

import numpy as np
import math as m
import random as r
from scipy import optimize
from scipy.optimize import NonlinearConstraint

r.seed(0)
NUMBER_OF_RUNS = 1024

def make_SDF(x, y):
    return lambda xy: (xy[0] - x)**2 + (xy[1] - y)**2 - 6

def unit_disk(xy):
    return xy[0]**2 + xy[1]**2 - 1

print('Unconstrained x,\t y,\t value;\t constrained x,\t y,\t value,\t FEvs;\t transformed, x,\t y,\t value,\t FEvs')

for i in range(NUMBER_OF_RUNS):
    SDF = make_SDF(r.random()*2-1, r.random()*2-1)

    # minimize traditionally
    xy_uc = optimize.minimize(SDF, [0, 0]).x

    om = optimize.minimize(SDF, [0, 0], method='COBYLA', constraints = [
        NonlinearConstraint(fun = unit_disk, lb = -np.inf, ub = 0),
    ])
    xy_c = om.x
    fev_c = om.nfev

    # minimize by transforming
    def transform_all_to_disk(xy):
        return [m.cos(xy[0])*m.sin(xy[1]),m.sin(xy[0])*m.sin(xy[1])]

    omt = optimize.minimize(lambda xy: SDF(transform_all_to_disk(xy)), [0, 0])
    xy_t_in_transformed_space = omt.x
    xy_t = transform_all_to_disk(xy_t_in_transformed_space)
    fev_t = omt.nfev

    print(xy_uc[0], '\t', xy_uc[0], '\t', SDF(xy_uc), '\t', xy_c[0], '\t', xy_c[1], '\t', SDF(xy_c), '\t', fev_c, '\t', xy_t[0], '\t', xy_t[1], '\t', SDF(xy_t), '\t', fev_t)
