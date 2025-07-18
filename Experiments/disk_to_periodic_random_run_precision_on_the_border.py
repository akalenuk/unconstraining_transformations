# Using unconstrained optimization instead of constrained by transforming the target function
#
# This runs the comparison between unconstrained optimization that ignores the constraints completely,
# COBYLA method, and Broyden-Fletcher-Goldfarb-Ghanno method on the transformed target function.
#
# All the runs look for the optimum outside (or exactly on) the unit circle. Therefore, the distance 
# from the constrained optimum to the (0, 0) should always be 1. This is validater for both approaches,
# and the total error for both algorithms is being computed.
#
# The output computed:
#   Total count of runs when COBYLA is more precise than BFGS
#   Total count of runs when BFGS is more precise than COBYLA
#   Total count of ties
#   Total error for all COBYLA runs
#   Total error for all BFGS runs

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

COBYLA_total_error = 0.
BFGS_total_error = 0.
COBYLA_is_more_precise_than_BFGS = 0
BFGS_is_more_precise_than_COBYLA = 0
for i in range(NUMBER_OF_RUNS):
    # randomized optimum
    opt_x = 3 * r.random()*2-1
    opt_y = 3 * r.random()*2-1
    # move if from the (-1, 1)^n square
    opt_x += 1 if opt_x > 1 else -1
    opt_y += 1 if opt_y > 1 else -1
    SDF = make_SDF(opt_x, opt_y)

    # minimize traditionally
    om = optimize.minimize(SDF, [0, 0], method='COBYLA', constraints = [
        NonlinearConstraint(fun = unit_disk, lb = -np.inf, ub = 0),
    ])
    xy_c = om.x
    COBYLA_error = abs((xy_c[0]**2 + xy_c[1]**2) - 1.)
    
    # minimize by transforming
    def transform_all_to_disk(xy):
        return [m.cos(xy[0])*m.sin(xy[1]),m.sin(xy[0])*m.sin(xy[1])]

    omt = optimize.minimize(lambda xy: SDF(transform_all_to_disk(xy)), [0, 0])
    xy_t_in_transformed_space = omt.x
    xy_t = transform_all_to_disk(xy_t_in_transformed_space)
    BFGS_error = abs((xy_t[0]**2 + xy_t[1]**2) - 1.)
    
    if COBYLA_error < BFGS_error:
        COBYLA_is_more_precise_than_BFGS += 1
    if BFGS_error < COBYLA_error:
        BFGS_is_more_precise_than_COBYLA += 1
    COBYLA_total_error += COBYLA_error
    BFGS_total_error += BFGS_error

print('COBYLA_is_more_precise_than_BFGS', COBYLA_is_more_precise_than_BFGS)
print('BFGS_is_more_precise_than_BFGS', BFGS_is_more_precise_than_COBYLA)
print('Ties', NUMBER_OF_RUNS - (BFGS_is_more_precise_than_COBYLA + COBYLA_is_more_precise_than_BFGS))
print('COBYLA_total_error', COBYLA_total_error)
print('BFGS_total_error', BFGS_total_error)

