# Using unconstrained optimization instead of constrained by transforming the target function
# This experiment runs the comparison between unconstrained optimization that ignores the constraints completely,
# COBYLA method, and Broyden-Fletcher-Goldfarb-Ghanno method on the transformed target function.
#
# The otput calculated is:
#   A number of function evaluations needed for the COBYLA optimization;
#   A number of function evaluations needed for the BFGS optimization.
#
#   Total count of runs when COBYLA outran BFGS
#   Total count of runs when BFGS outran COBYLA
#   Total count of ties
#   Total count of function evaluations for COBYLA
#   Total count of function evaluations for BFGS
#   Total count of constraint evaluations for COBYLA
#   Total count of constraint evaluations for BFGS

import numpy as np
import math as m
import random as r
from scipy import optimize
from scipy.optimize import NonlinearConstraint

r.seed(0)
NUMBER_OF_RUNS = 1024

def make_SDF(x, y):
    return lambda xy: (xy[0] - x)**2 + (xy[1] - y)**2 - 6

constraint_count = 0
def unit_disk(xy):
    global constraint_count
    constraint_count += 1
    return xy[0]**2 + xy[1]**2 - 1

print('COBYLA FEvs;\t BFGS on transformed T FEvs')

COBYLA_outruns_BFGS = 0
BFGS_outruns_COBYLA = 0
COBYLA_fevs = 0
BFGS_fevs = 0
COBYLA_cevs = 0
BFGS_tevs = 0
transformation_count = 0
for i in range(NUMBER_OF_RUNS):
    SDF = make_SDF(r.random()*2-1, r.random()*2-1)

    # minimize traditionally
    xy_uc = optimize.minimize(SDF, [0, 0]).x

    constraint_count = 0
    om = optimize.minimize(SDF, [0, 0], method='COBYLA', constraints = [
        NonlinearConstraint(fun = unit_disk, lb = -np.inf, ub = 0),
    ])
    xy_c = om.x
    fev_c = om.nfev
    COBYLA_cevs += constraint_count

    # minimize by transforming
    def transform_all_to_disk(xy):
        global transformation_count
        transformation_count +=1
        return [m.cos(xy[0])*m.sin(xy[1]),m.sin(xy[0])*m.sin(xy[1])]

    transformation_count = 0
    omt = optimize.minimize(lambda xy: SDF(transform_all_to_disk(xy)), [0, 0])
    xy_t_in_transformed_space = omt.x
    xy_t = transform_all_to_disk(xy_t_in_transformed_space)
    fev_t = omt.nfev
    BFGS_tevs += transformation_count

    print(fev_c, '\t', fev_t)
    if fev_c < fev_t:
        COBYLA_outruns_BFGS += 1
    elif fev_t < fev_c:
        BFGS_outruns_COBYLA += 1
    COBYLA_fevs += fev_c
    BFGS_fevs += fev_t

print('COBYLA_outruns_BFGS', COBYLA_outruns_BFGS)
print('BFGS_outruns_COBYLA', BFGS_outruns_COBYLA)
print('Ties', NUMBER_OF_RUNS - (BFGS_outruns_COBYLA + COBYLA_outruns_BFGS))
print('COBYLA_fevs', COBYLA_fevs)
print('BFGS_fevs', BFGS_fevs)
print('COBYLA_cevs', COBYLA_cevs)
print('BFGS_tevs', BFGS_tevs)
