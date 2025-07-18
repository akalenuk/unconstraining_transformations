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

def ab_through_2_points(xy1, xy2):
    x1, y1 = xy1
    x2, y2 = xy2
    return ((y1 - y2)/(x1 - x2), (x1*y2 - x2*y1)/(x1 - x2))

print('Unconstrained x,\t y,\t value;\t constrained x,\t y,\t value,\t FEvs;\t transformed, x,\t y,\t value,\t FEvs')

for i in range(NUMBER_OF_RUNS):
    SDF = make_SDF(r.random()*2-1, r.random()*2-1)
    #      ^      
    #      |   2  
    #      |      
    #  1   |------>
    #      |      
    #      |   3  
    #             
    x1 = -r.random()
    y1 = r.random() - 0.5
    x2 = r.random()
    y2 = r.random()
    x3 = r.random()
    y3 = -r.random()

    a1,b1 = ab_through_2_points([x1,y1], [x2, y2])
    @np.vectorize
    def y_max(x):
        return x*a1+b1

    a2,b2 = ab_through_2_points([y2,x2], [y3, x3])
    @np.vectorize
    def x_max(y):
        return y*a2+b2

    a3,b3 = ab_through_2_points([x3,y3], [x1, y1])
    @np.vectorize
    def y_min(x):
        return x*a3+b3


    # minimize traditionally
    xy_uc = optimize.minimize(SDF, [0, 0]).x
    om = optimize.minimize(SDF, [0, 0], method='COBYLA', constraints = [
        NonlinearConstraint(fun = lambda xy: xy[0] - x_max(xy[1]), lb = -np.inf, ub = 0),
        NonlinearConstraint(fun = lambda xy: xy[1] - y_max(xy[0]), lb = -np.inf, ub = 0),
        NonlinearConstraint(fun = lambda xy: xy[1] - y_min(xy[0]), lb = 0, ub = np.inf),
    ])
    xy_c = om.x
    fev_c = om.nfev

    # minimize by transforming
    # 1. find the intersections
    ymin_ymax = optimize.minimize(lambda xy: (xy[1] - y_min(xy[0]))**2 + (xy[1] - y_max(xy[0]))**2, [0, 0]).x
    xmax_ymax = optimize.minimize(lambda xy: (xy[0] - x_max(xy[1]))**2 + (xy[1] - y_max(xy[0]))**2, [0, 0]).x
    xmax_ymin = optimize.minimize(lambda xy: (xy[0] - x_max(xy[1]))**2 + (xy[1] - y_min(xy[0]))**2, [0, 0]).x

    (p1, p2, p3) = (ymin_ymax, xmax_ymax, xmax_ymin)

    # 2. define the inner coordinates and the (-1, 1)^N transformation
    def transform_square_to_AOI_in_2simplex(square2, a, b, c): # basic
        uv = [(a + 1.) / 2. for a in square2] # -1, 1 to 0, 1
        u = uv[0]
        v = uv[1]*(1-u)
        return a + (b-a)*u + (c-a)*v

    def transform_square_to_AOI(square2): # AKA g()
        return transform_square_to_AOI_in_2simplex(square2, p1, p2, p3)

    # 3. choose the (-inf, inf) to (-1, 1) transformation
    def transform_all_to_square(xy): # AKA f()
        return [m.sin(xy[0]), m.sin(xy[1])]

    omt = optimize.minimize(lambda xy: SDF(transform_square_to_AOI(transform_all_to_square(xy))), [0, 0])
    xy_t_in_transformed_space = omt.x
    xy_t = transform_square_to_AOI(transform_all_to_square(xy_t_in_transformed_space))
    fev_t = omt.nfev

    print(xy_uc[0], '\t', xy_uc[0], '\t', SDF(xy_uc), '\t', xy_c[0], '\t', xy_c[1], '\t', SDF(xy_c), '\t', fev_c, '\t', xy_t[0], '\t', xy_t[1], '\t', SDF(xy_t), '\t', fev_t)