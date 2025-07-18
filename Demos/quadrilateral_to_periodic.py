# using unconstrained optimization instead of constrained by transforming the target function
import numpy as np
import math as m
import random as r
from scipy import optimize
from scipy.optimize import NonlinearConstraint

target_function_count = 0
constraint_count = 0


# target finction
def T(xy):
    global target_function_count
    target_function_count += 1
    return -3*m.cos(xy[0] - 0.6) + 2*m.cos(xy[1] + 2.)

# constraints
@np.vectorize
def x_max(y):
    global constraint_count
    constraint_count += 1
    return y*0.1+0.9

@np.vectorize
def x_min(y):
    global constraint_count
    constraint_count += 1
    return -y*0.2-1.2

@np.vectorize
def y_max(x):
    global constraint_count
    constraint_count += 1
    return x*0.2+1.3

@np.vectorize
def y_min(x):
    global constraint_count
    constraint_count += 1
    return -x*0.3-1.1

# minimize traditionally
xy_uc = optimize.minimize(T, [0, 0]).x
print('Unconstrained: ', xy_uc, T(xy_uc))
print('target_function_count', target_function_count)
print('constraint_count', constraint_count)
print()
target_function_count = 0
constraint_count = 0

xy_c = optimize.minimize(T, [0, 0], method='COBYLA', constraints = [
    NonlinearConstraint(fun = lambda xy: xy[0] - x_max(xy[1]), lb = -np.inf, ub = 0),
    NonlinearConstraint(fun = lambda xy: xy[0] - x_min(xy[1]), lb = 0, ub = np.inf),
    NonlinearConstraint(fun = lambda xy: xy[1] - y_max(xy[0]), lb = -np.inf, ub = 0),
    NonlinearConstraint(fun = lambda xy: xy[1] - y_min(xy[0]), lb = 0, ub = np.inf),
]).x
print('Constrained: ', xy_c, T(xy_c))
print('Unconstrained: ', xy_uc, T(xy_uc))
print('target_function_count', target_function_count)
print('constraint_count', constraint_count)
print()
target_function_count = 0
constraint_count = 0


# minimize by transforming the R^n into the constraint space
# 1. find the intersections
xmin_ymin = optimize.minimize(lambda xy: (xy[0] - x_min(xy[1]))**2 + (xy[1] - y_min(xy[0]))**2, [0, 0]).x
xmin_ymax = optimize.minimize(lambda xy: (xy[0] - x_min(xy[1]))**2 + (xy[1] - y_max(xy[0]))**2, [0, 0]).x
xmax_ymax = optimize.minimize(lambda xy: (xy[0] - x_max(xy[1]))**2 + (xy[1] - y_max(xy[0]))**2, [0, 0]).x
xmax_ymin = optimize.minimize(lambda xy: (xy[0] - x_max(xy[1]))**2 + (xy[1] - y_min(xy[0]))**2, [0, 0]).x

# 2 find the [-1, 1]x[-1,1] to the intersections transformation
# we can take projective transformation but let's go with bilinear for simplicity (for now)
# ax * x*y + bx * x + cx * y + dx = x_transformed
# ay * x*y + by * x + cy * y + dy = y_transformed
Ax = Ay = [
[-1*-1, -1, -1, 1],
[-1*+1, -1, +1, 1],
[+1*+1, +1, +1, 1],
[+1*-1, +1, -1, 1]]
Bx = [xmin_ymin[0], xmin_ymax[0], xmax_ymax[0], xmax_ymin[0]]
By = [xmin_ymin[1], xmin_ymax[1], xmax_ymax[1], xmax_ymin[1]]
abcd_x = np.linalg.solve(Ax, Bx)
abcd_y = np.linalg.solve(Ay, By)

def transform_square_to_AOI(xy): # AKA g()
    return [abcd_x[0] * xy[0]*xy[1] + abcd_x[1] * xy[0] + abcd_x[2] * xy[1] + abcd_x[3],
            abcd_y[0] * xy[0]*xy[1] + abcd_y[1] * xy[0] + abcd_y[2] * xy[1] + abcd_y[3]]

# 3. choose the (-inf, inf) to (-1, 1) transformation
def transform_all_to_square(xy): # AKA f()
    return [m.sin(xy[0]), m.sin(xy[1])]

xy_t_in_transformed_space = optimize.minimize(lambda xy: T(transform_square_to_AOI(transform_all_to_square(xy))), [0, 0]).x
xy_t = transform_square_to_AOI(transform_all_to_square(xy_t_in_transformed_space))
print('Transformed: ', xy_t, T(xy_t))
print('Unconstrained: ', xy_uc, T(xy_uc))
print('target_function_count', target_function_count)
print('constraint_count', constraint_count)
print()
target_function_count = 0
constraint_count = 0

## output
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker, cm
from numpy import ma

matplotlib.interactive(True)
matplotlib.use('WebAgg')


# plot size
N = 100
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)

# to generate plot data
X, Y = np.meshgrid(x, y)

def discrete_data_from(a_function):
    discrete_data = []
    for y_i in y:
        row_i = []
        for x_j in x:
            row_i += [a_function((x_j, y_i))]
        discrete_data += [row_i]
    return discrete_data

plotable_data = discrete_data_from(T)


# plotting
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
bg = ax.contourf(X, Y, plotable_data, cmap=cm.coolwarm, levels = range(-6, 7, 1))
ax.plot(x_max(y), y, color = 'black', linestyle='dashed', linewidth=1)
ax.plot(x_min(y), y, color = 'black', linestyle='dashed', linewidth=1)
ax.plot(x, y_min(x), color = 'black', linestyle='dashed', linewidth=1)
ax.plot(x, y_max(x), color = 'black', linestyle='dashed', linewidth=1)


# in this demo you can turn some features on and off

# constrained result
if True:
    plt.scatter([xy_c[0]], [xy_c[1]], color='green', marker = 'o')

# intersections
if False:
    plt.scatter([xmin_ymin[0], xmin_ymax[0], xmax_ymax[0], xmax_ymin[0]], [xmin_ymin[1], xmin_ymax[1], xmax_ymax[1], xmax_ymin[1]], color='blue')

# transformed grid
if False:
    grid_xs = []
    grid_ys = []
    for xj in np.linspace(-1, 1, 10):
        for yi in np.linspace(-1, 1, 10):
            transfromed_xy = transform_square_to_AOI([xj, yi])
            grid_xs += [transfromed_xy[0]]
            grid_ys += [transfromed_xy[1]]
    plt.scatter(grid_xs, grid_ys, color='blue')

# transformed result
if True:
    plt.scatter([xy_t[0]], [xy_t[1]], color='orange', marker = 'o')

cbar = fig.colorbar(bg)


# transformed to [-1, 1]^2 plot
if True:
    N = 100
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    transformed_data = discrete_data_from(lambda xy: T(transform_square_to_AOI(xy)))
    fig2, ax2 = plt.subplots()
    ax2.set_aspect('equal', adjustable='box')
    bg2 = ax2.contourf(X, Y, transformed_data, cmap=cm.coolwarm, levels = range(-6, 7, 1))
    period = np.linspace(-m.pi/2, m.pi/2, N)
    cbar2 = fig2.colorbar(bg2)


# fully transformed plot
if True:
    N = 100
    x = np.linspace(-4, 4, N)
    y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(x, y)

    transformed_data = discrete_data_from(lambda xy: T(transform_square_to_AOI(transform_all_to_square(xy))))
    fig3, ax3 = plt.subplots()
    ax3.set_aspect('equal', adjustable='box')
    bg3 = ax3.contourf(X, Y, transformed_data, cmap=cm.coolwarm, levels = range(-6, 7, 1))
    period = np.linspace(-m.pi/2, m.pi/2, N)
    ax3.plot(period, [-m.pi/2 for _ in period], color = 'black', linestyle='dashed', linewidth=1)
    ax3.plot(period, [m.pi/2 for _ in period], color = 'black', linestyle='dashed', linewidth=1)
    ax3.plot([-m.pi/2 for _ in period], period, color = 'black', linestyle='dashed', linewidth=1)
    ax3.plot([m.pi/2 for _ in period], period, color = 'black', linestyle='dashed', linewidth=1)
    cbar3 = fig3.colorbar(bg3)


plt.show()