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
    return -3*m.cos(xy[0] - 0.8) + 2*m.cos(xy[1] + 1.0)

# constraint T with the unit circle so x^2 + y^2 < 1
def unit_disk(xy):
    global constraint_count
    constraint_count += 1
    return xy[0]**2 + xy[1]**2 - 1

# minimize traditionally
xy_uc = optimize.minimize(T, [0, 0]).x
print('Unconstrained: ', xy_uc, T(xy_uc))
print('target_function_count', target_function_count)
print('constraint_count', constraint_count)
print()
target_function_count = 0
constraint_count = 0

xy_c = optimize.minimize(T, [0, 0], method='COBYLA', constraints = [
    NonlinearConstraint(fun = unit_disk, lb = -np.inf, ub = 0),
]).x
print('Constrained: ', xy_c, T(xy_c))
print('Unconstrained: ', xy_uc, T(xy_uc))
print('target_function_count', target_function_count)
print('constraint_count', constraint_count)
print()
target_function_count = 0
constraint_count = 0


# minimize by transforming the R^n into the constraint space
# the combined (-inf, inf)^2 to (-1, 1)^2 and (-1, 1)^2 to the circle transformation
def transform_all_to_disk(xy):
    return [m.cos(xy[0])*m.sin(xy[1]),m.sin(xy[0])*m.sin(xy[1])]

xy_t_in_transformed_space = optimize.minimize(lambda xy: T(transform_all_to_disk(xy)), [0, 0]).x
xy_t = transform_all_to_disk(xy_t_in_transformed_space)
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
plotable_circle_data = discrete_data_from(lambda xy: xy[0]**2 + xy[1]**2 - 1)

# plotting
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
bg = ax.contourf(X, Y, plotable_data, cmap=cm.coolwarm, levels = range(-6, 7, 1))
fg = ax.contour(X, Y, plotable_circle_data, levels = [0], colors = ['0.1'], linewidths = [2.0])


# in this demo you can turn some features on and off

# constrained result
if True:
    plt.scatter([xy_c[0]], [xy_c[1]], color='green', marker = 'o')

# intersections
if False:
    plt.scatter([xmin_ymin[0], xmin_ymax[0], xmax_ymax[0], xmax_ymin[0]], [xmin_ymin[1], xmin_ymax[1], xmax_ymax[1], xmax_ymin[1]], color='blue')

# transformed result
if True:
    plt.scatter([xy_t[0]], [xy_t[1]], color='orange', marker = 'o')

cbar = fig.colorbar(bg)


# fully transformed plot
if True:
    N = 100
    x = np.linspace(-4, 4, N)
    y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(x, y)

    transformed_data = discrete_data_from(lambda xy: T(transform_all_to_disk(xy)))
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