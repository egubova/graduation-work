import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy import abs
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import quad_methods
import warnings
import matplotlib.pylab as pylab
warnings.filterwarnings('ignore')

solvers = {'FR2': quad_methods.FR2, 'HS2': quad_methods.HS2, 'PR2': quad_methods.PR2, 'PRP2019': quad_methods.test,
           'test': quad_methods.PRP2019}
methods = len(solvers)
method_names = np.array(list(solvers))
M = np.array([10, 10, 10, 10, 50, 50, 50, 100, 100, 100])
N = np.array([10, 100, 500, 1000, 10, 500, 1000, 10, 500, 1000])
dimensions = len(M)
circles = 1000
tasks = dimensions * circles
iterations = np.zeros((dimensions, circles, methods))
solution_time = np.zeros((dimensions, circles, methods))

report = PrettyTable()
report.field_names = ["method", "M", "N", "iteration", "norm(gk)", "deviation", "time"]


def save_results(dim, circle, method, m, n, it, gk, xmin, xk, time):
    report.add_row([method_names[method], m, n, it, gk, norm(abs(xmin - xk)), time])
    iterations[dim][circle][method] = it
    solution_time[dim][circle][method] = time


for dim in range(dimensions):
    m = M[dim]
    n = N[dim]
    for circle in range(circles):
        A = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                A[i][j] = random.uniform(0, 1)
        l = random.uniform(0, 1)
        y = np.zeros((m, 1))
        for i in range(m):
            y[i][0] = random.uniform(0, 1)
        I = np.eye(m)
        b = dot((dot(A, A.T) + l * I), y)
        xmin = dot(A.T, y)

        for method in range(methods):
            xk, it, gk, time = solvers[method_names[method]](A, b, l, n)
            save_results(dim, circle, method, m, n, it, gk, xmin, xk, time)

print(report)


for dim in range(dimensions):
    itmin = np.min(iterations[dim, :, :])
    timemin = np.min(solution_time[dim, :, :])
    for method in range(methods):
        for circle in range(circles):
            iterations[dim][circle][method] /= itmin
            solution_time[dim][circle][method] /= timemin


def perfomance(t, method, metric):
    ps = np.zeros((len(t)))
    k = 0
    for tau in t:
        count = 0
        for dim in range(dimensions):
            for circle in range(circles):
                if metric[dim][circle][method] <= tau:
                    count += 1
        ps[k] = count / tasks
        k += 1
    return ps

params = {'legend.fontsize': 'x-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

plt.figure()
t = np.linspace(1, 100, 100000)
line_styles = ['-', '--', '-', '--', '-', '--']
line_colors = ['b', 'g', 'r', 'y', 'm', 'k']

plt.subplot(1, 2, 1)
for method in range(methods):
    plt.plot(t, perfomance(t, method, iterations), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("iterations")
plt.xlabel("t")
plt.ylabel("P(t)")
plt.grid(True)

plt.subplot(1, 2, 2)
for method in range(methods):
    plt.plot(t, perfomance(t, method, solution_time), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("time")
plt.xlabel("t")
plt.ylabel("P(t)")
plt.grid(True)
plt.legend()
plt.show()