from __future__ import print_function
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
import cute_methods
import warnings
import matplotlib.pylab as pylab
warnings.filterwarnings('ignore')

problem_list = []
with open('problem_list.txt') as file:
    for line in file:
        problem_list.append(line.rstrip().split(';'))
problems = len(problem_list)

solvers = {'FR2': cute_methods.FR2, 'HS2': cute_methods.HS2, 'PR2': cute_methods.PR2, 'PRP2019': cute_methods.PRP2019,
           'test': cute_methods.test}
methods = len(solvers)
method_names = np.array(list(solvers))

feva = np.zeros((problems, methods))
geva = np.zeros((problems, methods))
iterations = np.zeros((problems, methods))
solution_time = np.zeros((problems, methods))
error = np.zeros((problems, methods))

report = PrettyTable()
report.field_names = ["method", "function", "dimension", "feva", "geva", "iterations", "norm(gk)", "time"]


def save_results(method, problem, function, dim, f, g, it, gk, time):
    report.add_row([method_names[method], function, dim, f, g, it, gk, time])
    if gk > 1e-4 or gk is None:
        error[problem][method] = 1
    feva[problem][method] = f
    geva[problem][method] = g
    iterations[problem][method] = it
    solution_time[problem][method] = time


for problem in range(problems):
    function = problem_list[problem][0]
    params = problem_list[problem][1]

    for method in range(methods):
        function, dim, f, g, it, gk, time = solvers[method_names[method]](function, params)
        save_results(method, problem, function, dim, f, g, it, gk, time)
    print(problem+1)

for problem in range(problems):
    for method in range(methods):
        if error[problem][method]:
            feva[problem][method] = np.max(feva)
            geva[problem][method] = np.max(geva)
            iterations[problem][method] = np.max(iterations)
            solution_time[problem][method] = np.max(solution_time)

print(report)

errors = PrettyTable()
errors.field_names = method_names
errors.add_row([sum(error[:, 0]), sum(error[:, 1]), sum(error[:, 2]), sum(error[:, 3]), sum(error[:, 4])])
print(errors)

for problem in range(problems):
    fmin = np.min(feva[problem, :])
    gmin = np.min(geva[problem, :])
    itmin = np.min(iterations[problem, :])
    timemin = np.min(solution_time[problem, :])
    for method in range(methods):
        feva[problem][method] /= fmin
        geva[problem][method] /= gmin
        iterations[problem][method] /= itmin
        solution_time[problem][method] /= timemin


def perfomance(t, method, metric):
    ps = np.zeros((len(t)))
    k = 0
    for tau in t:
        count = 0
        for r in metric[:, method]:
            if r <= tau:
                count += 1
        ps[k] = count / problems
        k += 1
    return ps


params = {'legend.fontsize': 'x-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


#################################################################
########################### [0; 2] ##############################
#################################################################
plt.figure('close to 1')
t = np.linspace(1, 1.5, 100000)
line_styles = ['-', '--', '-', '--', '-', '--']
line_colors = ['b', 'g', 'r', 'y', 'm', 'k']

plt.subplot(2, 2, 1)
for method in range(methods):
    plt.plot(t, perfomance(t, method, feva), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("function evaluations")
plt.ylabel("P(t)")
plt.grid(True)

plt.subplot(2, 2, 2)
for method in range(methods):
    plt.plot(t, perfomance(t, method, geva), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("gradient evaluations")
plt.grid(True)

plt.subplot(2, 2, 3)
for method in range(methods):
    plt.plot(t, perfomance(t, method, iterations), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("iterations")
plt.xlabel("t")
plt.ylabel("P(t)")
plt.grid(True)

plt.subplot(2, 2, 4)
for method in range(methods):
    plt.plot(t, perfomance(t, method, solution_time), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("time")
plt.xlabel("t")
plt.grid(True)
plt.legend()


#################################################################
########################### [0; 100] ############################
#################################################################
plt.figure('close to max')
t = np.linspace(2, 100, 100000)
line_styles = ['-', '--', '-', '--', '-', '--']
line_colors = ['b', 'g', 'r', 'y', 'm', 'k']

plt.subplot(2, 2, 1)
for method in range(methods):
    plt.plot(t, perfomance(t, method, feva), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("function evaluations")
plt.ylabel("P(t)")
plt.grid(True)

plt.subplot(2, 2, 2)
for method in range(methods):
    plt.plot(t, perfomance(t, method, geva), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("gradient evaluations")
plt.grid(True)

plt.subplot(2, 2, 3)
for method in range(methods):
    plt.plot(t, perfomance(t, method, iterations), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("iterations")
plt.xlabel("t")
plt.ylabel("P(t)")
plt.grid(True)

plt.subplot(2, 2, 4)
for method in range(methods):
    plt.plot(t, perfomance(t, method, solution_time), linestyle=line_styles[method], color=line_colors[method],
             label=method_names[method])
plt.title("time")
plt.xlabel("t")
plt.grid(True)
plt.legend()

plt.show()