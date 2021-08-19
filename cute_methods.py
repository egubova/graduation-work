import pycutest as cute
from time import process_time
import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy import abs
from scipy.optimize import linesearch

maxiter = 100000
prec = 1e-5


def problem(name, params):
    """
    Import test functions from the CUTEst library.

    :param name: CUTEst problem name
    :param params: SIF file parameters to use

    :return: a reference to the Python interface class for this problem (class pycutest.CUTEstProblem)
    """
    if ('ROSENBR' in name) or ('MODBEALE' in name) or ('BROYDN7D' in name):
        return cute.import_problem(name, sifParams={'N/2': int(params)})
    elif (name == 'FMINSURF') or (name == 'LMINSURF') or (name == 'NLMSURF') or (name == 'MSQRTA'):
        return cute.import_problem(name, sifParams={'P': int(params)})
    elif name == 'WOODS':
        return cute.import_problem(name, sifParams={'NS': int(params)})
    elif params:
        return cute.import_problem(name, sifParams={'N': int(params)})
    else:
        return cute.import_problem(name)


def FR2(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        beta = norm(gk) ** 2 / norm(gk_old) ** 2

        # direction
        pk = -gk + beta * pk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def FR3(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        beta = norm(gk) ** 2 / norm(gk_old) ** 2

        # direction
        ratio = dot(gk.T, pk) / dot(gk.T, gk)
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def HS2(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        beta = dot(gk.T, yk) / dot(pk.T, yk)

        # direction
        pk = -gk + beta * pk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def HS3(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        beta = dot(gk.T, yk) / dot(pk.T, yk)

        # direction
        ratio = dot(gk.T, pk) / dot(gk.T, gk)
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def PR2(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        beta = dot(gk.T, yk) / (norm(gk_old)) ** 2

        # direction
        pk = -gk + beta * pk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def PR3(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        beta = dot(gk.T, yk) / (norm(gk_old)) ** 2

        # direction
        ratio = dot(gk.T, pk) / dot(gk.T, gk)
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def DY2(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        beta = (norm(gk)) ** 2 / dot(pk.T, yk)

        # direction
        pk = -gk + beta * pk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def DY3(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        beta = (norm(gk)) ** 2 / dot(pk.T, yk)

        # direction
        ratio = dot(gk.T, pk) / dot(gk.T, gk)
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def DL2017(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    mu, t = 1.5, 0.9
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        sk = xk - xk_old
        yk = gk - gk_old
        num1 = norm(gk) ** 2 - norm(gk) * np.abs(np.dot(gk.T, gk_old)) / norm(gk_old)
        den1 = mu * np.abs(np.dot(gk.T, pk)) - np.dot(pk.T, gk_old)
        beta = num1 / den1 - t * np.dot(gk.T, sk) / np.dot(pk.T, yk)

        # direction
        ratio = np.dot(gk.T, pk) / np.dot(gk.T, gk)
        pk = -gk + beta * pk - beta * ratio * gk
        it += 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def PRP2019(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        if (norm(gk)) ** 2 > abs(dot(gk.T, gk_old)):
            beta = ((norm(gk)) ** 2 - dot(gk.T, gk_old)) / (norm(gk_old)) ** 2
        else:
            numNPRP = (norm(gk)) ** 2 - (norm(gk) / norm(gk_old)) * abs(dot(gk.T, gk_old))
            betaNPRP = numNPRP / (norm(gk_old)) ** 2
            betaFR = (norm(gk)) ** 2 / (norm(gk_old)) ** 2
            ro = ((norm(gk) / norm(gk_old)) * abs(dot(gk.T, gk_old))) / (norm(gk_old)) ** 2
            G = dot(yk.T, pk) - dot(yk.T, gk) * (dot(pk.T, gk) / (norm(gk)) ** 2)
            tetta = (dot(yk.T, gk) - betaNPRP * G) / dot(G, ro)
            if tetta > 1:
                tetta = 1
            elif tetta < 0:
                tetta = 0
            beta = (1 - tetta) * betaNPRP + tetta * betaFR

        # direction
        if abs(dot(gk.T, gk_old)) >= 0.2 * (norm(gk))**2:
            pk = -gk
        else:
            ratio = dot(pk.T, gk) / (norm(gk))**2
            pk = -gk + beta * pk - beta * ratio * gk
        it += 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def test(function, params):
    p = problem(function, params)
    x0 = p.x0
    f0, g0 = p.obj(x0, gradient=True)
    feva, geva, it = 1, 1, 1
    gk, pk, xk_old, xk = g0, -g0, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        # alpha
        fk_old = p.obj(xk_old)
        feva += 1
        alpha, fc, gc = line_search(p, xk, pk, fk_old)
        if alpha is None:
            result = linesearch.line_search_armijo(p.obj, xk, pk, gk, fk_old)
            alpha = result[0]
            feva += result[1]
        else:
            feva += fc
            geva += gc
        if alpha is None:
            break

        # step
        xk_old = xk
        xk = xk + alpha * pk
        gk_old = gk
        gk = p.gradhess(xk)[0]
        geva += 1

        # beta
        yk = gk - gk_old
        if (norm(gk)) ** 2 > abs(dot(gk.T, gk_old)):
            beta = dot(gk.T, yk) / norm(gk_old)**2
        else:
            numNPRP = (norm(gk)) ** 2 - (norm(gk) / norm(gk_old)) * abs(dot(gk.T, gk_old))
            betaNPRP = numNPRP / (norm(gk_old)) ** 2
            betaFR = (norm(gk)) ** 2 / (norm(gk_old)) ** 2
            ro = ((norm(gk) / norm(gk_old)) * abs(dot(gk.T, gk_old))) / (norm(gk_old)) ** 2
            G = dot(yk.T, pk) - dot(yk.T, gk) * (dot(pk.T, gk) / (norm(gk)) ** 2)
            tetta = (dot(yk.T, gk) - betaNPRP * G) / dot(G, ro)
            if tetta > 1:
                tetta = 1
            elif tetta < 0:
                tetta = 0
            beta = (1 - tetta) * betaNPRP + tetta * betaFR

        if beta < 0:
            beta = 0

        # direction
        omega = dot(pk.T, yk) / norm(gk_old)**2

        if abs(dot(gk.T, gk_old)) >= 0.2 * (norm(gk))**2:
            pk = -gk
        else:
            ratio = dot(pk.T, gk) / norm(gk)**2
            pk = -omega * gk + beta * pk - omega * beta * ratio * gk

        it = it + 1

    if params:
        return function, params, feva, geva, it, round(norm(gk), 7), round(process_time() - start_time, 7)
    else:
        return function, cute.problem_properties(function)['n'], feva, geva, it, round(norm(gk), 7), \
               round(process_time() - start_time, 7)


def line_search(p, xk, pk, prev_f, extra_condition=None):
    """
    Find alpha that satisfies strong Wolfe conditions.

    :param p: objective function
    :param xk: starting point
    :param pk: search direction
    :param prev_f: func value at previous x
    :param extra_condition:
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.

    :return:
    alpha : float or None
           Alpha for which ``x_new = x0 + alpha * pk``,
           or None if the line search algorithm did not converge.
       fc : int
           Number of function evaluations made.
       gc : int
           Number of gradient evaluations made.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.
    """

    extra_condition2 = None
    fc, gc = [0], [0]
    gfk, gfk_alpha = [None], [None]

    def derivative_evaluation(alpha):
        gfk[0] = p.gradhess(xk + alpha * pk)[0]
        gc[0] += 1
        gfk_alpha[0] = alpha
        return np.dot(gfk[0], pk)

    def f_evaluation(alpha):
        fc[0] += 1
        return p.obj(xk + alpha * pk)

    alpha_star = scalar_search(f_evaluation, derivative_evaluation, prev_f, extra_condition2)
    return alpha_star, fc[0], gc[0]


def scalar_search(f_evaluation, derivative_evaluation, prev_f, extra_condition=None):
    c1, c2 = 1e-4, 0.1
    f_0 = f_evaluation(0.)
    derivative0 = derivative_evaluation(0.)

    alpha0 = 0
    alpha1 = min(1, 1.01 * 2 * (f_0 - prev_f) / derivative0)
    if alpha1 == 0:
        alpha1 = 1.0
    f_a1 = f_evaluation(alpha1)
    f_a0 = f_0
    derivative_a0 = derivative0
    extra_condition = lambda alpha, f: True

    for i in range(10):
        if (f_a1 > f_0 + c1 * alpha1 * derivative0) or \
                ((f_a1 >= f_a0) and (i > 1)):
            alpha_star = \
                _zoom(alpha0, alpha1, f_a0,
                      f_a1, derivative_a0, f_evaluation, derivative_evaluation,
                      f_0, derivative0, c1, c2, extra_condition)
            break
        derivative_a1 = derivative_evaluation(alpha1)

        if abs(derivative_a1) <= -c2 * derivative0:
            alpha_star = alpha1
            break

        if derivative_a1 >= 0:
            alpha_star = \
                _zoom(alpha1, alpha0, f_a1,
                      f_a0, derivative_a1, f_evaluation, derivative_evaluation,
                      f_0, derivative0, c1, c2, extra_condition)
            break

        alpha2 = 2 * alpha1
        alpha0 = alpha1
        alpha1 = alpha2
        f_a0 = f_a1
        f_a1 = f_evaluation(alpha1)
        derivative_a0 = derivative_a1

    else:
        alpha_star = alpha1
        # warn('The line search algorithm did not converge', LineSearchWarning)

    return alpha_star


def cubicmin(a, fa, fpa, b, fb, c, fc):
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def quadmin(a, fa, fpa, b, fb):
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            d = fa
            c = fpa
            db = b - a
            b = (fb - d - c * db) / (db * db)
            x_min = a - c / (2.0 * b)
        except ArithmeticError:
            return None
    if not np.isfinite(x_min):
        return None
    return x_min


def _zoom(a_lo, a_hi, f_lo, f_hi, derivative_lo,
          f_evaluation, derivative_evaluation, f_0, derivative0, c1, c2, extra_condition):
    i = 0
    delta1 = 0.2  # cubic
    delta2 = 0.1  # quadratic
    f_rec = f_0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        cchk = delta1 * dalpha
        a_j = cubicmin(a_lo, f_lo, derivative_lo, a_hi, f_hi, a_rec, f_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = quadmin(a_lo, f_lo, derivative_lo, a_hi, f_hi)
            if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        f_aj = f_evaluation(a_j)
        if (f_aj > f_0 + c1 * a_j * derivative0) or (f_aj >= f_lo):
            f_rec = f_hi
            a_rec = a_hi
            a_hi = a_j
            f_hi = f_aj
        else:
            derivative_aj = derivative_evaluation(a_j)
            if abs(derivative_aj) <= -c2 * derivative0 and extra_condition(a_j, f_aj):
                a_star = a_j
                break
            if derivative_aj * (a_hi - a_lo) >= 0:
                f_rec = f_hi
                a_rec = a_hi
                a_hi = a_lo
                f_hi = f_lo
            else:
                f_rec = f_lo
                a_rec = a_lo
            a_lo = a_j
            f_lo = f_aj
            derivative_lo = derivative_aj
        i += 1
        if i > 20:
            a_star = None
            break
    return a_star