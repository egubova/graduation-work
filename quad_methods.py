import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy import abs
from time import process_time

maxiter = 100000
prec = 1e-5


def feva(x, A, b, l):
    return (norm(dot(A, x) - b)) ** 2 + l * ((norm(x)) ** 2)


def geva(x, A, b, l):
    return 2 * dot(A.T, (dot(A, x) - b)) + 2 * l * x


def aeva(xk, pk, A, b, l):
    axb = dot(A, xk) - b
    ap = dot(A, pk)
    num = -dot(axb.T, ap) - dot(l, dot(xk.T, pk))
    den = (norm(ap)) ** 2 + l * (norm(pk)) ** 2
    return num / den


def FR2(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        beta = norm(gk) ** 2 / norm(gk_old) ** 2
        pk = -gk + beta * pk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def FR3(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        beta = norm(gk) ** 2 / norm(gk_old) ** 2
        denominator = dot(gk.T, gk)
        numerator = dot(gk.T, pk)
        ratio = numerator / denominator
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def HS2(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old
        beta = dot(gk.T, yk) / dot(pk.T, yk)
        pk = -gk + beta * pk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def HS3(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old
        beta = dot(gk.T, yk) / dot(pk.T, yk)
        denominator = dot(gk.T, gk)
        numerator = dot(gk.T, pk)
        ratio = numerator / denominator
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def PR2(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old
        beta = dot(gk.T, yk) / (norm(gk_old)) ** 2
        pk = -gk + beta * pk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def PR3(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old
        beta = dot(gk.T, yk) / (norm(gk_old)) ** 2
        denominator = dot(gk.T, gk)
        numerator = dot(gk.T, pk)
        ratio = numerator / denominator
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def DY2(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old
        beta = (norm(gk)) ** 2 / dot(pk.T, yk)
        pk = -gk + beta * pk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def DY3(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk = -gk, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old
        beta = (norm(gk)) ** 2 / dot(pk.T, yk)
        denominator = dot(gk.T, gk)
        numerator = dot(gk.T, pk)
        ratio = numerator / denominator
        pk = -gk + beta * pk - beta * ratio * gk
        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def PRP2019(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk_old, xk = -gk, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
        yk = gk - gk_old

        if (norm(gk)) ** 2 > abs(dot(gk.T, gk_old)):
            beta = dot(gk.T, yk) / norm(gk_old)**2
        else:

            numNPRP = (norm(gk)) ** 2 - (norm(gk) / norm(gk_old)) * abs(dot(gk.T, gk_old))
            betaNPRP = numNPRP / (norm(gk_old)) ** 2
            betaFR = (norm(gk)) ** 2 / (norm(gk_old)) ** 2

            lam = dot(yk.T, gk)
            ro = ((norm(gk) / norm(gk_old)) * abs(dot(gk.T, gk_old))) / (norm(gk_old)) ** 2
            G = dot(yk.T, pk) - lam * (dot(pk.T, gk) / (norm(gk)) ** 2)
            tetta = (lam - betaNPRP * G) / dot(G, ro)
            if tetta > 1:
                tetta = 1
            elif tetta < 0:
                tetta = 0
            beta = (1 - tetta) * betaNPRP + tetta * betaFR

        if abs(dot(gk.T, gk_old)) >= 0.2 * (norm(gk)) ** 2:
            pk = -gk
        else:
            numerator = dot(pk.T, gk)
            denominator = (norm(gk)) ** 2
            ratio = numerator / denominator
            pk = -gk + beta * pk - beta * ratio * gk

        it = it + 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)


def test(A, b, l, n):
    x0 = np.zeros((n, 1))
    it = 1
    gk = geva(x0, A, b, l)
    pk, xk_old, xk = -gk, x0, x0
    start_time = process_time()

    while it < maxiter and norm(gk) > prec:
        alpha = aeva(xk, pk, A, b, l)
        xk = xk + alpha * pk
        gk_old = gk
        gk = geva(xk, A, b, l)
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
        omega = dot(pk.T, yk) / norm(gk_old) ** 2
        if abs(dot(gk.T, gk_old)) >= 0.2 * (norm(gk)) ** 2:
            pk = -gk
        else:
            ratio = dot(pk.T, gk) / norm(gk) ** 2
            pk = -omega * gk + beta * pk - omega * beta * ratio * gk
        it += 1

    return xk, it, norm(gk), round(process_time() - start_time, 7)