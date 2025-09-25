import numpy as np


def euler_explicit(
    force, vel_init, m, dt
):  # This function will take in some input ODE and evolve it according to the Euler Explicit rules
    a = force / m
    vel_final = vel_init + a * dt

    return vel_final



######################## INTEGRATION METHODS ###################################################

def riemann(func, lower: float, upper: float, N: float):

    dx = (upper - lower) / N  # step size
    
    integral = 0

    for k in range(1, N):
        integral += func(lower + k * dx)  # left hand riemann sum

    return dx * integral  # return integrated value


def simpson(
    func, lower: float, upper: float, N: int
):  # This function takes in some integrand and its bounds and evaluates it using simpson's rule

    dx = (upper - lower) / N  # determine step size

    integral = func(lower) + func(upper)  # evaluate at the end points

    for k in range(
        1, N, 2
    ):  # loop over odd terms, range(1,N,2) means all values from 1->N with a step size of 2, i,e 1,3,5,7,...,N-1
        integral += 4 * func(lower + k * dx)

    for k in range(2, N, 2):  # loop over even terms
        integral += 2 * func(lower + k * dx)

    return (dx / 3) * integral


def trapezoidal(
    func, lower: float, upper: float, N: int
):  # This computes the integrand of some function using the trapezoidal rule

    dx = (upper - lower) / N  # step size

    integral = 0.5 * func(lower) + 0.5 * func(upper)  # evaluate integral at end points

    for k in range(1, N):  # loop over number of steps, N
        integral += func(
            lower + k * dx
        )  # at each step, append value of function evaluated at a+k*dx

    return dx * integral  # return integral value
