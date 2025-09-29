import numpy as np


###################### ODE METHODS ###############################################################


def state(t, I, dot_I, alpha, omega_0):
    out1 = dot_I
    out2 = -2 * alpha * dot_I - (omega_0**2) * I

    return np.array([out1, out2])


def euler_explicit(
    initial_I, initial_dot_I, start_time, end_time, N, alpha, omega_0
):  # This function will take in some input ODE and evolve it according to the Euler Explicit rules

    delta_t = (end_time - start_time) / N

    t = np.empty(N)
    dot_I = np.empty(N)
    I = np.empty(N)

    t[0] = start_time
    dot_I[0] = initial_dot_I
    I[0] = initial_I

    for i in range(0, N - 1):
        dot_I[i + 1] = (
            dot_I[i] + delta_t * state(t[i], I[i], dot_I[i], alpha, omega_0)[1]
        )
        I[i + 1] = I[i] + delta_t * state(t[i], I[i], dot_I[i], alpha, omega_0)[0]
        t[i + 1] = t[i] + delta_t

    return (I, dot_I, t)


def runge_kutta4(initial_I, initial_dot_I, start_time, end_time, N, alpha, omega_0):

    delta_t = (end_time - start_time) / N

    t = np.empty(N)
    dot_I = np.empty(N)
    I = np.empty(N)

    t[0] = start_time
    dot_I[0] = initial_dot_I
    I[0] = initial_I

    for i in range(0, N - 1):
        k1f = delta_t * state(t[i], I[i], dot_I[i], alpha, omega_0)[0]
        k1g = delta_t * state(t[i], I[i], dot_I[i], alpha, omega_0)[1]

        k2f = (
            delta_t
            * state(
                t[i] + (delta_t / 2),
                I[i] + (k1f / 2),
                dot_I[i] + (k1g / 2),
                alpha,
                omega_0,
            )[0]
        )
        k2g = (
            delta_t
            * state(
                t[i] + (delta_t / 2),
                I[i] + (k1f / 2),
                dot_I[i] + (k1g / 2),
                alpha,
                omega_0,
            )[1]
        )

        k3f = (
            delta_t
            * state(
                t[i] + (delta_t / 2),
                I[i] + (k2f / 2),
                dot_I[i] + (k2g / 2),
                alpha,
                omega_0,
            )[0]
        )
        k3g = (
            delta_t
            * state(
                t[i] + (delta_t / 2),
                I[i] + (k2f / 2),
                dot_I[i] + (k2g / 2),
                alpha,
                omega_0,
            )[1]
        )

        k4f = (
            delta_t
            * state(t[i] + delta_t, I[i] + k3f, dot_I[i] + k3g, alpha, omega_0)[0]
        )
        k4g = (
            delta_t
            * state(t[i] + delta_t, I[i] + k3f, dot_I[i] + k3g, alpha, omega_0)[1]
        )

        I[i + 1] = I[i] + (1 / 6) * (k1f + 2 * k2f + 2 * k3f + k4f)
        dot_I[i + 1] = dot_I[i] + (1 / 6) * (k1g + 2 * k2g + 2 * k3g + k4g)
        t[i + 1] = t[i] + delta_t

    return (I, dot_I, t)


######################## INTEGRATION METHODS ###################################################


def riemann(func, lower: float, upper: float, N: float):
    """
    Compute the definite integral of a function using the left hand Riemann sum.

    Args:
        func: function to integrate
        lower: lower bound of integration
        upper: upper bound of integration
        N: number of steps for integration

    Returns:
        integral: final integrated value

    """
    dx = (upper - lower) / N  # step size

    integral = 0  # initial integral value

    for k in range(1, N):
        integral += func(lower + k * dx)  # left hand riemann sum

    return dx * integral  # return integrated value


def simpson(func, lower: float, upper: float, N: int):
    """
    Compute the definite integral of some function using Simpson's Rule

    Args:
        func: function to integrate
        lower: lower bound of integration
        upper: upper bound of integration
        N: number of steps

    Returns:
        integral: final integrated value

    """
    dx = (upper - lower) / N  # determine step size

    integral = func(lower) + func(upper)  # evaluate at the end points

    for k in range(
        1, N, 2
    ):  # loop over odd terms, range(1,N,2) means all values from 1->N with a step size of 2, i,e 1,3,5,7,...,N-1
        integral += 4 * func(lower + k * dx)

    for k in range(2, N, 2):  # loop over even terms
        integral += 2 * func(lower + k * dx)

    return (dx / 3) * integral  # return integrated value (divided by 3)


def trapezoidal(func, lower: float, upper: float, N: int):
    """
    Compute the definite integral of some function using Trapezoidal Rule

    Args:
        func: function to integrate
        lower: lower bound of integration
        upper: upper bound of integration
        N: number of steps

    Returns:
        integral: final integrated value

    """

    dx = (upper - lower) / N  # step size

    integral = 0.5 * func(lower) + 0.5 * func(upper)  # evaluate integral at end points

    for k in range(1, N):  # loop over number of steps, N
        integral += func(
            lower + k * dx
        )  # at each step, append value of function evaluated at a+k*dx

    return dx * integral  # return integral value
