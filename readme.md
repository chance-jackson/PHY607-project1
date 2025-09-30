NUMERICAL INTEGRATION METHODS IN PYTHON: SOLVING THE RLC CIRCUIT AND ALPHA DECAY INTEGRAL


This code was broadly written to simulate two important physical systems: a series RLC circuit and an alpha particle tunneling out of a nucleus. The code is divided into the functions.py file, which stores the numerical integration methods used to solve these systems, and the two main loop files (integration_loop.py and ODE_loop.py), which produce the results seen in the report. The use of the functions.py file is detailed more carefully below.

Dependencies: numpy, scipy, matplotlib

Usage Directions for functions.py

Functions included: 
state: initialize state vector for use in numerical ODE solvers, which are vectorized.
euler_explicit: compute the solution to an initial value problem using Euler's method
runge_kutta4: compute the solution to an initial value problem using Runge-Kutta 4
riemann: compute a definite integral using Riemann sums
trapezoidal: compute a definite integral using Trapezoidal sums
simpson: compute a definite integral using Simpson's rule

All integral methods accept an arbitrary input function, but ODE methods are linked directly to the RLC series circuit ODE.
