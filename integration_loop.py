import functions as func
import numpy as np

def function(x):
    return x**3 + 1

riemann = func.riemann(function, 0, 1, 100000)
trapz = func.trapezoidal(function, 0, 1, 100000)
simp = func.simpson(function, 0, 1, 100000)

print(riemann)
print(trapz)
print(simp)
