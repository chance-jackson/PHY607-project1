import functions as func
import numpy as np
from scipy.constants import epsilon_0, elementary_charge


Z = int(input("Atomic Number: "))
r1 = float(input("Nuclear Radius: "))
E = float(input("Energy of Emitted Alpha Particle: "))

r2 = (1/(4*np.pi*epsilon_0)) * (2*Z*elemetary_charge**2)/(E)


def gamow_integral(r, outer_radius=r2):
    return np.sqrt(outer_radius/r - 1)

riemann = func.riemann(gamow_integral, r1, r2, 100000)
trapz = func.trapezoidal(gamow_integral, r1, r2, 100000)
simp = func.simpson(gamow_integral, r1, r2, 100000)

print(riemann)
print(trapz)
print(simp)
