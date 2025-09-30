import functions as func
import numpy as np
from scipy.integrate import trapezoid, simpson
import matplotlib.pyplot as plt

############ DEFINITION OF INTEGRAL AND ANALYTIC FUNCTION ######
r2 = 10000; r1 = 1              #Initialize r1,r2
def analytic(r1,r2):            #analytic expression
    term_1 = r2 * (np.pi/2 - np.arcsin(np.sqrt(r1/r2)))     
    term_2 = np.sqrt(r1*(r2-r1))

    return term_1 - term_2

def gamow_integral(r, outer_radius=r2): #integrand
    return np.sqrt(outer_radius/r - 1)

############# CONVERGENCE TO SMALL ANGLE APPROX #################
r1 = 1 
r2 = [10,25,50,75,100,250,500,750,1000,2500,5000,7500,10000] #r2 list to demonstrate convergence

def small_angle_analytic(r1,r2):                             #small-angle approximation analytic expression
    term_1 = r2 * (np.pi/2 - np.sqrt(r1/r2))
    term_2 = np.sqrt(r1*(r2-r1))

    return term_1 - term_2

analytic_list = []                                           #initialize lists to store returned values from each method
riemann_list, simpson_list, trapz_list = [],[],[]
scipy_simpson, scipy_trapz = [], []
for i in range(len(r2)):
    riemann_list.append(func.riemann(gamow_integral, r1, r2[i], 10000))       #loop over r2, determine integral value at each r2
    simpson_list.append(func.simpson(gamow_integral, r1, r2[i], 10000))
    trapz_list.append(func.trapezoidal(gamow_integral, r1, r2[i], 10000))
    analytic_list.append(small_angle_analytic(r1,r2[i]))

    sample_pts = np.linspace(r1,r2[i],10000)                                 #probably a slicker way of doing this, reinitialize list of sample points
    gamow_sampled = gamow_integral(sample_pts)                               #sample integrand

    scipy_simpson.append(simpson(gamow_sampled, sample_pts))                 #pass sample integrand and sample points to scipy
    scipy_trapz.append(trapezoid(gamow_sampled, sample_pts))

fig, axs = plt.subplots(ncols=3,layout='tight',figsize=(12,6))               #plotting
axs[0].scatter(r2, simpson_list, label="Simpson's")
axs[0].scatter(r2, analytic_list, label='Analytic')
axs[0].scatter(r2, scipy_simpson, label="Simpson's, Scipy Implementation",alpha=0.5)

axs[1].scatter(r2, trapz_list, label="Trapezoidal")
axs[1].scatter(r2, analytic_list, label='Analytic')
axs[1].scatter(r2, scipy_trapz, label="Trapezoidal, Scipy Implementation",alpha=0.5)

axs[2].scatter(r2, riemann_list, label="Riemann")
axs[2].scatter(r2, analytic_list, label='Analytic')

axs[0].legend(loc='upper left',fontsize=8)
axs[0].set_xscale("log")

axs[1].legend(loc='upper left',fontsize=8)
axs[1].set_xscale("log")

axs[2].legend(loc='upper left',fontsize=8)
axs[2].set_xscale("log")

fig.supylabel(r"$\gamma$")
fig.supxlabel(r"$\frac{r_2}{r_1}$")
fig.suptitle("Convergence to Small Angle Approximation for Each Method")
plt.savefig("convergence_smallangle.png",dpi=300)

plt.clf()
############ ERROR FOR STEP SIZE #######################################
r2 = 11; r1 = 1 #chosen more or less at random

step_sizes = np.array([0.00001,0.0001,0.001,0.01,0.1,0.25,0.5,1])     #list of step sizes
N_steps = (r2-r1)/step_sizes                                          #determine number of steps for each step size

simpson_error, trapz_error = [],[]
analytic_val = analytic(r1,r2)
for i in range(len(N_steps)):

    simpson_val = func.simpson(gamow_integral, r1, r2, int(N_steps[i]))      #compute integral with certain step size
    trapz_val = func.trapezoidal(gamow_integral, r1, r2, int(N_steps[i]))    

    simpson_error.append(np.abs(analytic_val - simpson_val))                 #compare computed value to analytic value
    trapz_error.append(np.abs(analytic_val - trapz_val))
fig = plt.figure(figsize=(8,6))
plt.scatter(step_sizes, simpson_error, label= "Simpson")                     #plotting
plt.scatter(step_sizes, trapz_error, label="Trapezoidal")
plt.legend()
plt.title("Truncation Error as a Function of Time-step")
plt.xlabel("Time-step")
plt.ylabel("Error")
plt.savefig("error_integration.png",dpi=300)

