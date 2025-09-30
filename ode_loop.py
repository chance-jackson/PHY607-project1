import functions as func
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


R = np.array([0.25,0.5,1,1.5,2,2.5]) #initialize RLC
L = 1
C = 1

alpha = R/(2*L)                      #compute alpha and omega
omega_0 = 1/np.sqrt(L*C)

def analytic_for_R0(t, dot_I_init, omega_0):           #define analytic function for R=0
    return (dot_I_init)/(omega_0)*np.sin(omega_0*t)

end_time = 3 * np.pi
euler_list, rk4_list, rk4_scipy_list = [] ,[] ,[]
for i in range(len(alpha)):
    euler_list.append(func.euler_explicit(0, 10, 0, end_time, 10000, alpha[i], omega_0))   #append solution to list
    rk4_list.append(func.runge_kutta4(0, 10, 0, end_time, 10000, alpha[i], omega_0))
    
    def state_for_scipy(t, u, alpha = alpha[i], omega_0 = omega_0):                        #state vector for scipy, needed to be vectorized
        out1 = u[1]
        out2 = -2 * alpha * u[1] - (omega_0**2) * u[0]

        return np.array([out1, out2])


    rk4_scipy_list.append(solve_ivp(state_for_scipy, (0,end_time), (0,10),method = "RK45", vectorized = False,rtol = 1e-9)) #append scipy solution to list
    

fig, axs = plt.subplots(figsize = (12,8), ncols = 3, sharey = True,layout='tight')      #plotting
fig.suptitle(rf"Solutions for Different Values of Damping Constant")

axs[0].set_title("Euler's Method")
axs[1].set_title("Runge-Kutta 4")
axs[2].set_title("Runge-Kutta 4, Scipy Implementation")

for i in range(len(alpha)):
    axs[0].plot(euler_list[i][2], euler_list[i][0], label = rf'$\zeta$ = {alpha[i]}')
    axs[1].plot(rk4_list[i][2], rk4_list[i][0], label = rf'$\zeta$ = {alpha[i]}')
    axs[2].plot(rk4_scipy_list[i].t, rk4_scipy_list[i].y[0], label = rf'$\zeta$ = {alpha[i]}')
axs[0].set_ylim([-10,10])
axs[1].set_ylim([-10,10])
axs[2].set_ylim([-10,10])
axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
axs[2].legend(loc='upper right')
fig.supylabel("I(t) (A)")
fig.supxlabel("Time (s)")
plt.savefig("damping_cases.png",dpi=300)
fig.clear()

####################### CASE WHERE R=0 ###################################################

R = 0; L = 1; C = 1 #initialize new initial conditions (hehe)

end_time = 30*np.pi

omega_0 = 1/(np.sqrt(L*C))
alpha = R/(2*L)

rk4_sol = func.runge_kutta4(0,10,0,end_time,10000,alpha,omega_0)      #determine solution from each method
euler_sol = func.euler_explicit(0,10,0,end_time,10000,alpha,omega_0)
analytic_sol = analytic_for_R0(euler_sol[2], 10, omega_0)

plt.plot(rk4_sol[2], rk4_sol[0], label='Runge-Kutta 4')               #plot solutions
plt.plot(euler_sol[2], euler_sol[0], label="Euler's Method")
plt.plot(euler_sol[2], analytic_sol,linestyle='dashed', label="Analytic Solution")
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('I(t) (A)')
plt.title("Comparison to Analytic Solution for the R=0 Case")
plt.savefig("comparison.png", dpi=300)

plt.clf()

##################### STEP SIZE VALIDATION ###############################################
# For this, I'll again treat the case where R=0, and we'll see how my error grows with step size for both methods

R = 0; L = 1; C = 1 #initialize

end_time = 10

step_sizes = np.array([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])  #vary step size
N_steps = end_time/step_sizes                                     #find number of steps

omega_0 = 1/(np.sqrt(L*C))
alpha = R/(2*L)

rk4_error, euler_error = [], []

for i in range(len(step_sizes)):
    rk4_sol = func.runge_kutta4(0,10,0,end_time,int(N_steps[i]),alpha,omega_0)        #determine solution
    euler_sol = func.euler_explicit(0,10,0,end_time,int(N_steps[i]),alpha,omega_0)
    analytic_sol = analytic_for_R0(euler_sol[2], 10, omega_0)
    
    rk4_error.append(np.mean(np.abs(analytic_sol - rk4_sol[0])))                      #append MAE to list
    euler_error.append(np.mean(np.abs(analytic_sol - euler_sol[0])))

fig = plt.figure(figsize=(8,6))                                                       #plotting
plt.scatter(step_sizes, rk4_error, label='Runge-Kutta 4')
plt.scatter(step_sizes, euler_error, label="Euler's Method")
plt.legend(loc='best')
plt.xlabel('Time-Step')
plt.ylabel('MAE')
plt.title("Global Truncation Error as a Function of Time-Step")
plt.savefig("error.png",dpi=300)

