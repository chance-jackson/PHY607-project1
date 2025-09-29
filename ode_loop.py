import functions as func
import matplotlib.pyplot as plt
import numpy as np

R = np.array([0.5,1,1.5,2,2.5])
L = 1
C = 1

alpha = R/(2*L)
omega_0 = 1/np.sqrt(L*C)

end_time = 3 * np.pi

for i in range(len(alpha)):
    euler_method = func.euler_explicit(10, 0, 0, end_time, 10000, alpha[i], omega_0)
    rk4 = func.runge_kutta4(10, 0, 0, end_time, 10000, alpha[i], omega_0)

fig, axs = plt.subplots(figsize = (6,8), ncols = 2, sharey = True)
fig.suptitle(rf"$\alpha$={alpha}, $\omega_0$ = {omega_0}")

axs[0].set_title("Euler's Method")
axs[1].set_title("Runge-Kutta 4")

for i in range(len(alpha)):
    axs[0].plot(euler_method[2][i], euler_method[0][i], label = rf'$\zeta$ = {alpha[i]}')
    axs[1].plot(rk4[2][i], rk4[0][i], label = rf'$\zeta$ = {alpha[i]}')

axs[0].set_ylim([-10,10])
axs[1].set_ylim([-10,10])
axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
plt.savefig("test.png",dpi=300)
