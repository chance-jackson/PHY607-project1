import numpy as np

#def initial_conditions():
#    pos_x, vel_x = input("Enter Pos/Vel: ").split(",")
#    m, k = input("Enter m/k: ").split(",")
#    dt = input("Enter time-step: ")
#    T = input("Enter sim time: ")

#    return float(pos_x), float(vel_x), float(m), float(k), float(dt), float(T)


def update_vel(force, vel_init, m, dt):
    a = force/m
    vel_final = vel_init + a * dt

    return vel_final

def update_pos(vel, pos_init, dt):
    pos_final = pos_init + vel*dt

    return pos_final

def total_energy(vel, pos, k, m):
    T = (1/2) * m * (vel ** 2)
    U = (1/2) * k * (pos ** 2)

    return T+U
