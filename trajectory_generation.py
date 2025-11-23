import numpy as np
import matplotlib.pyplot as plt
from dynamics import *


def define_two_equilibria(theta1_e, theta2_e):
    
    #Define the state
    x_e = np.array([theta1_e, theta2_e, 0, 0])
    
    #In the equilibrium the first and secund derivatives are 0
    #So the dynamic equation becomes -> G(theta1, theta2) = torque
    _,_,G_num,_ = set_params(1)
    u_e = G_num.subs({theta1: theta1_e, theta2: theta2_e})

    
    return x_e, u_e

#Not all configurations are possible with only one torque
x_e1, u_e1 = define_two_equilibria(0.3398369, -1.9106332)
x_e2, u_e2 = define_two_equilibria(-0.3398369, 1.9106332)

print("x_e1: ", x_e1, "\n")
print("u_e1: ", u_e1, "\n")
print("x_e2: ", x_e2, "\n")
print("u_e2: ", u_e2, "\n")