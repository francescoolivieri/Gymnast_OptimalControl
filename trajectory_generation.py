import numpy as np
import matplotlib.pyplot as plt
from dynamics import *
import sympy as sp

T = 4.0 #Total time in secods
N = int(T / dt) + 1 # Number of steps (including the initial state)
nu = 2 #Dimension of control vector
nx = 4 #Dimension of state vector
N_opt = N * nx + (N-1) * nu #Total number of optimization values

#   Cost Weights    # -> pq estes valores
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([0.1, 0.1])
Q_T = np.diag([1000.0, 1000.0, 100.0, 100.0])


def define_two_equilibria(theta1_e, theta2_e):
    
    #Define the state
    x_e = np.array([theta1_e, theta2_e, 0, 0])
    
    #In the equilibrium the first and secund derivatives are 0
    #So the dynamic equation becomes -> G(theta1, theta2) = torque
    _,_,G_num,_ = set_params(1)
    u_e = G_num.subs({theta1: theta1_e, theta2: theta2_e})

    #Converting from 2*1 SymPy Matrix to a (2, 1) Numpy array
    
    u_e = np.array(u_e, dtype=float).reshape(-1)
    
    return x_e, u_e

#Not all configurations are possible with only one torque
x_e1, u_e1 = define_two_equilibria(0.3398369, -1.9106332)
x_e2, u_e2 = define_two_equilibria(-0.3398369, 1.9106332)

print("x_e1: ", x_e1, "\n")
print("u_e1: ", u_e1, "\n")
print("x_e2: ", x_e2, "\n")
print("u_e2: ", u_e2, "\n")

def define_reference_curve(T, x_e1, x_e2, u_e1, u_e2):
    
    #Number of steps based on the dt
    N = int(T / dt) + 1
    t_ref = np.linspace(0.0, T, N)
    
    x_ref = np.zeros((N, 4))
    u_ref = np.zeros((N, 2))
    
    # times for constant-transition-constant
    t1 = T/4.0
    t2 = 3.0 * T / 4.0
    trans_len = t2 - t1
    
    for k, tk in enumerate(t_ref):
        if tk <= t1:
            x_ref[k] = x_e1
            u_ref[k] = u_e1
            
        elif tk >= t2:
            x_ref[k] = x_e2
            u_ref[k] = u_e2
            
        else:
            #s as a normalized interpolation parameter
            s = (tk - t1) / trans_len
            
            #Result of the derivative of s wrt tk
            sdot = 1.0 / trans_len
            
            #interpolation between the two states
            theta = (1 - s) * x_e1[:2] + s * x_e2[:2]
            
            theta_dot = sdot * (x_e2[:2] - x_e1[:2])
            
            x_ref[k, :2] = theta
            x_ref[k, 2:] = theta_dot
            
            #interpolation between the two states
            u_ref[k] = (1 - s) * u_e1 + s * u_e2
            
    return t_ref, x_ref, u_ref

t_ref, x_ref, u_ref = define_reference_curve(T, x_e1, x_e2, u_e1, u_e2)

print("x_ref[0]   =", x_ref[0])        # should be close to x_e1
print("u_ref[0]   =", u_ref[0])        # close to u_e1
print("x_ref[-1]  =", x_ref[-1])       # close to x_e2
print("u_ref[-1]  =", u_ref[-1])       # close to u_e2

# Angles (in degrees)
plt.figure()
plt.plot(t_ref, x_ref[:, 0] * 180/np.pi, label=r'$\theta_1$')
plt.plot(t_ref, x_ref[:, 1] * 180/np.pi, label=r'$\theta_2$')
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.title('Reference Joint Angles')
plt.legend()
plt.grid(True)
plt.show()

# Torque on joint 2
plt.figure()
plt.plot(t_ref, u_ref[:, 1])
plt.xlabel('Time [s]')
plt.ylabel('Torque on joint 2 [Nm]')
plt.title('Reference Torque')
plt.grid(True)
plt.show()

def objective_function(Z, N, nx, nu, x_e2, u_e2, Q, Rm, Q_T):
    
    J = 0.0
    
    #running the cost
    for t in range(N - 1):
        x_start = t * (nx + nu)
        u_start = x_start + nx
        
        x_t = Z[x_start : x_start + nx]
        u_t = Z[u_start : u_start + nu]
        
        #L(x_t, u_t) -> confirmar se é x_e2 e u_e2
        state_dev = x_t - x_e2
        control_dev = u_t - u_e2
        
        #Stage_Cost
        J += 0.5 * np.dot(state_dev, Q @ state_dev)
        J += 0.5 * np.dot(control_dev, R @ control_dev)
        
    #Terminal Cost
    x_T_start = (N - 1) * (nx + nu)
    x_T = Z[x_T_start : x_T_start + nx]
    
    #Phi(x_T)
    state_dev_T = x_T - x_e2
    J += 0.5 * np.dot(state_dev_T, Q_T @ state_dev_T)
    
    return J


