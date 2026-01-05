import numpy as np
from trajectory_generation import *

#Regulator Parameters
Q_reg = np.diag([100.0, 100.0, 10.0, 10.0])
R_Reg = np.diag([1.0, 1.0])
Q_T_reg = Q_reg * 2.0 #the final position is twice as important as the intermediate position

def solve_LQR_tracking(x_opt, u_opt):
    
    N = x_opt.shape[0]
    
    #Linearize about the optimal trajectory(x_opt, u_opt)
    A_list = []
    B_list = []
    K_reg = [None] * (N-1)
    for t in range(N -1):
        Ac, Bc = Calculate_A_B_matrixes(x_opt[t], u_opt[t])
        Ad, Bd = discretize_linearization(Ac, Bc, dt)
        A_list.append(Ad)
        B_list.append(Bd)
    
    #Backward Riccati Recursion
    P = Q_T_reg.copy()
    
    for t in reversed(range(N - 1)):
        A = A_list[t]
        B = B_list[t]
        
        aux1 = B.T @ P @ B
        aux2 = B.T @ P @ A
        
        K_reg[t] = -np.linalg.inv(R_Reg + aux1) @ aux2
        P = Q_reg + (A.T @ P @ A) + (A.T @ P @ B) @ K_reg[t]
        
    return K_reg

def simulate_tracking(x_opt, u_opt, K_reg, x0_perturbed):
    N_steps = x_opt.shape[0]
    x_track = np.zeros_like(x_opt)
    u_track = np.zeros_like(u_opt)
    
    x_track[0] = x0_perturbed
    
    for t in range(N_steps-1):
        u_track[t] = u_opt[t] + K_reg[t] @ (x_track[t] - x_opt[t])
        x_track[t+1] = dynamics(x_track[t], u_track[t])
    return x_track, u_track

def LQR_tracking(x_opt, u_opt):
    
    x0_perturbed = x_opt[0].copy()
    x0_perturbed[0] += 0.2 #0.2 rad perturbation

    #Compute LQR gains -> step 1 and 2 of the slides
    K_reg_seq = solve_LQR_tracking(x_opt, u_opt)

    #Simulate -> Step 3 of slides
    x_track, u_track = simulate_tracking(x_opt, u_opt, K_reg_seq, x0_perturbed)

    # --- Plotting Tracking Performance ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_ref, x_opt[:, 0], 'k--', label='Reference (Optimal)')
    plt.plot(t_ref, x_track[:, 0], 'r', label='LQR Tracking')
    plt.title('Angle $\\theta_1$ Tracking with Perturbation')
    plt.xlabel('Time [s]'); plt.ylabel('Rad'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_ref[:-1], u_opt[:-1, 0], 'k--', label='Nominal $u$')
    plt.plot(t_ref[:-1], u_track[:-1, 0], 'g', label='LQR Action')
    plt.title('Control Action $\\tau_1$')
    plt.xlabel('Time [s]'); plt.ylabel('Torque'); plt.legend()
    plt.tight_layout()
    plt.show()