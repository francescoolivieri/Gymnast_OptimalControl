import numpy as np
import casadi as ca
from trajectory_generation import *
from scipy.linalg import solve_discrete_are

#Regulator Parameters
Q_reg = np.diag([100.0, 100.0, 10.0, 10.0])
R_Reg = np.diag([1.0, 1.0])
Q_T_reg = Q_reg * 2.0 #the final position is twice as important as the intermediate position


def solve_mpc_tracking(x0, x_ref, u_ref, T):
    
    T_hor = 30 # mpc at half T horizon
    dt = 2e-2
    N = x_ref.shape[0]
    
    #Linearize about the generated trajectory(x_ref, u_ref)
    A_list = []
    B_list = []
    
    for t in range(N -1):
        Ac, Bc = Calculate_A_B_matrixes(x_ref[t], u_ref[t])
        Ad, Bd = discretize_linearization(Ac, Bc, dt)
        
        A_list.append(Ad)
        B_list.append(Bd)
    
    x_mpc = np.zeros((ns, T_hor, T))
    #u_mpc = np.zeros((nu, T_hor, T))
        
    x_real = np.zeros(x_ref.shape)    
    u_real = np.zeros(u_ref.shape)
    
    x_real[0, :] = x0
        
    # Desired final equilibrium (used as padding)    
    x_f = np.array([np.pi, 0, 0, 0])
    u_f = np.array([0, 0])
    A_f, B_f = discretize_linearization(*Calculate_A_B_matrixes(x_f, u_f), dt)

    # Stage costs
    Q = np.diag([120.0, 100.0, 0.0001 , 0.0001])  
    R = np.diag([1e-6, 10.0])   
    Q_T = compute_P_inf(A_f, B_f, Q, R)

    
    for t in range(T-1):
        
        # Note: use sliding window no padding so index 0 is always the current reference at time t.
        x_t_mpc = x_real[t, :] - x_ref[0, :]
        
        # Solve MPC at
        u_t_mpc, x_tracked, u_tracked = solver_mpc(x_t_mpc, A_list[:T_hor], B_list[:T_hor], Q, R, Q_T, T_hor)
        
        # Apply correction around the current feedforward input
        u_real[t, :] = u_ref[0, :] + u_t_mpc
            
        # Measure state using MPC result
        x_real[t+1, :] = dynamics(x_real[t, :], u_real[t, :]) 
        
        # Check progress
        if t % 10 == 0:
            error = np.linalg.norm(x_real[t] - x_ref[0])
            print(f"t={t:3d}: !error!={error:.4f}, x={x_real[t, :2]}, u={u_real[t,1]:.2f}")

        # Prepare for next iteration (sliding window)      
        x_ref = np.vstack([x_ref[1:, :], x_f])
        u_ref = np.vstack([u_ref[1:, :], u_f])
        A_list = A_list[1:] + [A_f]
        B_list = B_list[1:] + [B_f]
        
    return x_real, u_real
    


def solver_mpc(x0, A_list, B_list, Q, R, Q_T, T_pred):
    
    Q = ca.DM(Q)
    R = ca.DM(R)
    Q_T = ca.DM(Q_T)
    x0 = ca.DM(x0)
    
    opti = ca.Opti()
    
    X = opti.variable(ns, T_pred)
    U = opti.variable(nu, T_pred)
      
    cost = 0
    
    opti.subject_to( X[:, 0] == x0 )
    
    for t in range(T_pred-1):
        A = ca.DM(A_list[t])
        B = ca.DM(B_list[t])
        
        xt = X[:, t]
        ut = U[:, t]
        
        # quadratic cost
        cost += ca.mtimes([xt.T, Q, xt]) + ca.mtimes([ut.T, R, ut])

        # dynamics constraint
        opti.subject_to(X[:, t + 1] == A @ xt + B @ ut)
    
    
    cost+= ca.mtimes([X[:, -1].T, Q_T, X[:, -1]])
    
    opti.minimize(cost)
    
    ipopt_opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 100,
    }
    opti.solver("ipopt", ipopt_opts)
    
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print("Attention! mpc solver failed:", e)
        exit()
    
    U0 = sol.value(U[:, 0])
    X_opt = sol.value(X)
    U_opt = sol.value(U)

    
    return np.asarray(U0), np.asarray(X_opt).T, np.asarray(U_opt).T




def compute_P_inf(A, B, Q, R):
    P = Q
    
    # simulating infinite horizon
    max_iter = 1000
    tol = 1e-6
    
    for i in range(max_iter):
        P_prev = P
        
        aux1 = R + B.T @ P @ B
        aux2 = B.T @ P @ A
        
        K = -np.linalg.inv(aux1) @ aux2             # Calculate gain
        P = Q + (A.T @ P @ A) + (A.T @ P @ B) @ K   # Riccati equation
        
        # Check if P has stopped changing
        if np.abs(P - P_prev).max() < tol:
            return P
            
    print("P_inf did not converge!!!")
    return P

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
        
        aux1 = R_Reg + B.T @ P @ B
        aux2 = B.T @ P @ A
        
        K_reg[t] = -np.linalg.inv(aux1) @ aux2
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