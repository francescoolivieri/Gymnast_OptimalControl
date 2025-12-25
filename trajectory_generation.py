import numpy as np
import matplotlib.pyplot as plt
from dynamics import *
import sympy as sp
from scipy.optimize import root


T = 4.0 #Total time in secods
N = int(T / dt) + 1 # Number of steps (including the initial state)
nu = 2 #Dimension of control vector
nx = 4 #Dimension of state vector
N_opt = N * nx + (N-1) * nu #Total number of optimization values

#   Cost Weights
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([0.1, 0.1])
Q_T = np.diag([200, 200, 50, 50])

def compute_equilibrium(u_target, theta_guess):
    #Equation yields G(theta1, theta2) = u_target
    
    _,_,G_num,_ = set_params(1)
    G_func = sp.lambdify((theta1, theta2), G_num, 'numpy')# Lambdify changes from symbolic to numerical function 

    def residual(theta): #We want to drag the residual to 0
        g = np.array(G_func(theta[0], theta[1]), dtype=float).reshape(-1)
        return g - u_target
    
    sol = root(residual, theta_guess, method='hybr')
    if not sol.success:
        raise RuntimeError("Root finder failed: " + sol.message)
    
    theta_sol = sol.x
    x_e = np.array([theta_sol[0], theta_sol[1], 0.0, 0.0])
    u_e = np.array(u_target, dtype=float).reshape(-1)
    return x_e, u_e

def define_reference_piecewise(T, x_e1, x_e2, u_e1, u_e2):
    """
    Build a reference with two constant segments: x_e1 for t < T/2, x_e2 for t ≥ T/2.
    """
    N = int(T / dt) + 1
    t_ref = np.linspace(0.0, T, N)
    x_ref = np.zeros((N, 4))
    u_ref = np.zeros((N, 2))

    for k, tk in enumerate(t_ref):
        if tk < T / 2.0:
            x_ref[k] = x_e1
            u_ref[k] = u_e1
        else:
            x_ref[k] = x_e2
            u_ref[k] = u_e2

    return t_ref, x_ref, u_ref

#Targeted Angle: approx 20 degrees ~ 0.35 radians
u_target1 = np.array([0.5, 0.5])
u_target2 = np.array([-0.5, -0.5])
theta_guess1 = (0.35, -0.35)
theta_guess2 = (-0.35, 0.35)

x_e1, u_e1 = compute_equilibrium(u_target1, theta_guess1)
x_e2, u_e2 = compute_equilibrium(u_target2, theta_guess2)

print("First equilibrium x_e1 and u_e1 = ", x_e1, u_e1)
print("Second equilibrium x_e2 and u_e2 = ", x_e2, u_e2)

t_ref, x_ref, u_ref = define_reference_piecewise(T, x_e1, x_e2, u_e1, u_e2)

def simulate_open_loop(x0, u_traj):
    N = u_traj.shape[0]
    x_traj = np.zeros((N, nx))
    x_traj[0] = x0
    for t in range(N-1):
        x_traj[t+1] = dynamics(x_traj[t], u_traj[t])  # uses your RK4 dynamics
    return x_traj

def derivatives_Cost(x, x_ref, u, u_ref, Q, R, Q_T=None, terminal=False):
    
    if not terminal:
        #Stage Cost
        l_t = float((x - x_ref).T @ Q @ (x - x_ref) + (u - u_ref).T @ R @ (u - u_ref))
        
        #First derivatives of Stage cost
        grad_l_t_x = 2 * Q @ (x - x_ref)
        grad_l_t_u = 2 * R @ (u - u_ref)
        
        #Second derivative Stage Cost
        hess_l_t_x = 2 * Q
        hess_l_t_u = 2 * R
        
        return l_t, grad_l_t_x, grad_l_t_u, hess_l_t_x, hess_l_t_u
    else:
        #Terminal Cost
        l_T = float((x - x_ref).T @ Q_T @ (x - x_ref))

        #First derivative of Total cost
        grad_l_T_x = 2 * Q_T @ (x - x_ref)
        
        #Second derivative Total Cost
        hess_l_T_x = 2 * Q_T
        
        return l_T, grad_l_T_x, hess_l_T_x

def stage_blocks_and_affine(x, u, x_ref, u_ref, Q, R, A, B, lambda_next):
    #implement in those iterations a “first-order” update Slide 15 (ppt8)-> impllement second order on the late iteractions
    # Stage derivatives
    _, grad_x, grad_u, hess_xx, hess_uu = derivatives_Cost(x, x_ref, u, u_ref, Q, R, terminal=False)

    # Quadratic blocks (first-order regularization; S=0 for tracking)
    Q_t = hess_xx
    R_t = hess_uu
    S_t = np.zeros((u.shape[0], x.shape[0])) #Because the cost has no cross terms

    # Affine terms (linear parts)
    q_t = grad_x
    r_t = grad_u
    return Q_t, R_t, S_t, q_t, r_t

def terminal_blocks(x_T, x_ref_T, Q_T):
    
    _, grad_xT, hess_xx = derivatives_Cost(x_T, x_ref_T, np.zeros(nu), np.zeros(nu), Q=None, R=None, Q_T=Q_T, terminal=True)
    Q_T = hess_xx
    q_t = grad_xT
    return Q_T, q_t   # (Q_T_block, q_T)


    #Q_t = hess_l_t_x[?] + (need yo calculate 11 f)*lam_kplus1
    #R_t = hess_l_t_u[?] + (need to calculate the hessian of the system position 22)*lam_kplus1
    #S_t = (need to calculate the hessian of l_t position 12) + (need to calculate the hessian of the system pos 12)*lam_kplus1
    #Q_T = hess_l_T_x

def compute_costate_trajectory(x_traj, u_traj, x_ref, u_ref):
    N = x_traj.shape[0]
    lambda_seq = [np.zeros(nx) for _ in range(N)]
    
    #Terminal Trajectory
    _, grad_xT, _ = derivatives_Cost(x_traj[-1], x_ref[-1], np.zeros(nu), np.zeros(nu), Q=None, R=None, Q_T=Q_T, terminal=True)
    lambda_seq[-1] = grad_xT
    
    #Backward Pass
    for t in range(N-2, -1, -1):
        # 1. Get continuous matrices
        A_c, _ = Calculate_A_B_matrixes(x_traj[t], u_traj[t])
        
        # 2. Discretize A (Critical Step)
        A_d = np.eye(A_c.shape[0]) + A_c * dt
        
        # 3. Get Cost Gradient
        _, grad_x, _, _, _ = derivatives_Cost(x_traj[t], x_ref[t], u_traj[t], u_ref[t], Q, R, terminal=False)
        
        # 4. Recursion
        lambda_seq[t] = grad_x + A_d.T @ lambda_seq[t+1]
    return lambda_seq

def discretize_linearization(Ac, Bc, dt):
    Ad = np.eye(Ac.shape[0]) + dt * Ac
    Bd = dt * Bc
    return Ad, Bd

def build_stage_lists(x_traj, u_traj, x_ref, u_ref, lambda_seq): #->>Check this function
    N = x_traj.shape[0]
    A_list, B_list = [], []
    Q_list, R_list, S_list = [], [], []
    q_list, r_list = [], []
    for t in range(N-1):
        A_c, B_c = Calculate_A_B_matrixes(x_traj[t], u_traj[t])
        A_t, B_t = discretize_linearization(A_c, B_c, dt)
        Q_t, R_t, S_t, q_t, r_t = stage_blocks_and_affine(
            x_traj[t], u_traj[t], x_ref[t], u_ref[t], Q, R, A_t, B_t, lambda_seq[t+1]
        )
        A_list.append(A_t); B_list.append(B_t)
        Q_list.append(Q_t); R_list.append(R_t); S_list.append(S_t)
        q_list.append(q_t); r_list.append(r_t)
    Q_T_block, q_T = terminal_blocks(x_traj[-1], x_ref[-1], Q_T)
    return A_list, B_list, Q_list, R_list, S_list, q_list, r_list, Q_T_block, q_T

def calculate_K_and_sigma(A_list, B_list, Q_list, R_list, S_list, q_list, r_list, Q_T_block, q_T):
    
    Tsteps = len(A_list)
    P = Q_T_block.copy()  # terminal quadratic
    p = q_T.copy()        # terminal linear

    K = [None] * Tsteps
    sigma = [None] * Tsteps

    for t in reversed(range(Tsteps)): #Riccati Recursion 
        A = A_list[t]; B = B_list[t]
        Q = Q_list[t]; R = R_list[t]; S = S_list[t]
        q = q_list[t]; r = r_list[t]

        G = R + B.T @ P @ B
        F = S + B.T @ P @ A
        g = r + B.T @ p

        # Numerically stable
        K_t = -np.linalg.solve(G, F)
        sigma_t = -np.linalg.solve(G, g)

        # Compact Riccati updates (use K_t, sigma_t here)
        P = Q + A.T @ P @ A - K_t.T @ G @ K_t
        p = q + A.T @ p - K_t.T @ G @ sigma_t

        K[t] = K_t
        sigma[t] = sigma_t

    return K, sigma

def forward_closed_loop_update(x_traj, u_traj, K, sigma, gamma=1.0):
    N =  x_traj.shape[0]
    x_new = x_traj.copy()
    u_new = u_traj.copy()
    dx = np.zeros(nx)
    
    for t in range(N-1):
        dx = x_new[t] - x_traj[t]
        u_new[t] = u_traj[t] + K[t] @ dx + gamma * sigma[t]
        x_new[t+1] = dynamics(x_new[t], u_new[t])
        
    return x_new, u_new
    
def newton_Algorithm(x0, x_ref, u_ref, max_iters, tol=1e-6, beta=0.7, c=0.5, gamma_0=1.0):
    
    #Initialize a feasible trajectory given the input
    u_traj = u_ref.copy()
    x_traj = simulate_open_loop(x0, u_traj)
    
    for k in range(max_iters):
        
        #Step 1
        lambda_seq = compute_costate_trajectory(x_traj, u_traj, x_ref, u_ref)
    
        # Stage lists
        A_list, B_list, Q_list, R_list, S_list, q_list, r_list, Q_T_block, q_T = \
        build_stage_lists(x_traj, u_traj, x_ref, u_ref, lambda_seq)
        
        #Riccati -> The second backward pass is within the function
        K, sigma = calculate_K_and_sigma(A_list, B_list, Q_list, R_list, S_list, q_list, r_list, Q_T_block, q_T)
    
        gamma_i = gamma_0
        max_line_search_iters = 20
        i = 0
        
        # Initial forward pass and check -> RHS eq
        #But here the formula is with the derivative of the cost times the direction 
        target_cost_k = cost_k + c * gamma_i * directional_derivative
        
        #Forward loop is within the function    
        x_new, u_new = forward_closed_loop_update(x_traj, u_traj, K, sigma, gamma=gamma_i)

        cost_new = total_cost(x_new, u_new, x_ref, u_ref, Q, R, Q_T)

        while cost_new > target_cost_k and i < max_line_search_iters:
            gamma_i *= beta
            target_cost_k = cost_k + c * gamma_i * directional_derivative

            # Re-run forward pass with new gamma
            x_new, u_new = forward_closed_loop_update(x_traj, u_traj, K, sigma, gamma=gamma_i)
            cost_new = total_cost(x_new, u_new, x_ref, u_ref, Q, R, Q_T)

            i += 1
            
        if i == max_line_search_iters:
            print(f"Line search failed to find a suitable step size at iter {k}. Stopping.")
            break
        
        
        delta = np.max(np.abs(x_new - x_traj)) + np.max(np.abs(u_new - u_traj))
        x_traj, u_traj = x_new, u_new
        if delta < tol:
            break
        
    return x_traj, u_traj, K, sigma

def total_cost(x_traj, u_traj, x_ref, u_ref, Q, R, Q_T):
    
    """Computes the total cost
    """
    N = x_traj.shape[0]
    cost = 0.0
    
    for t in range(N - 1): #Stage Cost (t=0 to N-2)
        x = x_traj[t]
        u = u_traj[t]
        x_r = x_ref[t]
        u_r = u_ref[t]
        
        cost += (x - x_r).T @ Q @ (x - x_r)
        cost += (u - u_r).T @ R @ (u - u_r)

    #Terminal Cost (t = N-1)
    x_T = x_traj[-1]
    x_r_T = x_ref[-1]
    cost += (x_T - x_r_T).T @ Q_T @ (x_T - x_r_T)
    
    return cost
    
# Choose initial state (e.g., start at first equilibrium)
x0 = x_e1.copy()
    
# Run Newton
x_opt, u_opt, K_seq, sigma_seq = newton_Algorithm(x0, x_ref, u_ref, max_iters=50, tol=1e-6)

def plot_results(t_ref, x_ref, u_ref, x_opt, u_opt):
    # Print results
    print("\n--- Optimal Trajectory Results ---")
    print(f"Final State (x_opt[-1]): {x_opt[-1]}")
    print(f"Final Input (u_opt[-1]): {u_opt[-1]}")
    
    # Plot 1: Reference Trajectory
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(t_ref, x_ref[:,0], label='theta1 (ref)')
    plt.plot(t_ref, x_ref[:,1], label='theta2 (ref)')
    plt.ylabel('Angles [rad]'); plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t_ref, u_ref[:,0], label='tau1 (ref)')
    plt.plot(t_ref, u_ref[:,1], label='tau2 (ref)')
    plt.xlabel('Time [s]'); plt.ylabel('Torque'); plt.legend()
    plt.tight_layout()

    # Plot 2: Optimal vs. Reference Trajectory
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t_ref, x_opt[:,0], label='theta1 (opt)')
    plt.plot(t_ref, x_opt[:,1], label='theta2 (opt)')
    plt.plot(t_ref, x_ref[:,0], '--', label='theta1 (ref)')
    plt.plot(t_ref, x_ref[:,1], '--', label='theta2 (ref)')
    plt.legend(); plt.ylabel('Angles [rad]')

    plt.subplot(2,1,2)
    plt.plot(t_ref[:-1], u_opt[:-1,0], label='tau1 (opt)')
    plt.plot(t_ref[:-1], u_opt[:-1,1], label='tau2 (opt)')
    plt.plot(t_ref[:-1], u_ref[:-1,0], '--', label='tau1 (ref)')
    plt.plot(t_ref[:-1], u_ref[:-1,1], '--', label='tau2 (ref)')
    plt.legend(); plt.xlabel('Time [s]'); plt.ylabel('Torque')
    plt.tight_layout()