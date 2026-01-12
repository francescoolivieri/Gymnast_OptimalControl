import numpy as np
import matplotlib.pyplot as plt
from dynamics import *
import sympy as sp
from scipy.optimize import root
import copy

T = 10.0 #Total time in secods
N = int(T / dt) + 1 # Number of steps (including the initial state)
nu = 2 #Dimension of control vector
nx = 4 #Dimension of state vector
#N_opt = N * nx + (N-1) * nu #Total number of optimization values

#   Cost Weights
# Relaxed for tracking fully actuated reference with underactuated acrobot
Q = np.diag([130.0, 30.0, 0.0001 , 0.0001])  
R = np.diag([1e-6, 1.5])                  
Q_T = np.diag([130, 130., 1., 1.0])  

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

# #Targeted Angle: approx 20 degrees ~ 0.35 radians
# u_target1 = np.array([0.5, 0.5])
# u_target2 = np.array([-0.5, -0.5])
# theta_guess1 = (0.35, -0.35)
# theta_guess2 = (-0.35, 0.35)

# x_e1, u_e1 = compute_equilibrium(u_target1, theta_guess1)
# x_e2, u_e2 = compute_equilibrium(u_target2, theta_guess2)

# print("First equilibrium x_e1 and u_e1 = ", x_e1, u_e1)
# print("Second equilibrium x_e2 and u_e2 = ", x_e2, u_e2)

# t_ref, x_ref, u_ref = define_reference_piecewise(T, x_e1, x_e2, u_e1, u_e2)

def simulate_open_loop(x0, u_traj):
    """
    Simulate system forward from x0 using control sequence u_traj.
    """
    N_controls = u_traj.shape[0]
    N_states = N_controls + 1  # We get one more state than controls
    
    x_traj = np.zeros((N_states, nx))
    x_traj[0] = x0
    
    for t in range(N_controls):
        x_traj[t+1] = dynamics(x_traj[t], u_traj[t])
    
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
        
        # 2. Discretize A
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

def build_stage_lists(x_traj, u_traj, x_ref, u_ref, lambda_seq): #
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
    expected_reduction = 0.0
    
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

        # Update expected reduction: sum of g^T * sigma
        expected_reduction += g.T @ sigma_t
        
        # Compact Riccati updates
        P = Q + A.T @ P @ A - K_t.T @ G @ K_t
        p = q + A.T @ p - K_t.T @ G @ sigma_t

        K[t] = K_t
        sigma[t] = sigma_t

    return K, sigma, expected_reduction

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

def plot_armijo_line_search(iteration, x_traj, u_traj, K, sigma, cost_current, x_ref, u_ref, delta_J, gamma_accepted, stepsizes_tested, costs_tested, c=0.5, beta=0.7):
    
    # Generate a range of step sizes 
    max_step = max(1.25, max(stepsizes_tested) * 1.3 if stepsizes_tested else 1.25)
    steps = np.linspace(0, max_step, 200)
    costs = np.zeros(len(steps))
    
    # Compute cost along the descent direction
    for ii, step in enumerate(steps):
        x_new, u_new = forward_closed_loop_update(x_traj, u_traj, K, sigma, gamma=step)
        costs[ii] = total_cost(x_new, u_new, x_ref, u_ref, Q, R, Q_T)
    
    plt.figure(f'Armijo Line Search - Iteration {iteration}', figsize=(12, 7))
    plt.clf()
    
    # Plot actual cost along search direction (blue line)
    plt.plot(steps, costs, color='blue', linewidth=2.5, label=r'$J(x_k + \gamma \cdot \delta x, u_k + \gamma \cdot \delta u)$', alpha=0.8)
    
    # Plot linear approximation (first-order Taylor expansion) (red line)
    # J(new) ≈ J(current) + gamma * directional_derivative
    linear_approx = cost_current + delta_J * steps
    plt.plot(steps, linear_approx, color='red', linewidth=2.5, linestyle='-', label=r'$J_k + \gamma \cdot \nabla J^T \delta$', alpha=0.8)
    
    # Plot Armijo condition line (green dashed)
    # Armijo accepts if: J(new) < J(current) + c * gamma * directional_derivative
    armijo_line = cost_current + c * delta_J * steps
    plt.plot(steps, armijo_line, color='green', linestyle='--', linewidth=2.5, label=rf'$J_k + c \cdot \gamma \cdot \nabla J^T \delta$ (c={c})', alpha=0.8)
    
    # Mark tested stepsizes from Armijo procedure (orange stars)
    if stepsizes_tested and costs_tested:
        plt.scatter(stepsizes_tested, costs_tested, marker='*', s=150, color='orange', edgecolor='black', linewidth=1.5, zorder=5, label=rf'Tested stepsizes ($\beta$={beta})')
        
        # Mark the accepted stepsize (large red circle)
        plt.scatter(gamma_accepted, costs_tested[-1], marker='o', s=200, color='red', edgecolor='black', linewidth=2.5, zorder=6, label=rf'Accepted: $\gamma$={gamma_accepted:.4f}')
    
    plt.xlabel(r'Step Size $\gamma$', fontsize=14)
    plt.ylabel('Cost Value J', fontsize=14)
    plt.title(f'Armijo Line Search - Iteration {iteration}\n' + f'Current Cost = {cost_current:.4f}, Expected Reduction = {delta_J:.2e}', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    plt.xlim(0, max_step)
    plt.tight_layout()
    plt.show()

def newton_Algorithm(x0, x_ref, u_ref, max_iters, tol=1e-6, beta=0.7, c=0.5, gamma_0=1, plot_armijo_iters=10):
    
    # Ensure u_ref has correct dimension
    if u_ref.shape[0] == x_ref.shape[0]:
        print(f" u_ref has same length as x_ref ({u_ref.shape[0]}). Using first N-1 controls.")
        u_ref = u_ref[:-1]
    
    if u_ref.shape[0] != x_ref.shape[0] - 1:
        raise ValueError(f"Incompatible dimensions: x_ref has {x_ref.shape[0]} states but u_ref has {u_ref.shape[0]} controls (expected {x_ref.shape[0]-1})")
    
 
    #Initialize a feasible trajectory given the input
    #u_traj = u_ref.copy()
    u_traj = np.zeros_like(u_ref)
    x_traj = simulate_open_loop(x0, u_traj)
    
    # Check dimensions match
    assert x_traj.shape[0] == x_ref.shape[0], \
        f"Simulated trajectory length mismatch: {x_traj.shape[0]} vs {x_ref.shape[0]}"

    # Initial cost calculation
    cost_k = total_cost(x_traj, u_traj, x_ref, u_ref, Q, R, Q_T)

    # Logging initialization for plotting for the report
    history = {
        'cost': [cost_k],
        'sigma_norm': [],
        'x_trajs': [x_traj.copy()],
        'sigmas': []
    }
          
    for k in range(max_iters):
        
        #Step 1
        lambda_seq = compute_costate_trajectory(x_traj, u_traj, x_ref, u_ref)
    
        # Stage lists
        A_list, B_list, Q_list, R_list, S_list, q_list, r_list, Q_T_block, q_T = build_stage_lists(x_traj, u_traj, x_ref, u_ref, lambda_seq)
        
        #Riccati -> The second backward pass is within the function
        K, sigma, delta_J = calculate_K_and_sigma(A_list, B_list, Q_list, R_list, S_list, q_list, r_list, Q_T_block, q_T)

        # Log sigma metrics
        history['sigmas'].append(copy.deepcopy(sigma))
        history['sigma_norm'].append(np.max(np.abs(sigma)))
        
        gamma_i = gamma_0
        max_line_search_iters = 20
        success = False
        
        # Track stepsizes and costs for Armijo plot
        stepsizes_tested = []
        costs_tested = []
        
        for i in range(max_line_search_iters):
            x_new, u_new = forward_closed_loop_update(x_traj, u_traj, K, sigma, gamma=gamma_i)
            cost_new = total_cost(x_new, u_new, x_ref, u_ref, Q, R, Q_T)
            
            # Store for plotting
            stepsizes_tested.append(gamma_i)
            costs_tested.append(cost_new)
            
            # cost_new < current_cost + alpha * step_size * directional_derivative
            if cost_new < cost_k + c * gamma_i * delta_J:
                success = True
                break
            
            gamma_i *= beta
        
        if not success:
            print(f"Iteration {k}: Line search failed to find sufficient decrease.")
            break
        
        # Plot Armijo line search for first iterations or specific iterations
        if k < plot_armijo_iters and (k % 2 == 0 or k < 3):
            plot_armijo_line_search(k, x_traj, u_traj, K, sigma, cost_k, 
                                   x_ref, u_ref, delta_J, gamma_i, 
                                   stepsizes_tested, costs_tested, c, beta)
            
        # Convergence check
        cost_reduction = cost_k - cost_new
        x_traj, u_traj, cost_k = x_new, u_new, cost_new

        # Log iteration results
        history['cost'].append(cost_k)
        history['x_trajs'].append(x_traj.copy())
        
        # Print progress
        if k % 10 == 0:
            print(f"Iter {k}: Cost={cost_k:.2f}, diff_cost={cost_reduction:.2e}, ")
        
        if np.max(np.abs(sigma)) < tol:
            print(f"Converged at iteration {k}!")
            break
        
    return x_traj, u_traj, K, sigma, history

# Choose initial state (e.g., start at first equilibrium)
# x0 = x_e1.copy()

# Run Newton newton_Algorithm(x0, x_ref, u_ref, max_iters, tol=1e-6, beta=0.7, c=0.5, gamma_0=1.0)
# x_opt, u_opt, K_seq, sigma_seq, history = newton_Algorithm(x0, x_ref, u_ref, max_iters=50, tol=1e-6,  beta=0.7, c=0.5, gamma_0=1)

def plot_results(t_ref, x_ref, u_ref, x_opt, u_opt, history):
    # Print results
    print("\n--- Optimal Trajectory Results ---")
    print(f"Final State (x_opt[-1]): {x_opt[-1]}")
    print(f"Final Input (u_opt[-1]): {u_opt[-1]}")
    
    # For plotting controls, use t_ref[:-1] since controls have N-1 elements
    t_control = t_ref[:-1]
    
    # Plot 1: Reference Trajectory
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(t_ref, x_ref[:,0], label='theta1 (ref)')
    plt.plot(t_ref, x_ref[:,1], label='theta2 (ref)')
    plt.ylabel('Angles [rad]'); plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t_control, u_ref[:,0], label='tau1 (ref)')
    plt.plot(t_control, u_ref[:,1], label='tau2 (ref)')
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
    plt.plot(t_control, u_opt[:,0], label='tau1 (opt)')
    plt.plot(t_control, u_opt[:,1], label='tau2 (opt)')
    plt.plot(t_control, u_ref[:,0], '--', label='tau1 (ref)')
    plt.plot(t_control, u_ref[:,1], '--', label='tau2 (ref)')
    plt.legend(); plt.xlabel('Time [s]'); plt.ylabel('Torque')
    plt.tight_layout()
    
    plt.savefig("figures/acrobot_traj.png") 
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(history['cost'], 'b-o')
    plt.title('Cost along iterations')
    plt.xlabel('Iteration'); plt.grid(True, which="both")

    plt.subplot(1, 2, 2)
    plt.semilogy(history['sigma_norm'], 'r-o')
    plt.title('Norm of descent direction ($||\sigma||_\infty$)')
    plt.xlabel('Iteration'); plt.grid(True, which="both")
    
    plt.figure(figsize=(10, 4))
    for i, x_h in enumerate(history['x_trajs']):
        if i % 5 == 0 or i in [1, 2] or i == len(history['x_trajs'])-1:
            plt.plot(t_ref, x_h[:, 0], alpha=0.3, label=f'Iter {i}')
    plt.plot(t_ref, x_ref[:, 0], 'k--', linewidth=2, label='Desired')
    plt.title('Evolution of $\ theta_1$ Trajectory')
    plt.legend(); plt.ylabel('Angle [rad]'); plt.xlabel('Time [s]')

    plt.figure(figsize=(10, 6))
    # Plot specific iterations to see the 'shrinking' effect
    iters_to_plot = [0, 1, 2, 4] 
    colors = plt.cm.viridis(np.linspace(0, 1, len(iters_to_plot)))

    for idx, i in enumerate(iters_to_plot):
        if i < len(history['sigmas']):
            sig = np.array(history['sigmas'][i])
            plt.plot(t_ref[:-1], sig[:, 0], color=colors[idx], 
                    label=f'Iteration {i}', alpha=0.8)

    plt.title('Evolution of Armijo Descent Direction ($\sigma_t$) for $\ tau_1$')
    plt.xlabel('Time [s]')
    plt.ylabel('Correction Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if len(history['sigmas']) > 0:
        plt.figure(figsize=(10, 4))
        first_sigma = np.array(history['sigmas'][0])
        final_sigma = np.array(history['sigmas'][-1])
        
        plt.plot(t_ref[:-1], first_sigma[:, 0], label='Initial $\sigma$ (Iter 0)')
        plt.plot(t_ref[:-1], final_sigma[:, 0], label='Final $\sigma$ (Converged)')
        plt.title('Descent Direction (Armijo $\sigma$)')
        plt.legend()
        plt.ylabel('Control Correction')
        plt.xlabel('Time [s]')
    

    
def get_fully_actuated_ref():
    
    data = np.load("trajectories_npz/fully_actuated_trajectory.npz")
    u_ref = np.zeros(data["u"].shape)
    u_ref[:, 1] = data["u"][:, 1]  # Acrobot: tau1=0, tau2=elbow torque from reference
    
    # Note: multiply ref torque to be more accurate (>effort for underactuated system)  
    return data["x"], np.multiply(u_ref, 2), data["time"] #6.5


