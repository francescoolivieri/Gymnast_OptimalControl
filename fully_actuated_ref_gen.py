import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from animation import create_and_save_animation

# 1 set of parameters !
m1 = 1.0      
m2 = 1.0      
l1 = 1.0      
lc1 = 0.5     
l2 = 1.0      
lc2 = 0.5     
I1 = 0.33     
I2 = 0.33    
g = 9.81     
f1 = 1.0      
f2 = 1.0      


def acrobot_dynamics_casadi(x, u):
    """
    Continuous dynamics for FULLY actuated acrobot (both joints controlled).
    """
    # states
    theta1 = x[0]
    theta2 = x[1]
    theta1_dot = x[2]
    theta2_dot = x[3]
    
    # controls
    tau1 = u[0]
    tau2 = u[1]
    
    M11 = I1 + I2 + lc1**2 * m1 + m2*(l1**2 + 2*l1*lc2*ca.cos(theta2) + lc2**2)
    M12 = I2 + lc2*m2*(l1*ca.cos(theta2) + lc2)
    M21 = M12  # Symmetric
    M22 = I2 + lc2**2 * m2
    
    M = ca.vertcat(
        ca.horzcat(M11, M12),
        ca.horzcat(M21, M22)
    )
    
    C11 = -l1*lc2*m2*theta2_dot*ca.sin(theta2)
    C12 = -l1*lc2*m2*(theta1_dot + theta2_dot)*ca.sin(theta2)
    C21 = l1*lc2*m2*theta1_dot*ca.sin(theta2)
    C22 = 0
    
    C = ca.vertcat(
        ca.horzcat(C11, C12),
        ca.horzcat(C21, C22)
    )
    
    G1 = g*lc1*m1*ca.sin(theta1) + g*m2*(l1*ca.sin(theta1) + lc2*ca.sin(theta1 + theta2))
    G2 = g*m2*lc2*ca.sin(theta1 + theta2)
    G = ca.vertcat(G1, G2)
    
    F = ca.vertcat(
        ca.horzcat(f1, 0),
        ca.horzcat(0, f2)
    )
    
    qdot = ca.vertcat(theta1_dot, theta2_dot)
    tau = ca.vertcat(tau1, tau2)  # FULLY actuated: both torques applied
    
    # Solve for accelerations: qddot = M^(-1) * RHS
    RHS = tau - ca.mtimes((C + F), qdot) - G
    qddot = ca.solve(M, RHS)
    
    # Return state derivative: [qdot; qddot]
    x_dot = ca.vertcat(qdot, qddot)
    
    return x_dot


def solve_swing_up_trajectory(T=10.0, N=500 ):
    
    dt = T / N  # Time step
    
    opti = ca.Opti()
    
    X = opti.variable(4, N+1)
    U = opti.variable(2, N)
    
    x_init = np.array([0.0, 0.0, 0.0, 0.0])     
    x_final = np.array([np.pi, 0.0, 0.0, 0.0])  
    
    opti.subject_to(X[:, 0] == x_init)  # Initial state constraint
    
    # Add pumping
    oscillation_duration = 0.45 # 45% of trajectory
    n_oscillations = 3
    
    for i in range(1, n_oscillations + 1):
        t_osc = int((oscillation_duration * i / (n_oscillations + 1)) * N) # Time index for this waypoint
        
        # Alternate swing direction
        if i % 2 == 1:
            target_theta1 = -np.pi/5 
        else:
            target_theta1 = np.pi/6  
        
        # aplly theta target for oscillation
        opti.subject_to(X[0, t_osc] == target_theta1)
        
        # Keep theta2 relatively aligned during pumping
        opti.subject_to(opti.bounded(-np.pi/10, X[1, t_osc], np.pi/10))
    

    # Add dynamics constraints (RK4)
    for k in range(N):
        
        # time k
        x_k = X[:, k]    
        u_k = U[:, k]    
        
        # RK4 integration scheme
        k1 = acrobot_dynamics_casadi(x_k, u_k)
        k2 = acrobot_dynamics_casadi(x_k + dt/2 * k1, u_k)
        k3 = acrobot_dynamics_casadi(x_k + dt/2 * k2, u_k)
        k4 = acrobot_dynamics_casadi(x_k + dt * k3, u_k)
        
        x_next = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # applying dynamics!
        opti.subject_to(X[:, k+1] == x_next)
    
    
    # Cost function
    cost = 0

    # Terminal cost
    final_error = X[:, -1] - x_final
    cost += 100.0 * ca.sumsqr(final_error)
    
    # Control cost
    for k in range(N):
        cost += 0.01 * ca.sumsqr(U[:, k])
    
    # State cost
    oscillation_end = int(0.4 * N)
    
    for k in range(N+1):
        state_error = X[:, k] - x_final
        if k < oscillation_end:
            cost += 0.01 * ca.sumsqr(state_error)  # Light during pumping
        else:
            cost += 0.5 * ca.sumsqr(state_error)  
    
    opti.minimize(cost)
    
   
    opts = {
        'ipopt.print_level': 5,        
        'ipopt.max_iter': 3000,       
        'ipopt.tol': 1e-6,             
    }
    opti.solver('ipopt', opts)
    
    
    try:
        sol = opti.solve()
        
        # Get solution
        x_traj = sol.value(X).T  # Transpose to (N+1, 4)
        u_traj = sol.value(U).T  # Transpose to (N, 2)
        time = np.linspace(0, T, N+1)
        
        print(f"Final state error:   {np.linalg.norm(x_traj[-1] - x_final):.6f}")
        
        return x_traj, u_traj, time
        
    except Exception as e:
        print(f"FAILED: {e}")
        exit()
        


def plot_trajectory(time, x_traj, u_traj, save_path='figures/fully_actuated_acrobot_trajectory.png'):
    # time[:-1] since controls have N-1 elements
    time_u = time[:-1]
    
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(time, x_traj[:,0], label='theta1')
    plt.plot(time, x_traj[:,1], label='theta2')
    plt.ylabel('Angles [rad]'); plt.legend()
    plt.subplot(2,1,2)
    plt.plot(time_u, u_traj[:,0], label='tau1')
    plt.plot(time_u, u_traj[:,1], label='tau2')
    plt.xlabel('Time [s]'); plt.ylabel('Torque'); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def main():
    """FULLY ACTUATED ACROBOT - Reference trajectory generation"""
    
    # Trajectory generation parameters
    T = 10.0            # Total time [seconds]
    dt = 2e-2
    N = int(T/dt)             # Discretization points
    
    # Solve optimal control problem
    x_traj, u_traj, time = solve_swing_up_trajectory(T, N)
    
    # Save trajectory for later use
    np.savez(
        'fully_actuated_trajectory.npz', 
        x=x_traj, 
        u=u_traj, 
        time=time,
        T=T,
        N=N
    )
    
    plot_trajectory(time, x_traj, u_traj)
    
    print("GENERATING ANIMATION...")

    
    x_e1 = np.array([0.0, 0.0, 0.0, 0.0])     # Initial equilibrium (down)
    x_e2 = np.array([np.pi, 0.0, 0.0, 0.0])   # Target equilibrium (up)
    
    create_and_save_animation(
        x_opt=x_traj,
        x_ref=x_traj,  # Use same as reference
        T=T,
        x_e1=x_e1,
        x_e2=x_e2,
        l1=l1,
        l2=l2,
        filename='figures/acrobot_fully_actuated_swingup.gif'
    )
    
    
if __name__ == "__main__":
    main()
