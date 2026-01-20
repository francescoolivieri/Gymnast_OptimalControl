
# Libraries imported from prof. examples:
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns  #--> for better visuals
# sns.set(style='whitegrid')

from matplotlib.animation import FuncAnimation

# import control as ctrl  #control package python

# Import gymnast dynamics
import dynamics as dyn

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Import functions and global constants from trajectory_generation
import trajectory_generation as tg
#from trajectory_generation import x_e1, x_e2, t_ref, x_ref, u_ref, T
import trajectory_tracking as tt

# Import the animation function
from animation import create_and_save_animation

def task_1():
    T = 10.0
    
    # Setup Equilibria and Reference
    # Using your compute_equilibrium from trajectory_generation.py
    u_target1 = np.array([0.0, 0.0])
    u_target2 = np.array([0.5, 0.5])
    
    x_e1, u_e1 = tg.compute_equilibrium(u_target1, (0.1, -0.1))
    x_e2, u_e2 = tg.compute_equilibrium(u_target2, (0.35, -0.35))
    t_ref, x_ref, u_ref = tg.define_reference_piecewise(T, x_e1, x_e2, u_e1, u_e2)

    # Run the Newton Algorithm
    x0 = x_e1.copy()
    x_opt, u_opt, K_seq, sigma_seq, history = tg.newton_Algorithm(
        x0, x_ref, u_ref, 
        max_iters=5000, 
        tol=1e-4, 
        gamma_0=0.02,
        plot_armijo_iters=5
    )
    
    tg.generate_report_graphs(t_ref, x_ref, u_ref, x_opt, u_opt, history)

def task_2():
    
    print("--- Starting optimal trajectory computation---")
    x0 = np.array([0, 0, 0, 0]) # Start at the first equilibrium point
    
    #x_ref, u_ref, t_ref = tg.get_target_trajectory(True)
    x_ref, u_ref, t_ref = tg.get_fully_actuated_ref()
    
    print(f"States: {x_ref.shape}")
    print(f"Controls: {u_ref.shape}")
    print(f"Time: {t_ref.shape}")
    print(f"Duration: {t_ref[-1]:.2f}s")
    
    x_opt, u_opt, K_seq, sigma_seq, history = tg.newton_Algorithm(
        x0, x_ref, u_ref, 
        max_iters=5000, 
        tol=1e-4, 
        gamma_0=0.02,
        plot_armijo_iters=5
    )
    
    # Save trajectory for later use
    np.savez(
        'trajectories_npz/acrobot_optimal_trajectory.npz', 
        x=x_opt, 
        u=u_opt, 
        t=t_ref,
    )
    
    print("Generating trajectory plots...")
    tg.plot_results(t_ref, x_ref, u_ref, x_opt, u_opt, history)
    
    print("Generating animation...")    
    create_and_save_animation(
        x_opt=x_opt,      
        x_ref=x_ref,       
        T=t_ref[-1],       
        x_e1=x_ref[0],     # Initial state
        x_e2=x_ref[-1],    # Final state
        filename='figures/acrobot_swingup_newton_algorithm.gif'
    )
    
    
    #Show all figures (Plots and Animation Window)
    print("\nDisplaying all results. Close the windows to finish the script.")
    plt.show()
    

def task_3():
    
    data = np.load("trajectories_npz/acrobot_optimal_trajectory.npz")
    x_ref, u_ref, t_ref = data["x"], data["u"], data["t"] 
    
    test_disturbances = [0.2, 0.3]

    # Actual initial state from the reference trajectory
    x0 = x_ref[0].copy()
        
    for disturbance in test_disturbances:
        x0_test = x0 + disturbance
        
        x_track, u_track = tt.LQR_tracking(x_ref, u_ref, t_ref, x0_perturbed=x0_test)
        
        # Print results
        print("\n--- Optimal Trajectory Results ---")
        print(f"Final State (x_opt[-1]): {x_track[-1]}")
        print(f"Final Input (u_opt[-1]): {u_track[-1]}")

        plot_tracking(x_ref, u_ref, x_track, u_track, t_ref, f"figures/LQR/tracking_dx_{disturbance}")  


def task_4():
    
    data = np.load("trajectories_npz/acrobot_optimal_trajectory.npz")
    x_ref, u_ref, t_ref = data["x"], data["u"], data["t"]
    
    test_disturbances = [0.1, 0.3]
    
    # Actual initial state from the reference trajectory
    x0 = x_ref[0].copy()
    
    for disturbance in test_disturbances:
        x0_test = x0 + disturbance
        
        # MPC tracking
        x_opt, u_opt = tt.solve_mpc_tracking(x0_test, x_ref, u_ref, len(t_ref))
        
        # Print results
        print("\n Optimal Trajectory Results: ")
        print(f"Final State (x_opt[-1]): {x_opt[-1]}")
        print(f"Final Input (u_opt[-1]): {u_opt[-1]}")
        
        plot_tracking(x_ref, u_ref, x_opt, u_opt, t_ref, f"figures/mpc/tracking_dx_{disturbance}")    
        
    return

def plot_tracking(x_ref, u_ref, x_opt, u_opt, t_ref, name="figures/unnamed"):
    # For plotting controls, use t_ref[:-1] since controls have N-1 elements
    t_control = t_ref[:-1]
    
    # Trajectory tracking
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    # Plot velocities 
    plt.plot(t_ref, x_opt[:,2], label='theta1 vel (opt)', linewidth=1.25, alpha=0.4, zorder=1)
    plt.plot(t_ref, x_opt[:,3], label='theta2 vel (opt)', linewidth=1.25, alpha=0.4,  zorder=1)
    plt.plot(t_ref, x_ref[:,2], '--', label='theta1 vel (ref)', linewidth=1.25, alpha=0.4, zorder=1)
    plt.plot(t_ref, x_ref[:,3], '--', label='theta2 vel (ref)', linewidth=1.25, alpha=0.4,  zorder=1)
    
    # Plot theta1 and theta2 with prominent styling (higher z-order, brighter colors, thicker lines)
    plt.plot(t_ref, x_opt[:,0], label='theta1 (opt)', linewidth=2., alpha=0.9, zorder=3)
    plt.plot(t_ref, x_opt[:,1], label='theta2 (opt)', linewidth=2.,  alpha=0.9, zorder=3)
    plt.plot(t_ref, x_ref[:,0], '--', label='theta1 (ref)', linewidth=2.,  alpha=0.7, zorder=2)
    plt.plot(t_ref, x_ref[:,1], '--', label='theta2 (ref)', linewidth=2.,  alpha=0.7, zorder=2)
    plt.legend(); plt.grid(True, alpha=0.3); plt.ylabel('Angles [rad]')

    plt.subplot(2,1,2)
    plt.plot(t_control, u_opt[:,0], label='tau1 (opt)', linewidth=2.)
    plt.plot(t_control, u_opt[:,1], label='tau2 (opt)', linewidth=2.)
    plt.plot(t_control, u_ref[:,0], '--', label='tau1 (ref)', linewidth=2.)
    plt.plot(t_control, u_ref[:,1], '--', label='tau2 (ref)', linewidth=2.)
    plt.legend(); plt.grid(True, alpha=0.3); plt.xlabel('Time [s]'); plt.ylabel('Torque')
  
    plt.savefig(name + "_ref.png")
    
    # Norm error
    plt.figure(figsize=(10,5))
    plt.plot(t_ref, np.linalg.norm(x_opt - x_ref, axis=1), label='State error (norm)')
    control_error_padded = np.append(np.abs(u_opt[:, 1] - u_ref[:,1]), np.nan)
    plt.plot(t_ref, control_error_padded, label='Control error (norm)')
    plt.xlabel('Time [s]'); plt.grid(True, alpha=0.3); plt.ylabel('Error Magnitude'); plt.legend()
    
    plt.savefig(name + "_err.png")
    plt.show()


if __name__ == '__main__':
    
    task_4()
    task_3()
    
   