
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
        x_opt=x_opt,      # Your optimal trajectory
        x_ref=x_ref,       # Your reference trajectory (the ghost)
        T=t_ref[-1],       # The total time (scalar: last time value)
        x_e1=x_ref[0],     # Initial state
        x_e2=x_ref[-1],    # Final state
        filename='figures/acrobot_swingup_newton_algorithm.gif'
    )
    
    
    #Show all figures (Plots and Animation Window)
    print("\nDisplaying all results. Close the windows to finish the script.")
    plt.show()
    

def task_3():
    
    data = np.load("trajectories_npz/acrobot_optimal_trajectory.npz")
    x_opt, u_opt, t_ref = data["x"], data["u"], data["t"]
    
    
    x_ref, u_ref = tt.LQR_tracking(x_opt, u_opt)
    
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
    
    return



def task_4():
    
    data = np.load("trajectories_npz/acrobot_optimal_trajectory.npz")
    x_ref, u_ref, t_ref = data["x"], data["u"], data["t"]
    
    print(f"Last u: {u_ref[-1,:]}")
    
    
    # Use the actual initial state from the reference trajectory
    x0 = x_ref[0].copy() + 0.2
    
    x_opt, u_opt = tt.solve_mpc_tracking(x0, x_ref, u_ref, len(t_ref))
    
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
    
    plt.show()
    plt.savefig
    
    return




if __name__ == '__main__':
    
    task_4()
    
   