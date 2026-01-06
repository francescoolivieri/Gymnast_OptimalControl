
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

# Import the animation function
from animation import create_and_save_animation


if __name__ == '__main__':
    
    print("--- Starting optimal trajectory computation---")
    x0 = np.array([0, 0, 0, 0]) # Start at the first equilibrium point
    
    #x_ref, u_ref, t_ref = tg.get_target_trajectory(True)
    
    x_ref, u_ref, t_ref = tg.get_fully_actuated_ref()
    
    
    
    print(f"States: {x_ref.shape}")
    print(f"Controls: {u_ref.shape}")
    print(f"Time: {t_ref.shape}")
    print(f"Duration: {t_ref[-1]:.2f}s")
    
    x_opt, u_opt, K_seq, sigma_seq = tg.newton_Algorithm(
        x0, x_ref, u_ref, 
        max_iters=5000, 
        tol=1e-4, 
        gamma_0=0.01 
    )
    
    print("Generating trajectory plots...")
    tg.plot_results(t_ref, x_ref, u_ref, x_opt, u_opt)
    
    print("Generating animation...")
    

    create_and_save_animation(
    x_opt=x_opt,      # Your optimal trajectory
    x_ref=x_ref,       # Your reference trajectory (the ghost)
    T=t_ref[-1],       # The total time (scalar: last time value)
    x_e1=x_ref[0],     # Initial state
    x_e2=x_ref[-1],    # Final state
    filename='figures/acrobot_swingup.gif'
    )
    
    #Show all figures (Plots and Animation Window)
    print("\nDisplaying all results. Close the windows to finish the script.")
    plt.show()