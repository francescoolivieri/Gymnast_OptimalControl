
# Libraries imported from prof. examples:
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns  #--> for better visuals
# sns.set(style='whitegrid')

from matplotlib.animation import FuncAnimation

import control as ctrl  #control package python

# Import gymnast dynamics
import dynamics as dyn

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Import functions and global constants from trajectory_generation
import trajectory_generation as tg
from trajectory_generation import x_e1, x_e2, t_ref, x_ref, u_ref, T

# Import the animation function
from animation import create_and_save_animation


if __name__ == '__main__':
    #num_states = 4
    #num_inputs = 2
    
    #xx = np.zeros((num_states,))
    #uu = np.zeros((num_inputs,))
    
    #xx[:] = np.array([1, 1, 1, 1])  
    #uu[:] = np.array([0, 1])        
    
    #print(dyn.discrete_dynamics(xx, uu))
    #print(dyn.dynamics(xx, uu))
    
    print("--- Starting optimal trajectory computation---")
    x0 = x_e1.copy() # Start at the first equilibrium point
    
    x_opt, u_opt, K_seq, sigma_seq = tg.newton_Algorithm(
        x0, x_ref, u_ref, 
        max_iters=1000, 
        tol=1e-6, 
        gamma_0=0.1 #Still need to add the Newton step
    )
    
    print("Generating trajectory plots...")
    tg.plot_results(t_ref, x_ref, u_ref, x_opt, u_opt)
    
    print("Generating animation...")
    
    # Use the T and l1/l2 values from the optimization setup (l1=1.0, l2=1.0)
    '''
        create_and_save_animation(
        x_opt, T,
        x_e1=x_e1,        
        x_e2=x_e2,
        l1=1.0, 
        l2=1.0, 
        filename='acrobot_optimal.gif'
    )
    '''
    create_and_save_animation(
    x_opt=x_opt,      # Your optimal trajectory
    x_ref=x_ref,       # Your reference trajectory (the ghost)
    T=T,               # The total time (e.g., 10.0)
    x_e1=x_e1,         # Equilibrium 1
    x_e2=x_e2,         # Equilibrium 2
    filename='acrobot_swingup.gif'
    )
    
    #Show all figures (Plots and Animation Window)
    print("\nDisplaying all results. Close the windows to finish the script.")
    plt.show()