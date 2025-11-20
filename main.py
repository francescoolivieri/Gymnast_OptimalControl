
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



if __name__ == '__main__':
    num_states = 4
    num_inputs = 2
    
    xx = np.zeros((num_states,))
    uu = np.zeros((num_inputs,))
    
    xx[:] = np.array([1, 1, 1, 1])  
    uu[:] = np.array([0, 1])        
    
    print(dyn.discrete_dynamics(xx, uu))