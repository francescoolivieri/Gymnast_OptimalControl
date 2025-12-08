import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- In Dynamics ---
l1 = 1.0  # Length of link 1
l2 = 1.0  # Length of link 2

def get_link_positions(theta1, theta2, l1, l2):
    """Calculates the (x, y) coordinates of the joint (P1) and the end-effector (P2)."""
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta1 + theta2)
    y2 = y1 - l2 * np.cos(theta1 + theta2)
    return (0, x1, x2), (0, y1, y2)

# --- Animation Function  ---
def create_and_save_animation(x_opt, T, x_e1, x_e2, l1=l1, l2=l2, filename='acrobot_optimal_swing.gif'):
    """Sets up, runs, and saves the animation, including equilibrium points."""
    
    # --- Setup Animation Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-(l1 + l2) * 1.1, (l1 + l2) * 1.1)
    ax.set_ylim(-(l1 + l2) * 1.1, (l1 + l2) * 1.1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Acrobot Optimal Trajectory (T={T}s)')
    ax.grid(True)
    
    # --- PLOT EQUILIBRIUM POINTS (STATIC) ---
    e1_x_coords, e1_y_coords = get_link_positions(x_e1[0], x_e1[1], l1, l2)
    ax.plot(e1_x_coords[-1], e1_y_coords[-1], 'x', color='red', markersize=10, 
            markeredgewidth=2, label='Final Position (e1)')
    
    e2_x_coords, e2_y_coords = get_link_positions(x_e2[0], x_e2[1], l1, l2)
    ax.plot(e2_x_coords[-1], e2_y_coords[-1], 'P', color='green', markersize=10, 
            markeredgewidth=2, label='Target Equilibrium (e2)')
    ax.legend(loc='lower right')


    # The line object representing the two links (DYNAMIC)
    line, = ax.plot([], [], 'o-', lw=3, color='blue', 
                    markeredgecolor='black', markerfacecolor='red', markersize=6)
    # The text object to display the time
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

    # --- Animation Functions (NESTED) ---
    def init_animation():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update_animation(frame):
        # The variables fig and ax MUST NOT be recreated here.
        theta1 = x_opt[frame, 0]
        theta2 = x_opt[frame, 1]
        
        x_coords, y_coords = get_link_positions(theta1, theta2, l1, l2)
        
        line.set_data(x_coords, y_coords)
        
        current_time = frame * T / (x_opt.shape[0] - 1)
        time_text.set_text(f'Time: {current_time:.2f} s')
        
        return line, time_text

    # Create and save the animation object
    ani = FuncAnimation(fig, update_animation, frames=x_opt.shape[0], 
                        init_func=init_animation, blit=False, interval=50)

    # Save the animation
    ani.save(filename, writer='pillow', fps=20) 
    print(f"Animation successfully saved as {filename}.")

    # IMPORTANT: DO NOT call plt.show() here. main.py calls it once.