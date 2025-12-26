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
def create_and_save_animation(x_opt, x_ref, T, x_e1, x_e2, l1=l1, l2=l2, filename='acrobot_optimal_swing.gif'):
    """Sets up, runs, and saves the animation, including equilibrium points and reference ghost."""
    
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

    # --- DYNAMIC OBJECTS ---
    # The "Ghost" Reference Acrobot (Green, Dashed)
    ref_line, = ax.plot([], [], 'o--', lw=2, color='green', alpha=0.5, label="Reference")
    
    # The Optimal Acrobot (Blue, Solid)
    opt_line, = ax.plot([], [], 'o-', lw=3, color='blue', 
                        markeredgecolor='black', markerfacecolor='red', markersize=6, label="Optimal")
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='lower right')

    def init_animation():
        opt_line.set_data([], [])
        ref_line.set_data([], [])
        time_text.set_text('')
        return opt_line, ref_line, time_text

    def update_animation(frame):
        # 1. Update Optimal Acrobot
        theta1_opt = x_opt[frame, 0]
        theta2_opt = x_opt[frame, 1]
        x_coords_opt, y_coords_opt = get_link_positions(theta1_opt, theta2_opt, l1, l2)
        opt_line.set_data(x_coords_opt, y_coords_opt)
        
        # 2. Update Reference Acrobot (The Ghost)
        theta1_ref = x_ref[frame, 0]
        theta2_ref = x_ref[frame, 1]
        x_coords_ref, y_coords_ref = get_link_positions(theta1_ref, theta2_ref, l1, l2)
        ref_line.set_data(x_coords_ref, y_coords_ref)
        
        current_time = frame * T / (x_opt.shape[0] - 1)
        time_text.set_text(f'Time: {current_time:.2f} s')
        
        return opt_line, ref_line, time_text

    ani = FuncAnimation(fig, update_animation, frames=x_opt.shape[0], 
                        init_func=init_animation, blit=False, interval=50)

    ani.save(filename, writer='pillow', fps=20) 
    print(f"Animation successfully saved as {filename}.")