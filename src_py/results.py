import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src_py.main import create_cube_mesh, solve_fem

def plot_deformed_mesh(ax, mesh, initial_positions, final_positions, color, alpha, label):
    """Plot the deformed mesh showing only outer surfaces"""
    # Define outer surface node indices
    outer_nodes = {
        'front': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'back': [18, 19, 20, 21, 22, 23, 24, 25, 26],
        'left': [0, 3, 6, 9, 12, 15, 18, 21, 24],
        'right': [2, 5, 8, 11, 14, 17, 20, 23, 26],
        'top': [18, 19, 20, 21, 22, 23, 24, 25, 26],
        'bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8]
    }
    
    # Define the faces as quadrilaterals (4 corners for each face)
    face_quads = {
        'front': [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]],
        'back': [[18, 19, 22, 21], [19, 20, 23, 22], [21, 22, 25, 24], [22, 23, 26, 25]],
        'left': [[0, 3, 12, 9], [3, 6, 15, 12], [9, 12, 21, 18], [12, 15, 24, 21]],
        'right': [[2, 5, 14, 11], [5, 8, 17, 14], [11, 14, 23, 20], [14, 17, 26, 23]],
        'top': [[18, 19, 22, 21], [19, 20, 23, 22], [21, 22, 25, 24], [22, 23, 26, 25]],
        'bottom': [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    }

    # Plot original mesh in light gray
    for face_name, quads in face_quads.items():
        for quad in quads:
            vertices = [initial_positions[idx] for idx in quad]
            poly = Poly3DCollection([vertices], alpha=0.1, color='lightgray')
            ax.add_collection3d(poly)
    
    # Plot deformed mesh with filled surfaces
    for face_name, quads in face_quads.items():
        for quad in quads:
            vertices = [final_positions[idx] for idx in quad]
            poly = Poly3DCollection([vertices], alpha=alpha/2, color=color)
            ax.add_collection3d(poly)
    
    # Plot edges of the deformed mesh
    for face_nodes in outer_nodes.values():
        for i in range(len(face_nodes)-1):
            for j in range(i+1, len(face_nodes)):
                if abs(face_nodes[i] - face_nodes[j]) in [1, 3, 9]:  # Only connect adjacent nodes
                    x = [final_positions[face_nodes[i]][0], final_positions[face_nodes[j]][0]]
                    y = [final_positions[face_nodes[i]][1], final_positions[face_nodes[j]][1]]
                    z = [final_positions[face_nodes[i]][2], final_positions[face_nodes[j]][2]]
                    ax.plot(x, y, z, color=color, alpha=alpha, linewidth=1)
    
    # Plot outer nodes of the deformed mesh
    all_outer_nodes = list(set([node for face in outer_nodes.values() for node in face]))
    ax.scatter(final_positions[all_outer_nodes,0], 
              final_positions[all_outer_nodes,1], 
              final_positions[all_outer_nodes,2], 
              color=color, alpha=alpha, label=label, s=50)

def animate_solutions():
    # Create mesh and setup problem
    mesh = create_cube_mesh()
    materials = [(0, 200e9, 0.3)]
    
    # Define constraints and prescribed displacements
    constraints = []
    prescribed = []
    
    # Fix bottom nodes
    for i in range(9):
        constraints.extend([(i, 0), (i, 1), (i, 2)])
    
    # Apply displacement to top nodes with small off-axis components
    displacement = 0.1  # primary strain
    off_axis = 0.1   # small off-axis component
    for i in range(18, 27):
        prescribed.extend([
            (i, 0, off_axis),        # Small x displacement
            (i, 1, off_axis),        # Small y displacement
            (i, 2, displacement)      # Main z displacement
        ])
    
    # Solve using both methods with steps
    steps_direct = solve_fem(mesh, materials, constraints, prescribed, 'direct', num_steps=20)
    steps_cg = solve_fem(mesh, materials, constraints, prescribed, 'cg', num_steps=20)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(121, projection='3d')  # XZ view
    ax2 = fig.add_subplot(122, projection='3d')  # YZ view
    
    # Calculate axis limits from all positions
    all_positions = []
    for step in steps_direct + steps_cg:
        all_positions.extend([step[0], step[1]])
    all_positions = np.array(all_positions)
    
    x_min, x_max = all_positions[:,:,0].min(), all_positions[:,:,0].max()
    y_min, y_max = all_positions[:,:,1].min(), all_positions[:,:,1].max()
    z_min, z_max = all_positions[:,:,2].min(), all_positions[:,:,2].max()
    
    # Add small padding to the limits
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range
    
    # Animation pausing
    anim_running = True
    
    def onClick(event):
        nonlocal anim_running
        if event.button == 1:  # Left click
            if anim_running:
                anim.pause()
            else:
                anim.resume()
            anim_running = not anim_running
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        initial_pos, final_pos_direct = steps_direct[frame]
        _, final_pos_cg = steps_cg[frame]
        
        # Plot in both views
        plot_deformed_mesh(ax1, mesh, initial_pos, final_pos_direct, 'blue', 0.8, 'Direct Method')
        plot_deformed_mesh(ax1, mesh, initial_pos, final_pos_cg, 'red', 0.8, 'CG Method')
        plot_deformed_mesh(ax2, mesh, initial_pos, final_pos_direct, 'blue', 0.8, 'Direct Method')
        plot_deformed_mesh(ax2, mesh, initial_pos, final_pos_cg, 'red', 0.8, 'CG Method')
        
        # Set consistent axis limits for both views
        for ax in [ax1, ax2]:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1,1,1])
        
        # Set different viewing angles for each subplot
        ax1.view_init(elev=0, azim=0)    # XZ view
        ax2.view_init(elev=0, azim=90)   # YZ view
        
        # Titles
        ax1.set_title('XZ View')
        ax2.set_title('YZ View')
        fig.suptitle(f'Comparison of Node Positions - Step {frame}\n(Blue: Direct, Red: CG)\nClick to pause/resume')
        
        # Add legend to only one subplot
        ax1.legend()
    
    fig.canvas.mpl_connect('button_press_event', onClick)
    anim = FuncAnimation(fig, update, frames=len(steps_direct), 
                        interval=100, repeat=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_solutions() 