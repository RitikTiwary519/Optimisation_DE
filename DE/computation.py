# import numpy as np
# from scipy.optimize import differential_evolution
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Define the ranges
# x_range = np.arange(-4, 4, 0.01)
# y_range = np.arange(-4, 4, 0.01)

# # Create meshgrid
# x, y = np.meshgrid(x_range, y_range)

# # Define the function for plotting
# z = np.sqrt(np.sqrt(x**2 + y**2))

# # Define the bounds
# bounds = [[-4, 4], [-4, 4]]

# # Define the function to minimize
# def func(p):
#     x, y = p
#     r = np.sqrt(x**2 + y**2)
#     return r

# # Set up the plot
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(x, y, z, cmap=cm.jet, linewidth=0, antialiased=False, alpha=0.6)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Initialize a scatter plot for the optimization path
# scat = ax.scatter([], [], [], color='red', marker='o', s=30, label='Path')
# fig.colorbar(ax.plot_surface(x, y, z, cmap=cm.jet, linewidth=0, antialiased=False, alpha=0.6), 
#              shrink=0.5, aspect=5)

# # Callback function to store intermediate steps
# path_data = []

# def callback(xk, convergence=None):
#     z_val = func(xk)
#     path_data.append((xk[0], xk[1], z_val))

# # Run differential evolution with callback for path tracking
# result = differential_evolution(func, bounds, callback=callback, disp=True)

# # Scatter the optimal point at the end
# optimal_point = result.x
# z_value = func(optimal_point)
# ax.scatter(optimal_point[0], optimal_point[1], z_value, color='green', marker='x', s=100, 
#            label='Optimal Point')
# ax.legend()

# # Define animation update function
# def update(frame):
#     scat._offsets3d = ([point[0] for point in path_data[:frame]], 
#                        [point[1] for point in path_data[:frame]], 
#                        [point[2] for point in path_data[:frame]])
#     return scat,

# # Create animation
# ani = FuncAnimation(fig, update, frames=len(path_data), blit=False, interval=200)

# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import differential_evolution

# Define grid dimensions
grid_size = 20
x_range = np.linspace(-4, 4, grid_size)
y_range = np.linspace(-4, 4, grid_size)

# Define the meshgrid
x, y = np.meshgrid(x_range, y_range)
z = np.sqrt(np.sqrt(x**2 + y**2))  # Function values for visualization

# Set bounds for the optimizer
bounds = [(-10, 10), (-10, 10)]

# Define the function to minimize
def func(p):
    x, y = p
    return np.sqrt(x**2 + y**2)

# Track path data and function values for convergence visualization
path_data = []
func_values = []

# Callback function to record steps of differential evolution
def callback(xk, convergence=None):
    path_data.append((xk[0], xk[1]))
    func_values.append(func(xk))

# Run the differential evolution algorithm with a callback
result = differential_evolution(func, bounds, callback=callback, disp=True)
optimal_point = result.x

# Set up the plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap and path on the first subplot
heatmap = ax1.imshow(z, extent=(-4, 4, -4, 4), origin='lower', cmap='coolwarm', alpha=0.6)
ax1.plot(optimal_point[0], optimal_point[1], 'go', markersize=10, label='Optimal Point')  # Final point
path_scat = ax1.scatter([], [], color='red', s=30, label='Path')

# Set labels for pathfinding plot
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Pathfinding Animation to Optimal Solution")
ax1.legend()

# Set up convergence plot on the second subplot
ax2.set_title("Convergence Over Iterations")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Objective Function Value")
convergence_line, = ax2.plot([], [], color='blue')

# Define update function for animation
def update(frame):
    # Update path data on the pathfinding plot
    if frame < len(path_data):
        path_scat.set_offsets(np.c_[[p[0] for p in path_data[:frame]], [p[1] for p in path_data[:frame]]])
    # Update convergence data on the second plot
    convergence_line.set_data(range(frame), func_values[:frame])
    ax2.set_xlim(0, len(func_values))
    ax2.set_ylim(min(func_values), max(func_values) * 1.1)
    return path_scat, convergence_line

# Create the animation
ani = FuncAnimation(fig, update, frames=len(path_data), blit=True, interval=200)

plt.tight_layout()
plt.show()
