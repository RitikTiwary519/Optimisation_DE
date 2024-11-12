import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import differential_evolution

# Problem-specific parameters
num_vehicles = 3
vehicle_capacity = 10
num_customers = 15
depot_location = np.array([0, 0])
customer_locations = np.random.rand(num_customers, 2) * 100  # Customer locations randomly
customer_demands = np.random.randint(1, 5, num_customers)  # Random customer demands

# Differential Evolution parameters
population_size = 50
num_generations = 100
mutation_factor = 0.8
crossover_probability = 0.9

# Helper function to calculate Euclidean distance
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Objective function to minimize (total distance)
def total_distance(solution):
    # Decode the solution (a permutation of customer indices)
    solution = np.round(solution).astype(int)  # Round to nearest integer for valid customer indices
    routes = decode_solution(solution)
    total_cost = 0
    for route in routes:
        if route:
            total_cost += distance(depot_location, customer_locations[route[0]])  # Depot to first customer
            for i in range(len(route) - 1):
                total_cost += distance(customer_locations[route[i]], customer_locations[route[i + 1]])
            total_cost += distance(customer_locations[route[-1]], depot_location)  # Last customer to depot
    return total_cost

# Decode a solution into routes for vehicles
def decode_solution(solution):
    routes = []
    current_route = []
    load = 0
    for customer in solution:
        customer = int(round(customer))  # Ensure customer index is an integer after rounding
        demand = customer_demands[customer]
        if load + demand > vehicle_capacity:
            routes.append(current_route)
            current_route = []
            load = 0
        current_route.append(customer)
        load += demand
    if current_route:
        routes.append(current_route)
    return routes

# Function to initialize the Differential Evolution algorithm
def run_differential_evolution():
    bounds = [(0, num_customers - 1)] * num_customers  # Each customer index will be chosen
    result = differential_evolution(total_distance, bounds, maxiter=num_generations, popsize=population_size, mutation=mutation_factor, recombination=crossover_probability)
    return decode_solution(result.x), result.fun

# Plot and animate the best solution
def plot_solution(routes, depot_location, customer_locations):
    fig, ax = plt.subplots()
    ax.scatter(*depot_location, color='red', s=100, label="Depot")
    ax.scatter(customer_locations[:, 0], customer_locations[:, 1], color='blue', s=50, label="Customers")
    
    colors = ['orange', 'green', 'purple']
    
    for idx, route in enumerate(routes):
        if route:
            route_points = [depot_location] + [customer_locations[i] for i in route] + [depot_location]
            route_points = np.array(route_points)
            ax.plot(route_points[:, 0], route_points[:, 1], color=colors[idx % len(colors)], label=f"Vehicle {idx+1}")
    
    ax.legend()
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Vehicle Routing Problem Solution")
    plt.show()

def animate_solution(routes, depot_location, customer_locations):
    fig, ax = plt.subplots()
    ax.scatter(*depot_location, color='red', s=100, label="Depot")
    ax.scatter(customer_locations[:, 0], customer_locations[:, 1], color='blue', s=50, label="Customers")
    
    colors = ['orange', 'green', 'purple']
    lines = []
    for idx, route in enumerate(routes):
        line, = ax.plot([], [], color=colors[idx % len(colors)], label=f"Vehicle {idx+1}")
        lines.append(line)
    
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    
    def update(frame):
        for idx, route in enumerate(routes):
            if frame < len(route) + 1:
                route_points = [depot_location] + [customer_locations[i] for i in route[:frame]] + [depot_location]
                route_points = np.array(route_points)
                lines[idx].set_data(route_points[:, 0], route_points[:, 1])
        return lines

    ani = FuncAnimation(fig, update, frames=max(len(route) for route in routes) + 1, init_func=init, blit=True, repeat=False)
    ax.legend()
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Vehicle Routing Problem Solution Animation")
    plt.show()

# Run the Differential Evolution algorithm and plot the result
best_routes, best_cost = run_differential_evolution()
print("Best Routes:", best_routes)
print("Best Cost:", best_cost)

# Plot the solution
plot_solution(best_routes, depot_location, customer_locations)

# Animate the solution
animate_solution(best_routes, depot_location, customer_locations)
