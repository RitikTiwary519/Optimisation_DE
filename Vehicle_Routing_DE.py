import numpy as np

# Problem-specific parameters
num_vehicles = 3
vehicle_capacity = 10
num_customers = 15
depot_location = np.array([0, 0])
customer_locations = np.random.rand(num_customers, 2) * 100  # Random locations within 100x100 grid
customer_demands = np.random.randint(1, 5, num_customers)

# Differential Evolution parameters
population_size = 50
num_generations = 100
mutation_factor = 0.8
crossover_probability = 0.9

# Helper function to calculate Euclidean distance
def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Objective function: total route distance
def total_distance(routes):
    dist = 0.0
    for route in routes:
        if route:
            dist += distance(depot_location, customer_locations[route[0]])  # Depot to first customer
            for i in range(len(route) - 1):
                dist += distance(customer_locations[route[i]], customer_locations[route[i + 1]])
            dist += distance(customer_locations[route[-1]], depot_location)  # Last customer to depot
    return dist

# Initialize population with random solutions
def initialize_population():
    population = []
    for _ in range(population_size):
        individual = np.random.permutation(num_customers)  # Random permutation of customers
        population.append(individual)
    return np.array(population)

# Decode a solution into routes for vehicles
def decode_solution(solution):
    routes = []
    current_route = []
    load = 0
    for customer in solution:
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

# Differential Evolution main loop
def differential_evolution():
    population = initialize_population()
    best_solution = None
    best_cost = float('inf')

    for gen in range(num_generations):
        new_population = []
        for i in range(population_size):
            # Mutation and crossover
            idxs = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.copy(a)
            for j in range(num_customers):
                if np.random.rand() < mutation_factor:
                    mutant[j] = c[j] if np.random.rand() < crossover_probability else b[j]
            
            # Ensure the mutant is a valid permutation
            mutant = np.unique(mutant, return_index=True)[1]  # Remove duplicates
            missing_customers = np.setdiff1d(np.arange(num_customers), mutant)  # Find missing customers
            mutant = np.concatenate((mutant, missing_customers))  # Ensure all customers are present
            mutant = mutant[:num_customers]  # Trim to the correct length

            # Select the fitter individual
            candidate_routes = decode_solution(mutant)
            candidate_cost = total_distance(candidate_routes)
            current_routes = decode_solution(population[i])
            current_cost = total_distance(current_routes)
            
            if candidate_cost < current_cost:
                new_population.append(mutant)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_solution = candidate_routes
            else:
                new_population.append(population[i])
                
        population = np.array(new_population)
        print(f"Generation {gen + 1} | Best Cost: {best_cost}")

    return best_solution, best_cost

# Run the Differential Evolution algorithm
best_routes, best_cost = differential_evolution()
print("Best Routes:", best_routes)
print("Best Cost:", best_cost)
