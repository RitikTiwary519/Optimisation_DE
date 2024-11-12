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
            _, idx = np.unique(mutant, return_index=True)
            mutant = mutant[np.sort(idx)]

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
