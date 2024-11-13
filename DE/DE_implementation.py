import numpy as np

# Define the function to minimize (Rastrigin function here)
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Differential Evolution parameters
dim = 5               # Number of dimensions (variables)
pop_size = 20         # Population size
max_gen = 100         # Maximum number of generations
F = 0.8               # Mutation factor
CR = 0.7              # Crossover probability
bounds = (-5.12, 5.12)  # Variable bounds for each dimension

# Initialize the population randomly within the bounds
population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

# Evolution process
for gen in range(max_gen):
    for i in range(pop_size):
        # Mutation step: select 3 random individuals different from i
        idxs = [idx for idx in range(pop_size) if idx != i]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        
        # Create a mutant vector
        mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
        
        # Crossover step: combine target and mutant vector
        crossover = np.random.rand(dim) < CR
        if not np.any(crossover):  # ensure at least one dimension crosses
            crossover[np.random.randint(0, dim)] = True
        
        trial = np.where(crossover, mutant, population[i])
        
        # Selection step: choose the better between target and trial vector
        if rastrigin(trial) < rastrigin(population[i]):
            population[i] = trial
    
    # Track the best solution so far
    best_idx = np.argmin([rastrigin(ind) for ind in population])
    best_solution = population[best_idx]
    best_fitness = rastrigin(best_solution)
    
    print(f"Generation {gen}: Best fitness = {best_fitness}")

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
