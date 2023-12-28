import logging
import numpy as np
from tqdm import tqdm


class DifferentialEvolution():
    """Differential Evolution (DE) algorithm (rand/1/exp).

    R. Storn and K. Price, "Differential evolution - A Simple and efﬁcient adaptive scheme for
    global optimization over continuous spaces," International Computer Science Institute, Tech.
    Rep. TR-95-012, (1995).

    Attributes
    ----------
    best_solution: np.ndarray
        Best solution found during the optimization process.
    best_fitness: float
        Fitness value of the best solution found during the optimization process.
    """

    def __init__(self,
                 pop_size: int,
                 n_dim: int,
                 scaling_factor: float,
                 crossover_probability: float,
                 max_generations: int,
                 bounds: tuple[float, float],
                 maximization: bool,
                 fitness_function: callable,
                ):
        """
        Parameters
        ----------
        pop_size: int
            Number of individuals in the population.
        n_dim: int
            Number of dimensions/variables.
        scaling_factor: float
            The mutation constant. In the literature this is also known as differential weight,
            being denoted by F. It should be in the range [0, 2].
        crossover_probability: float
            The recombination constant, should be in the range [0, 1]. In the literature this is
            also known as the crossover probability. Increasing this value allows a larger number
            of mutants to progress into the next generation, but at the risk of population
            stability.
        max_generations: int
            The maximum number of generations over which the entire population is evolved.
        bounds: tuple[float, float]
            Bounds for variables (min, max).
        maximization: bool
            If true, consider maximizing the objective function. Otherwise, consider minimizing
            the objective function.
        fitness_function: callable
            The objective function to be optimized. Must be in the form f(x), where x is the
            argument in the form of a 1-D array.
        """
        if pop_size < 4:
            raise ValueError(
                "To perform the crossover, it is expected to have at least 3 solutions that are "
                "different from each other and from the target vector. However, the 'pop_size' is"
                f" set to {pop_size}."
            )
        self.pop_size = pop_size
        if scaling_factor < 0 or scaling_factor > 2:
            raise ValueError("Scaling factor should be in the range [0, 2].")
        self.scaling_factor = scaling_factor
        if crossover_probability < 0 or crossover_probability > 1:
            raise ValueError("Crossover probability should be in the range [0, 1].")
        self.crossover_probability = crossover_probability
        self.n_dim = n_dim
        self.max_generations = max_generations
        self.bounds = bounds
        self.maximization = maximization
        self.fitness_function = fitness_function

    def _init_population(self):
        """Randomly initializes individuals from the population."""
        pop = np.random.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.n_dim)
        )
        return pop

    def _eval_population(self, pop: np.ndarray):
        """Evaluate all individuals of the population."""
        fitness = np.zeros(self.pop_size)
        for i, indiv in enumerate(pop):
            fitness[i] = self.fitness_function(indiv)
        return fitness

    def _select_solutions(self, pop: np.ndarray, target_vector_idx: float):
        """Select 3 solutions randomly from the population.

        These solutions must be different from each other and different from the target vector.
        """
        possible_solution_idxs = [idx for idx in range(self.pop_size) if idx != target_vector_idx]
        selected_solution_idxs = np.random.choice(possible_solution_idxs, size=3, replace=False)
        selected_solutions = pop[selected_solution_idxs].copy()
        return selected_solutions

    def _mutation(self, pop: np.ndarray, target_vector_idx: float):
        """Perform difference-vector based mutation."""
        indiv_1, indiv_2, indiv_3 = self._select_solutions(pop, target_vector_idx)
        donor_vector = indiv_1 + self.scaling_factor*(indiv_2 - indiv_3)
        return donor_vector

    def _exponential_crossover(self, target_vector: np.ndarray, donor_vector: np.ndarray):
        """Perform exponential crossover."""
        n = np.random.choice(range(self.n_dim))
        trial_vector = np.zeros(self.n_dim)
        trial_vector[n] = donor_vector[n].copy()
        indices = [i if i < self.n_dim else i-self.n_dim for i in range(n+1, n+self.n_dim)]
        remaining_idxs = indices.copy()
        for i in indices:
            r = np.random.uniform()
            if r <= self.crossover_probability:
                trial_vector[i] = donor_vector[i].copy()
                remaining_idxs.remove(i)
            else:
                trial_vector[remaining_idxs] = target_vector[remaining_idxs].copy()
                break
        return trial_vector

    def _apply_boundary_constraints(self, trial_vectors: np.ndarray):
        """Apply boundary constraints on trial vectors."""
        trial_vectors = np.clip(trial_vectors, a_min=self.bounds[0], a_max=self.bounds[1])
        return trial_vectors

    def _greedy_selection(self, pops: np.ndarray, fevals: np.ndarray):
        """Perform greedy selection."""
        if self.maximization:
            pop_idxs = np.argmax(fevals, axis=0)
        else:
            pop_idxs = np.argmin(fevals, axis=0)
        pop = np.array([pops[pop_idx][indiv_idx] for indiv_idx, pop_idx in enumerate(pop_idxs)])
        fitness = np.array([fevals[pop_idx][indiv_idx] for indiv_idx, pop_idx in enumerate(pop_idxs)])
        return pop, fitness

    def _get_best_solution(self, pop: np.ndarray, fitness: np.ndarray):
        """Get the current best solution ans its evaluation."""
        if self.maximization:
            best_idx = np.argmax(fitness)
        else:
            best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx].copy()
        return best_solution, best_fitness

    def evolve(self):
        """Evolve the population."""

        pop = self._init_population()
        fitness = self._eval_population(pop)
        self.best_solution, self.best_fitness = self._get_best_solution(pop, fitness)

        progress_bar = tqdm(total=self.max_generations, desc="Number of generations")

        for n in range(self.max_generations):
            next_pop = list()
            for i in range(self.pop_size):
                donor_vector = self._mutation(pop=pop, target_vector_idx=i)
                trial_vector = self._exponential_crossover(
                    target_vector=pop[i].copy(),
                    donor_vector=donor_vector.copy()
                )
                next_pop.append(trial_vector)
            next_pop = self._apply_boundary_constraints(next_pop)
            next_fitness = self._eval_population(next_pop)
            next_pop, next_fitness = self._greedy_selection(
                pops=[pop, next_pop],
                fevals=[fitness, next_fitness]
            )
            best_solution, best_fitness = self._get_best_solution(pop, fitness)
            if self.maximization:
                if best_fitness > self.best_fitness:
                    self.best_solution = best_solution.copy()
                    self.best_fitness = best_fitness.copy()
            else:
                if best_fitness < self.best_fitness:
                    self.best_solution = best_solution.copy()
                    self.best_fitness = best_fitness.copy()
            progress_bar.update(1)
            pop = next_pop.copy()
            fitness = next_fitness.copy()
        progress_bar.close()

        return pop, fitness


def schaffern4(x):
    """Schaffer function N. 4."""
    tmp1 = np.power(np.cos(np.sin(np.absolute(np.power(x[0],2)-np.power(x[1],2)))),2)-0.5
    tmp2 = np.power(1+0.001*(np.power(x[0],2)+np.power(x[1],2)),2)
    return 0.5+tmp1/tmp2


if __name__ == '__main__':

    # Initialize logger with info level
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    # Reset handlers
    logging.getLogger().handlers = []
    # Add a custom handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(handler)

    optimizer = DifferentialEvolution(
        n_dim=2,
        pop_size=40,
        scaling_factor=1.0,
        crossover_probability=0.25,
        max_generations=5000,
        bounds=(-100, 100),
        maximization=False,
        fitness_function=schaffern4
    )

    pop, fitness = optimizer.evolve()

    logging.info(f"Best solution: {optimizer.best_solution}")
    logging.info(f"Best fitness: {optimizer.best_fitness}")
