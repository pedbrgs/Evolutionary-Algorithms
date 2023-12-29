import logging
import numpy as np
from tqdm import tqdm


class AngleModulatedDifferentialEvolution():
    """Angle Modulated Differential Evolution (AMDE) algorithm (rand/1/exp).

    Perform a homomorphous mapping to abstract a problem (defined in binary-valued space) into a
    simpler problem (defined in continuos-valued space).

    Engelbrecht, Andries P., and Gary Pampara. "Binary differential evolution strategies."
    2007 IEEE congress on evolutionary computation. IEEE, (2007).

    Attributes
    ----------
    best_solution: np.ndarray
        Best solution found during the optimization process.
    best_fitness: float
        Fitness value of the best solution found during the optimization process.
    bounds: tuple[float, float], default (-1, 1)
        Bounds for variables (min, max).
    n_coeffs: int, default 4
        Number of coefficients in the angle modulation function. The AMDE evolves values for
        the four coefficients, a, b, c, and d. The first coefficient represents the horizontal
        shift of the function, the second coefficient represents the maximum frequency of the
        sine function, the third coefficient represents the frequency of the cosine function,
        and the fourth coefficient represents the vertical shift of the function.
    """

    def __init__(self,
                 pop_size: int,
                 n_dim: int,
                 scaling_factor: float,
                 crossover_probability: float,
                 max_generations: int,
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
        self.bounds = (-1, 1)
        self.n_coeffs = 4
        self.maximization = maximization
        self.fitness_function = fitness_function

    def _init_population(self) -> np.ndarray:
        """Randomly initializes individuals from the population.

        The task of solving a binary-valued problem is reduced to a 4-dimensional problem,
        where 4 ﬂoating-point parameters need to be optimized.
        """
        pop = np.random.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.pop_size, self.n_coeffs)
        )
        return pop

    def _eval_population(self, pop: np.ndarray) -> np.ndarray:
        """Evaluate all individuals of the population."""
        fitness = np.zeros(self.pop_size)
        for i, indiv in enumerate(pop):
            bit_vector = self._angle_modulation_function(indiv)
            fitness[i] = self.fitness_function(bit_vector)
        return fitness

    def _decode_population(self, pop: np.ndarray) -> np.ndarray:
        """Decode all individuals of the population to the original binary-valued space."""
        decoded_pop = list()
        for indiv in pop:
            decoded_pop.append(self._angle_modulation_function(indiv))
        decoded_pop = np.array(decoded_pop)
        return decoded_pop

    def _select_solutions(self, pop: np.ndarray, target_vector_idx: float) -> np.ndarray:
        """Select 3 solutions randomly from the population.

        These solutions must be different from each other and different from the target vector.
        """
        possible_solution_idxs = [idx for idx in range(self.pop_size) if idx != target_vector_idx]
        selected_solution_idxs = np.random.choice(possible_solution_idxs, size=3, replace=False)
        selected_solutions = pop[selected_solution_idxs].copy()
        return selected_solutions

    def _mutation(self, pop: np.ndarray, target_vector_idx: float) -> np.ndarray:
        """Perform difference-vector based mutation."""
        indiv_1, indiv_2, indiv_3 = self._select_solutions(pop, target_vector_idx)
        donor_vector = indiv_1 + self.scaling_factor*(indiv_2 - indiv_3)
        return donor_vector

    def _exponential_crossover(self, target_vector: np.ndarray, donor_vector: np.ndarray) -> np.ndarray:
        """Perform exponential crossover."""
        n = np.random.choice(range(self.n_coeffs))
        trial_vector = np.zeros(self.n_coeffs)
        trial_vector[n] = donor_vector[n].copy()
        indices = [i if i < self.n_coeffs else i-self.n_coeffs for i in range(n+1, n+self.n_coeffs)]
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

    def _apply_boundary_constraints(self, trial_vectors: np.ndarray) -> np.ndarray:
        """Apply boundary constraints on trial vectors."""
        trial_vectors = np.clip(trial_vectors, a_min=self.bounds[0], a_max=self.bounds[1])
        return trial_vectors

    def _greedy_selection(self, pops: list, fevals: list) -> tuple[np.ndarray, np.ndarray]:
        """Perform greedy selection."""
        if self.maximization:
            pop_idxs = np.argmax(fevals, axis=0)
        else:
            pop_idxs = np.argmin(fevals, axis=0)
        pop = np.array([pops[pop_idx][indiv_idx] for indiv_idx, pop_idx in enumerate(pop_idxs)])
        fitness = np.array([fevals[pop_idx][indiv_idx] for indiv_idx, pop_idx in enumerate(pop_idxs)])
        return pop, fitness

    def _get_best_solution(self, pop: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, float]:
        """Get the current best solution ans its evaluation."""
        if self.maximization:
            best_idx = np.argmax(fitness)
        else:
            best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx].copy()
        return best_solution, best_fitness

    def _angle_modulation_function(self, coeffs: np.ndarray) -> np.ndarray:
        """Homomorphous mapping between binary-valued and continuous-valued space."""
        a, b, c, d = coeffs
        x = np.linspace(0, 1, self.n_dim)
        trig_function = np.sin(2 * np.pi * (x - a) * b * np.cos(2 * np.pi * (x - a) * c)) + d
        bit_vector = (trig_function > 0).astype(int)
        return bit_vector

    def evolve(self) -> tuple[np.ndarray, np.ndarray]:
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
        pop = self._decode_population(pop)

        return pop, fitness


def subset_sum_function(bit_vector: np.ndarray) -> float:
    """The Subset Sum Problem (SSP).

    Given n positive integers w1,…,wn, find a combination amongst them such that their sum is the
    closest to, but not exceeding, a positive integer k.”
    """
    # Array of 100 random integers
    integers = np.array(
        [
            72, 40, 47, 94, 65,  3, 27,  8, 10,  3, 23, 97,  6, 96, 62, 66, 94, 30, 99, 61, 70,
            89, 31, 14,  3,  9, 21, 82, 62, 41, 83,  8, 93, 15, 44, 48, 30, 11, 98, 99, 34, 64,
            50, 26, 10, 45, 45, 70, 27, 27, 96, 15, 84,  3, 26, 89, 51, 65, 29, 11, 98, 21, 28,
            20, 74, 69, 60, 76, 38, 11, 61, 81, 52, 50, 92, 28, 89, 88, 43, 22, 99, 67, 66,  2,
            24, 29, 41, 90, 94,  4, 62, 13, 91, 96, 96, 35, 68, 16, 31, 93
        ]
    )
    target_sum = 17
    selected_subset = integers[bit_vector == 1]
    subset_sum = np.sum(selected_subset)
    # Calculate the difference between the subset sum and the target sum
    diff = subset_sum - target_sum
    # Penalize cases where the subset sum exceeds the target sum
    penalty = max(0, diff)
    # Minimize the difference and penalize if the sum exceeds the target
    return abs(diff) + penalty


if __name__ == '__main__':

    # Initialize logger with info level
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    # Reset handlers
    logging.getLogger().handlers = []
    # Add a custom handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(handler)

    optimizer = AngleModulatedDifferentialEvolution(
        pop_size = 50,
        n_dim = 100,
        scaling_factor = 1.0,
        crossover_probability = 0.25,
        max_generations = 1000,
        maximization = False,
        fitness_function = subset_sum_function,
    )

    pop, fitness = optimizer.evolve()

    logging.info(f"Best solution: {optimizer._angle_modulation_function(optimizer.best_solution)}")
    logging.info(f"Best fitness: {optimizer.best_fitness}")
