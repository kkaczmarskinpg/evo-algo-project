import time
import random
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass

# Import all components
from chromosome import Individual
from config import GAConfig
from population import Population
from selection import SelectionOperators
from crossover import CrossoverOperators
from mutation import MutationOperators, InversionOperator


@dataclass
class GenerationResult:
    """Results from a single generation."""
    generation: int
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    std_fitness: float
    best_individual: Individual
    execution_time: float


class GeneticAlgorithm:
    """
    Main genetic algorithm implementation.
    
    This class orchestrates all genetic algorithm components to solve
    optimization problems using configurable parameters and operators.
    """
    
    def __init__(self, config: GAConfig):
        """
        Initialize genetic algorithm.
        
        Args:
            config: GA configuration object
        """
        self.config = config
        self.population = Population(config)
        self.results: List[GenerationResult] = []
        self.best_ever_individual: Optional[Individual] = None
        self.minimize = True  # Default to minimization
        
    def set_objective(self, minimize: bool = True) -> None:
        """
        Set optimization objective.
        
        Args:
            minimize: True for minimization, False for maximization
        """
        self.minimize = minimize
    
    def initialize_population(self, objective_function: Callable) -> None:
        """
        Initialize population with random individuals.
        
        Args:
            objective_function: Function to optimize
        """
        self.population.initialize_random(objective_function)
        self.results = []
        self.best_ever_individual = None
    
    def evolve_generation(self, objective_function: Callable) -> GenerationResult:
        """
        Evolve one generation of the population.
        
        Args:
            objective_function: Function to optimize
            
        Returns:
            Results from this generation
        """
        start_time = time.time()
        
        # Evaluate population
        self.population.evaluate_population(objective_function)
        
        # Selection
        parents = SelectionOperators.select_parents(
            self.population.individuals, 
            self.config,
            self.config.population_size,
            self.minimize
        )
        
        # Crossover
        offspring = CrossoverOperators.apply_crossover(parents, self.config)
        
        # Mutation
        offspring = MutationOperators.apply_mutation(offspring, self.config)
        
        # Inversion
        offspring = InversionOperator.apply_inversion(offspring, self.config)
        
        # Evaluate offspring
        for individual in offspring:
            if individual.fitness is None:
                individual.evaluate(objective_function)
        
        # Update population with elitism
        self.population.update_population(offspring, self.minimize)
        
        # Calculate generation statistics
        stats = self.population.get_statistics()
        best_individual = self.population.get_best_individual(self.minimize)
        
        # Update best ever individual
        if (self.best_ever_individual is None or
            (self.minimize and best_individual.fitness < self.best_ever_individual.fitness) or
            (not self.minimize and best_individual.fitness > self.best_ever_individual.fitness)):
            self.best_ever_individual = best_individual.copy()
        
        execution_time = time.time() - start_time
        
        # Create generation result
        result = GenerationResult(
            generation=self.population.generation,
            best_fitness=stats['best_fitness'],
            worst_fitness=stats['worst_fitness'],
            average_fitness=stats['average_fitness'],
            std_fitness=stats['std_fitness'],
            best_individual=best_individual.copy(),
            execution_time=execution_time
        )
        
        self.results.append(result)
        return result
    
    def run(self, objective_function: Callable, 
            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the complete genetic algorithm.
        
        Args:
            objective_function: Function to optimize
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with final results
        """
        start_time = time.time()
        
        # Initialize population
        self.initialize_population(objective_function)
        
        # Evolution loop
        for generation in range(self.config.num_epochs):
            result = self.evolve_generation(objective_function)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(result)
        
        total_time = time.time() - start_time
        
        # Prepare final results
        final_results = {
            'best_individual': self.best_ever_individual,
            'best_fitness': self.best_ever_individual.fitness,
            'best_solution': self.best_ever_individual.chromosome.decode(),
            'total_generations': len(self.results),
            'total_execution_time': total_time,
            'config': self.config.to_dict(),
            'generation_results': self.results,
            'convergence_data': self._extract_convergence_data()
        }
        
        return final_results
    
    def _extract_convergence_data(self) -> Dict[str, List[float]]:
        """
        Extract convergence data for plotting.
        
        Returns:
            Dictionary with convergence data
        """
        generations = [r.generation for r in self.results]
        best_fitness = [r.best_fitness for r in self.results]
        average_fitness = [r.average_fitness for r in self.results]
        std_fitness = [r.std_fitness for r in self.results]
        
        return {
            'generations': generations,
            'best_fitness': best_fitness,
            'average_fitness': average_fitness,
            'std_fitness': std_fitness
        }
    
    def get_best_solution(self) -> tuple:
        """
        Get the best solution found.
        
        Returns:
            Tuple of (solution_values, fitness_value)
        """
        if self.best_ever_individual is None:
            return None, None
        
        return (self.best_ever_individual.chromosome.decode(), 
                self.best_ever_individual.fitness)


class GARunner:
    """
    Utility class for running multiple GA experiments.
    """
    
    @staticmethod
    def run_experiment(config: GAConfig, objective_function: Callable,
                      num_runs: int = 10, minimize: bool = True,
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run multiple GA experiments for statistical analysis.
        
        Args:
            config: GA configuration
            objective_function: Function to optimize
            num_runs: Number of independent runs
            minimize: True for minimization, False for maximization
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with experiment results
        """
        all_results = []
        best_fitnesses = []
        best_solutions = []
        execution_times = []
        
        for run in range(num_runs):
            # Set different random seed for each run
            random.seed(run * 42)
            
            # Create GA instance
            ga = GeneticAlgorithm(config)
            ga.set_objective(minimize)
            
            # Run GA
            if progress_callback:
                def run_progress_callback(result):
                    progress_callback(run, result)
            else:
                run_progress_callback = None
                
            results = ga.run(objective_function, run_progress_callback)
            
            # Store results
            all_results.append(results)
            best_fitnesses.append(results['best_fitness'])
            best_solutions.append(results['best_solution'])
            execution_times.append(results['total_execution_time'])
        
        # Calculate statistics
        import statistics
        
        experiment_results = {
            'num_runs': num_runs,
            'config': config.to_dict(),
            'all_results': all_results,
            'statistics': {
                'best_fitness': min(best_fitnesses) if minimize else max(best_fitnesses),
                'worst_fitness': max(best_fitnesses) if minimize else min(best_fitnesses),
                'average_fitness': statistics.mean(best_fitnesses),
                'std_fitness': statistics.stdev(best_fitnesses) if len(best_fitnesses) > 1 else 0.0,
                'average_execution_time': statistics.mean(execution_times),
                'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            },
            'best_solutions': best_solutions,
            'execution_times': execution_times
        }
        
        return experiment_results


if __name__ == "__main__":
    # Test genetic algorithm
    print("Testing Genetic Algorithm:")
    print("=" * 50)
    
    # Define test function (sphere function)
    def sphere_function(x):
        """n-dimensional sphere function: f(x) = sum(x_i^2)"""
        return sum(xi**2 for xi in x)
    
    # Configure GA
    config = GAConfig()
    config.set_population_size(50)
    config.set_num_epochs(100)
    config.set_bounds([(-5.0, 5.0), (-5.0, 5.0)])  # 2D problem
    config.set_chromosome_precision(12)
    config.set_elitism_config(enabled=True, count=2)
    
    print(f"Configuration:\n{config}")
    
    # Create and run GA
    ga = GeneticAlgorithm(config)
    ga.set_objective(minimize=True)
    
    # Progress callback
    def progress_callback(result: GenerationResult):
        if result.generation % 20 == 0:
            print(f"Generation {result.generation}: "
                  f"Best = {result.best_fitness:.6f}, "
                  f"Avg = {result.average_fitness:.6f}")
    
    # Run optimization
    results = ga.run(sphere_function, progress_callback)
    
    print(f"\nOptimization completed!")
    print(f"Best solution: {results['best_solution']}")
    print(f"Best fitness: {results['best_fitness']:.6f}")
    print(f"Total time: {results['total_execution_time']:.2f} seconds")
    
    # Test multiple runs
    print(f"\nTesting multiple runs:")
    config.set_num_epochs(50)  # Shorter for testing
    
    experiment_results = GARunner.run_experiment(
        config, sphere_function, num_runs=5, minimize=True
    )
    
    stats = experiment_results['statistics']
    print(f"Results from {experiment_results['num_runs']} runs:")
    print(f"  Best fitness: {stats['best_fitness']:.6f}")
    print(f"  Worst fitness: {stats['worst_fitness']:.6f}")
    print(f"  Average fitness: {stats['average_fitness']:.6f}")
    print(f"  Std fitness: {stats['std_fitness']:.6f}")
    print(f"  Average time: {stats['average_execution_time']:.2f} seconds")