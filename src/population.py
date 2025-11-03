"""
Population Management Module
Author: kkaczmarski
Date: October 25, 2025

This module implements population management including initialization,
elitism strategy, and population statistics.
"""

from typing import List, Callable
from chromosome import Chromosome, Individual
from config import GAConfig


class Population:
    """
    Population management class for genetic algorithms.
    """
    
    def __init__(self, config: GAConfig):
        """
        Initialize population manager.
        
        Args:
            config: GA configuration
        """
        self.config = config
        self.individuals: List[Individual] = []
        self.generation = 0
        
    def initialize_random(self, objective_function: Callable = None) -> None:
        """
        Initialize population with random individuals.
        
        Args:
            objective_function: Function to evaluate individuals (optional)
        """
        self.individuals = []
        
        for _ in range(self.config.population_size):
            chromosome = Chromosome(self.config.bounds, self.config.chromosome_precision)
            individual = Individual(chromosome)
            
            if objective_function:
                individual.evaluate(objective_function)
                
            self.individuals.append(individual)
        
        self.generation = 0
    
    def evaluate_population(self, objective_function: Callable) -> None:
        """
        Evaluate all individuals in the population.
        
        Args:
            objective_function: Function to evaluate individuals
        """
        for individual in self.individuals:
            if individual.fitness is None:
                individual.evaluate(objective_function)
    
    def get_best_individual(self, minimize: bool = True) -> Individual:
        """
        Get the best individual in the population.
        
        Args:
            minimize: True for minimization, False for maximization
            
        Returns:
            Best individual
        """
        if not self.individuals:
            raise ValueError("Population is empty")
        
        return min(self.individuals, key=lambda ind: ind.fitness) if minimize else \
               max(self.individuals, key=lambda ind: ind.fitness)
    
    def get_worst_individual(self, minimize: bool = True) -> Individual:
        """
        Get the worst individual in the population.
        
        Args:
            minimize: True for minimization, False for maximization
            
        Returns:
            Worst individual
        """
        if not self.individuals:
            raise ValueError("Population is empty")
        
        return max(self.individuals, key=lambda ind: ind.fitness) if minimize else \
               min(self.individuals, key=lambda ind: ind.fitness)
    
    def get_statistics(self) -> dict:
        """
        Get population statistics.
        
        Returns:
            Dictionary with population statistics
        """
        if not self.individuals:
            return {
                'best_fitness': None,
                'worst_fitness': None,
                'average_fitness': None,
                'std_fitness': None,
                'population_size': 0
            }
        
        fitness_values = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        
        if not fitness_values:
            return {
                'best_fitness': None,
                'worst_fitness': None,
                'average_fitness': None,
                'std_fitness': None,
                'population_size': len(self.individuals)
            }
        
        import statistics
        
        return {
            'best_fitness': min(fitness_values),
            'worst_fitness': max(fitness_values),
            'average_fitness': statistics.mean(fitness_values),
            'std_fitness': statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0,
            'population_size': len(self.individuals)
        }
    
    def apply_elitism(self, new_population: List[Individual], minimize: bool = True) -> List[Individual]:
        """
        Apply elitism strategy to preserve best individuals.
        
        Args:
            new_population: New population from genetic operations
            minimize: True for minimization, False for maximization
            
        Returns:
            Population with elitism applied
        """
        if not self.config.elitism_enabled:
            return new_population[:self.config.population_size]
        
        elite_count = self.config.get_elitism_count()
        
        if elite_count == 0:
            return new_population[:self.config.population_size]
        
        # Get elite individuals from current population
        sorted_current = sorted(self.individuals, 
                              key=lambda ind: ind.fitness,
                              reverse=not minimize)
        elite_individuals = sorted_current[:elite_count]
        
        # Combine elite with new population
        combined_population = elite_individuals + new_population
        
        # Sort and select best individuals
        combined_population.sort(key=lambda ind: ind.fitness, reverse=not minimize)
        
        return combined_population[:self.config.population_size]
    
    def update_population(self, new_individuals: List[Individual], 
                         minimize: bool = True) -> None:
        """
        Update population with new individuals.
        
        Args:
            new_individuals: New individuals to replace current population
            minimize: True for minimization, False for maximization
        """
        # Apply elitism if enabled
        self.individuals = self.apply_elitism(new_individuals, minimize)
        self.generation += 1
    
    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)
    
    def __iter__(self):
        """Make population iterable."""
        return iter(self.individuals)
    
    def __getitem__(self, index):
        """Allow indexing of population."""
        return self.individuals[index]


class ElitismStrategy:
    """
    Elitism strategy implementation.
    """
    
    @staticmethod
    def select_elite(population: List[Individual], count: int, 
                    minimize: bool = True) -> List[Individual]:
        """
        Select elite individuals from population.
        
        Args:
            population: List of individuals
            count: Number of elite individuals to select
            minimize: True for minimization, False for maximization
            
        Returns:
            List of elite individuals
        """
        if count <= 0:
            return []
        
        if count >= len(population):
            return population.copy()
        
        # Sort by fitness and select best
        sorted_population = sorted(population, 
                                 key=lambda ind: ind.fitness,
                                 reverse=not minimize)
        
        return [ind.copy() for ind in sorted_population[:count]]
    
    @staticmethod
    def replace_worst(population: List[Individual], elite: List[Individual],
                     minimize: bool = True) -> List[Individual]:
        """
        Replace worst individuals with elite individuals.
        
        Args:
            population: Current population
            elite: Elite individuals to insert
            minimize: True for minimization, False for maximization
            
        Returns:
            Population with elite individuals
        """
        if not elite:
            return population.copy()
        
        # Sort population by fitness (worst first)
        sorted_population = sorted(population, 
                                 key=lambda ind: ind.fitness,
                                 reverse=minimize)
        
        # Replace worst individuals with elite
        elite_count = min(len(elite), len(population))
        
        new_population = elite[:elite_count] + sorted_population[elite_count:]
        
        return new_population


if __name__ == "__main__":
    # Test population management
    from config import GAConfig
    
    print("Testing Population Management:")
    print("=" * 50)
    
    # Create configuration
    config = GAConfig()
    config.set_population_size(10)
    config.set_bounds([(-5.0, 5.0), (-5.0, 5.0)])
    config.set_elitism_config(enabled=True, count=2)
    
    # Simple test function
    def sphere_function(x):
        return sum(xi**2 for xi in x)
    
    # Create and initialize population
    population = Population(config)
    population.initialize_random(sphere_function)
    
    print(f"Initialized population with {len(population)} individuals")
    
    # Get statistics
    stats = population.get_statistics()
    print(f"Population statistics:")
    print(f"  Best fitness: {stats['best_fitness']:.4f}")
    print(f"  Worst fitness: {stats['worst_fitness']:.4f}")
    print(f"  Average fitness: {stats['average_fitness']:.4f}")
    print(f"  Std fitness: {stats['std_fitness']:.4f}")
    
    # Test best individual
    best = population.get_best_individual()
    print(f"\nBest individual: {best.chromosome.decode()}, fitness: {best.fitness:.4f}")
    
    # Test elitism
    print(f"\nTesting elitism:")
    
    # Create some new individuals (potentially worse)
    new_individuals = []
    for _ in range(config.population_size):
        chromosome = Chromosome(config.bounds, config.chromosome_precision)
        individual = Individual(chromosome)
        individual.evaluate(sphere_function)
        new_individuals.append(individual)
    
    print(f"New population stats before elitism:")
    new_stats = {
        'best_fitness': min(ind.fitness for ind in new_individuals),
        'worst_fitness': max(ind.fitness for ind in new_individuals),
        'average_fitness': sum(ind.fitness for ind in new_individuals) / len(new_individuals)
    }
    print(f"  Best: {new_stats['best_fitness']:.4f}")
    print(f"  Average: {new_stats['average_fitness']:.4f}")
    
    # Apply elitism
    elite_population = population.apply_elitism(new_individuals)
    
    elite_stats = {
        'best_fitness': min(ind.fitness for ind in elite_population),
        'worst_fitness': max(ind.fitness for ind in elite_population),
        'average_fitness': sum(ind.fitness for ind in elite_population) / len(elite_population)
    }
    print(f"\nAfter elitism:")
    print(f"  Best: {elite_stats['best_fitness']:.4f}")
    print(f"  Average: {elite_stats['average_fitness']:.4f}")
    
    # Update population
    population.update_population(new_individuals)
    print(f"\nGeneration after update: {population.generation}")