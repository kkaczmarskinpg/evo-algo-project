import random
from typing import List
from chromosome import Individual
from config import GAConfig, SelectionMethod


class SelectionOperators:
    """
    Collection of selection operators for genetic algorithms.
    """
    
    @staticmethod
    def best_selection(population: List[Individual], num_parents: int, 
                      minimize: bool = True) -> List[Individual]:
        """
        Select best individuals from population.
        
        Args:
            population: List of individuals
            num_parents: Number of parents to select
            minimize: True for minimization, False for maximization
            
        Returns:
            List of selected parents
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        if num_parents > len(population):
            raise ValueError("Number of parents cannot exceed population size")
        
        # Sort population by fitness
        sorted_population = sorted(population, 
                                 key=lambda ind: ind.fitness, 
                                 reverse=not minimize)
        
        return sorted_population[:num_parents]
    
    @staticmethod
    def roulette_wheel_selection(population: List[Individual], num_parents: int,
                                minimize: bool = True, selection_pressure: float = 2.0) -> List[Individual]:
        """
        Roulette wheel selection (fitness proportionate selection).
        
        Args:
            population: List of individuals
            num_parents: Number of parents to select
            minimize: True for minimization, False for maximization
            selection_pressure: Selection pressure factor
            
        Returns:
            List of selected parents
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        if num_parents > len(population):
            raise ValueError("Number of parents cannot exceed population size")
        
        # Calculate fitness values for selection
        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
        
        if not fitness_values:
            raise ValueError("No individuals with fitness values found")
        
        # For minimization, we need to transform fitness values
        if minimize:
            # Use ranking-based selection for better numerical stability
            sorted_indices = sorted(range(len(population)), 
                                  key=lambda i: population[i].fitness)
            
            # Assign selection probabilities based on rank
            probabilities = []
            total_rank = len(population) * (len(population) + 1) / 2
            
            for i, idx in enumerate(sorted_indices):
                # Higher rank (better fitness) gets higher probability
                rank = len(population) - i
                prob = (selection_pressure * rank + (2 - selection_pressure)) / (2 * len(population))
                probabilities.append((prob, population[idx]))
        else:
            # For maximization, use fitness values directly
            min_fitness = min(fitness_values)
            if min_fitness < 0:
                # Shift fitness values to be non-negative
                adjusted_fitness = [f - min_fitness + 1e-10 for f in fitness_values]
            else:
                adjusted_fitness = [f + 1e-10 for f in fitness_values]  # Avoid division by zero
            
            total_fitness = sum(adjusted_fitness)
            probabilities = [(f / total_fitness, ind) for f, ind in 
                           zip(adjusted_fitness, population)]
        
        # Normalize probabilities
        total_prob = sum(prob for prob, _ in probabilities)
        probabilities = [(prob / total_prob, ind) for prob, ind in probabilities]
        
        # Select parents using roulette wheel
        selected_parents = []
        
        for _ in range(num_parents):
            r = random.random()
            cumulative_prob = 0.0
            
            for prob, individual in probabilities:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected_parents.append(individual.copy())
                    break
            else:
                # Fallback: select last individual if rounding errors occur
                selected_parents.append(probabilities[-1][1].copy())
        
        return selected_parents
    
    @staticmethod
    def tournament_selection(population: List[Individual], num_parents: int,
                           tournament_size: int = 3, minimize: bool = True) -> List[Individual]:
        """
        Tournament selection.
        
        Args:
            population: List of individuals
            num_parents: Number of parents to select
            tournament_size: Size of tournament
            minimize: True for minimization, False for maximization
            
        Returns:
            List of selected parents
        """
        if not population:
            raise ValueError("Population cannot be empty")
        
        if num_parents > len(population):
            raise ValueError("Number of parents cannot exceed population size")
        
        if tournament_size > len(population):
            raise ValueError("Tournament size cannot exceed population size")
        
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        
        selected_parents = []
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_candidates = random.sample(population, tournament_size)
            
            # Find best individual in tournament
            best_individual = min(tournament_candidates, 
                                key=lambda ind: ind.fitness) if minimize else \
                             max(tournament_candidates, 
                                key=lambda ind: ind.fitness)
            
            selected_parents.append(best_individual.copy())
        
        return selected_parents
    
    @staticmethod
    def select_parents(population: List[Individual], config: GAConfig, 
                      num_parents: int = None, minimize: bool = True) -> List[Individual]:
        """
        Select parents using configured selection method.
        
        Args:
            population: List of individuals
            config: GA configuration
            num_parents: Number of parents to select (default: population size)
            minimize: True for minimization, False for maximization
            
        Returns:
            List of selected parents
        """
        if num_parents is None:
            num_parents = len(population)
        
        if config.selection_method == SelectionMethod.BEST:
            return SelectionOperators.best_selection(population, num_parents, minimize)
        
        elif config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return SelectionOperators.roulette_wheel_selection(
                population, num_parents, minimize, config.selection_pressure)
        
        elif config.selection_method == SelectionMethod.TOURNAMENT:
            return SelectionOperators.tournament_selection(
                population, num_parents, config.tournament_size, minimize)
        
        else:
            raise ValueError(f"Unknown selection method: {config.selection_method}")
