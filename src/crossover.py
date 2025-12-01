import random
from typing import List, Tuple
from chromosome import Individual
from config import GAConfig, CrossoverMethod, ChromosomeType
from real_crossover import RealCrossover


class CrossoverOperators:
    """
    Collection of crossover operators for genetic algorithms.
    Supports both binary and real-valued chromosome representations.
    """
    
    @staticmethod
    def perform_crossover(parent1: Individual, parent2: Individual, config: GAConfig) -> Tuple[Individual, Individual]:
        """
        Perform crossover based on configuration.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            config: GA configuration containing crossover method and parameters
            
        Returns:
            Tuple of two offspring individuals
        """
        chromosome_type = getattr(config, 'chromosome_type', ChromosomeType.BINARY)
        
        if chromosome_type == ChromosomeType.REAL:
            return CrossoverOperators._real_crossover(parent1, parent2, config)
        else:
            return CrossoverOperators._binary_crossover(parent1, parent2, config)
    
    @staticmethod
    def _binary_crossover(parent1: Individual, parent2: Individual, config: GAConfig) -> Tuple[Individual, Individual]:
        """Perform binary crossover operations."""
        method = config.crossover_method
        
        if method == CrossoverMethod.ONE_POINT:
            return CrossoverOperators.one_point_crossover(parent1, parent2)
        elif method == CrossoverMethod.TWO_POINT:
            return CrossoverOperators.two_point_crossover(parent1, parent2)
        elif method == CrossoverMethod.UNIFORM:
            return CrossoverOperators.uniform_crossover(parent1, parent2)
        elif method == CrossoverMethod.DISCRETE:
            return CrossoverOperators.discrete_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown binary crossover method: {method}")
    
    @staticmethod 
    def _real_crossover(parent1: Individual, parent2: Individual, config: GAConfig) -> Tuple[Individual, Individual]:
        """Perform real-valued crossover operations."""
        method = config.crossover_method
        
        # Extract real chromosomes
        real_chrom1 = parent1.chromosome
        real_chrom2 = parent2.chromosome
        
        # Perform crossover
        if method == CrossoverMethod.ARITHMETIC:
            offspring_chrom1, offspring_chrom2 = RealCrossover.arithmetic_crossover(
                real_chrom1, real_chrom2, config.alpha)
        elif method == CrossoverMethod.LINEAR:
            offspring_chrom1, offspring_chrom2 = RealCrossover.linear_crossover(
                real_chrom1, real_chrom2)
        elif method == CrossoverMethod.BLEND_ALPHA:
            offspring_chrom1, offspring_chrom2 = RealCrossover.blend_alpha_crossover(
                real_chrom1, real_chrom2, config.alpha)
        elif method == CrossoverMethod.BLEND_ALPHA_BETA:
            offspring_chrom1, offspring_chrom2 = RealCrossover.blend_alpha_beta_crossover(
                real_chrom1, real_chrom2, config.alpha, config.beta)
        elif method == CrossoverMethod.AVERAGING:
            offspring_chrom1, offspring_chrom2 = RealCrossover.averaging_crossover(
                real_chrom1, real_chrom2)
        else:
            raise ValueError(f"Unknown real crossover method: {method}")
        
        # Create new individuals
        offspring1 = Individual(offspring_chrom1, None)  # fitness will be calculated later
        offspring2 = Individual(offspring_chrom2, None)
        
        return offspring1, offspring2

    @staticmethod
    def one_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        One-point crossover operator.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        # Create copies for offspring
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Get chromosome length
        chromosome_length = len(parent1.chromosome)
        
        if chromosome_length <= 1:
            return offspring1, offspring2
        
        # Choose random crossover point
        crossover_point = random.randint(1, chromosome_length - 1)
        
        # Perform crossover
        offspring1.chromosome.genes = (parent1.chromosome.genes[:crossover_point] + 
                                     parent2.chromosome.genes[crossover_point:])
        offspring2.chromosome.genes = (parent2.chromosome.genes[:crossover_point] + 
                                     parent1.chromosome.genes[crossover_point:])
        
        # Reset fitness values
        offspring1.fitness = None
        offspring1.objective_value = None
        offspring2.fitness = None
        offspring2.objective_value = None
        
        return offspring1, offspring2
    
    @staticmethod
    def two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Two-point crossover operator.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        # Create copies for offspring
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Get chromosome length
        chromosome_length = len(parent1.chromosome)
        
        if chromosome_length <= 2:
            return CrossoverOperators.one_point_crossover(parent1, parent2)
        
        # Choose two random crossover points
        point1 = random.randint(1, chromosome_length - 2)
        point2 = random.randint(point1 + 1, chromosome_length - 1)
        
        # Perform crossover (swap middle segment)
        offspring1.chromosome.genes = (parent1.chromosome.genes[:point1] + 
                                     parent2.chromosome.genes[point1:point2] + 
                                     parent1.chromosome.genes[point2:])
        offspring2.chromosome.genes = (parent2.chromosome.genes[:point1] + 
                                     parent1.chromosome.genes[point1:point2] + 
                                     parent2.chromosome.genes[point2:])
        
        # Reset fitness values
        offspring1.fitness = None
        offspring1.objective_value = None
        offspring2.fitness = None
        offspring2.objective_value = None
        
        return offspring1, offspring2
    
    @staticmethod
    def uniform_crossover(parent1: Individual, parent2: Individual, 
                         swap_probability: float = 0.5) -> Tuple[Individual, Individual]:
        """
        Uniform crossover operator.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            swap_probability: Probability of swapping each gene
            
        Returns:
            Tuple of two offspring individuals
        """
        # Create copies for offspring
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Get chromosome length
        chromosome_length = len(parent1.chromosome)
        
        # Perform uniform crossover
        for i in range(chromosome_length):
            if random.random() < swap_probability:
                # Swap genes at position i
                offspring1.chromosome.genes[i] = parent2.chromosome.genes[i]
                offspring2.chromosome.genes[i] = parent1.chromosome.genes[i]
        
        # Reset fitness values
        offspring1.fitness = None
        offspring1.objective_value = None
        offspring2.fitness = None
        offspring2.objective_value = None
        
        return offspring1, offspring2
    
    @staticmethod
    def discrete_crossover(parent1: Individual, parent2: Individual, 
                          alpha: float = 0.5) -> Tuple[Individual, Individual]:
        """
        Discrete (arithmetic) crossover operator.
        Works on decoded real values, then encodes back to binary.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            alpha: Blending factor [0, 1]
            
        Returns:
            Tuple of two offspring individuals
        """
        # Create copies for offspring
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Decode parent chromosomes to real values
        values1 = parent1.chromosome.decode()
        values2 = parent2.chromosome.decode()
        
        # Perform arithmetic crossover
        offspring_values1 = []
        offspring_values2 = []
        
        for v1, v2 in zip(values1, values2):
            # Linear combination
            new_v1 = alpha * v1 + (1 - alpha) * v2
            new_v2 = alpha * v2 + (1 - alpha) * v1
            
            offspring_values1.append(new_v1)
            offspring_values2.append(new_v2)
        
        # Encode back to binary chromosomes
        offspring1.chromosome.encode(offspring_values1)
        offspring2.chromosome.encode(offspring_values2)
        
        # Reset fitness values
        offspring1.fitness = None
        offspring1.objective_value = None
        offspring2.fitness = None
        offspring2.objective_value = None
        
        return offspring1, offspring2
    
    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, 
                 config: GAConfig) -> Tuple[Individual, Individual]:
        """
        Perform crossover using configured method.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            config: GA configuration
            
        Returns:
            Tuple of two offspring individuals
        """
        return CrossoverOperators.perform_crossover(parent1, parent2, config)
    
    @staticmethod
    def apply_crossover(parents: List[Individual], config: GAConfig) -> List[Individual]:
        """
        Apply crossover to population of parents.
        
        Args:
            parents: List of parent individuals
            config: GA configuration
            
        Returns:
            List of offspring individuals
        """
        offspring = []
        
        # Ensure even number of parents
        if len(parents) % 2 == 1:
            parents = parents + [parents[-1].copy()]  # Duplicate last parent
        
        # Apply crossover to pairs
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if random.random() < config.crossover_probability:
                child1, child2 = CrossoverOperators.crossover(parent1, parent2, config)
                offspring.extend([child1, child2])
            else:
                # No crossover, just copy parents
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring
