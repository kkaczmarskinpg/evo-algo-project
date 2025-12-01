import random
from typing import List
from chromosome import Individual
from config import GAConfig, MutationMethod, ChromosomeType


class MutationOperators:
    """
    Collection of mutation operators for genetic algorithms.
    Supports both binary and real-valued chromosome representations.
    """
    
    @staticmethod
    def perform_mutation(individual: Individual, config: GAConfig) -> Individual:
        """
        Perform mutation based on configuration.
        
        Args:
            individual: Individual to mutate
            config: GA configuration containing mutation method and parameters
            
        Returns:
            Mutated individual
        """
        chromosome_type = getattr(config, 'chromosome_type', ChromosomeType.BINARY)
        
        if chromosome_type == ChromosomeType.REAL:
            return MutationOperators._real_mutation(individual, config)
        else:
            return MutationOperators._binary_mutation(individual, config)
    
    @staticmethod
    def _binary_mutation(individual: Individual, config: GAConfig) -> Individual:
        """Perform binary mutation operations."""
        method = config.mutation_method
        
        if method == MutationMethod.BOUNDARY:
            return MutationOperators.boundary_mutation(individual)
        elif method == MutationMethod.ONE_POINT:
            return MutationOperators.one_point_mutation(individual)
        elif method == MutationMethod.TWO_POINT:
            return MutationOperators.two_point_mutation(individual)
        else:
            raise ValueError(f"Unknown binary mutation method: {method}")
    
    @staticmethod
    def _real_mutation(individual: Individual, config: GAConfig) -> Individual:
        """Perform real-valued mutation operations."""
        mutated = individual.copy()
        method = config.mutation_method
        probability = config.mutation_probability
        
        if method == MutationMethod.UNIFORM:
            mutated.chromosome.mutate_uniform(
                probability, 
                getattr(config, 'mutation_strength', 0.1)
            )
        elif method == MutationMethod.GAUSSIAN:
            mutated.chromosome.mutate_gaussian(
                probability,
                getattr(config, 'gaussian_std', 0.1)
            )
        else:
            raise ValueError(f"Unknown real mutation method: {method}")
        
        # Reset fitness
        mutated.fitness = None
        mutated.objective_value = None
        
        return mutated
    
    @staticmethod
    def boundary_mutation(individual: Individual) -> Individual:
        """
        Boundary mutation - sets a random variable to its boundary value.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        # Choose random variable to mutate
        var_index = random.randint(0, mutated.chromosome.num_variables - 1)
        
        # Get bounds for the variable
        min_val, max_val = mutated.chromosome.bounds[var_index]
        
        # Choose boundary value (min or max)
        boundary_value = random.choice([min_val, max_val])
        
        # Decode current values
        current_values = mutated.chromosome.decode()
        
        # Set variable to boundary value
        current_values[var_index] = boundary_value
        
        # Encode back to chromosome
        mutated.chromosome.encode(current_values)
        
        # Reset fitness
        mutated.fitness = None
        mutated.objective_value = None
        
        return mutated
    
    @staticmethod
    def one_point_mutation(individual: Individual) -> Individual:
        """
        One-point mutation - flips a single random bit.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        # Choose random gene to flip
        gene_index = random.randint(0, len(mutated.chromosome.genes) - 1)
        
        # Flip the bit
        mutated.chromosome.genes[gene_index] = 1 - mutated.chromosome.genes[gene_index]
        
        # Reset fitness
        mutated.fitness = None
        mutated.objective_value = None
        
        return mutated
    
    @staticmethod
    def two_point_mutation(individual: Individual) -> Individual:
        """
        Two-point mutation - flips two random bits.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        chromosome_length = len(mutated.chromosome.genes)
        
        if chromosome_length < 2:
            return MutationOperators.one_point_mutation(individual)
        
        # Choose two different random genes to flip
        indices = random.sample(range(chromosome_length), 2)
        
        for index in indices:
            mutated.chromosome.genes[index] = 1 - mutated.chromosome.genes[index]
        
        # Reset fitness
        mutated.fitness = None
        mutated.objective_value = None
        
        return mutated
    
    @staticmethod
    def mutate(individual: Individual, config: GAConfig) -> Individual:
        """
        Apply mutation using configured method.
        
        Args:
            individual: Individual to mutate
            config: GA configuration
            
        Returns:
            Mutated individual
        """
        return MutationOperators.perform_mutation(individual, config)
    
    @staticmethod
    def apply_mutation(population: List[Individual], config: GAConfig) -> List[Individual]:
        """
        Apply mutation to population.
        
        Args:
            population: List of individuals
            config: GA configuration
            
        Returns:
            List of potentially mutated individuals
        """
        mutated_population = []
        
        for individual in population:
            if random.random() < config.mutation_probability:
                mutated_individual = MutationOperators.mutate(individual, config)
                mutated_population.append(mutated_individual)
            else:
                mutated_population.append(individual.copy())
        
        return mutated_population


class InversionOperator:
    """
    Inversion operator for genetic algorithms.
    """
    
    @staticmethod
    def inversion(individual: Individual) -> Individual:
        """
        Apply inversion operator - reverses a random segment of the chromosome.
        
        Args:
            individual: Individual to apply inversion to
            
        Returns:
            Individual with inverted segment
        """
        inverted = individual.copy()
        
        chromosome_length = len(inverted.chromosome.genes)
        
        if chromosome_length < 2:
            return inverted
        
        # Choose two random points
        point1 = random.randint(0, chromosome_length - 1)
        point2 = random.randint(0, chromosome_length - 1)
        
        # Ensure point1 <= point2
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Reverse the segment between the points
        segment = inverted.chromosome.genes[point1:point2 + 1]
        segment.reverse()
        inverted.chromosome.genes[point1:point2 + 1] = segment
        
        # Reset fitness
        inverted.fitness = None
        inverted.objective_value = None
        
        return inverted
    
    @staticmethod
    def apply_inversion(population: List[Individual], config: GAConfig) -> List[Individual]:
        """
        Apply inversion operator to population.
        
        Args:
            population: List of individuals
            config: GA configuration
            
        Returns:
            List of potentially inverted individuals
        """
        inverted_population = []
        
        for individual in population:
            if random.random() < config.inversion_probability:
                inverted_individual = InversionOperator.inversion(individual)
                inverted_population.append(inverted_individual)
            else:
                inverted_population.append(individual.copy())
        
        return inverted_population
