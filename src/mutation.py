import random
from typing import List
from chromosome import Individual
from config import GAConfig, MutationMethod


class MutationOperators:
    """
    Collection of mutation operators for genetic algorithms.
    """
    
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
        if config.mutation_method == MutationMethod.BOUNDARY:
            return MutationOperators.boundary_mutation(individual)
        
        elif config.mutation_method == MutationMethod.ONE_POINT:
            return MutationOperators.one_point_mutation(individual)
        
        elif config.mutation_method == MutationMethod.TWO_POINT:
            return MutationOperators.two_point_mutation(individual)
        
        else:
            raise ValueError(f"Unknown mutation method: {config.mutation_method}")
    
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


if __name__ == "__main__":
    # Test mutation operators
    from chromosome import Chromosome, Individual
    from config import GAConfig, MutationMethod
    
    print("Testing Mutation Operators:")
    print("=" * 50)
    
    # Create test individual
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    precision = 8
    
    chromosome = Chromosome(bounds, precision)
    chromosome.encode([2.0, -1.5])
    individual = Individual(chromosome)
    
    print("Original individual:")
    print(f"Values: {individual.chromosome.decode()}")
    print(f"Genes: {''.join(map(str, individual.chromosome.genes))}")
    
    # Test boundary mutation
    print("\nBoundary mutation:")
    mutated = MutationOperators.boundary_mutation(individual)
    print(f"Values: {mutated.chromosome.decode()}")
    print(f"Genes: {''.join(map(str, mutated.chromosome.genes))}")
    
    # Test one-point mutation
    print("\nOne-point mutation:")
    mutated = MutationOperators.one_point_mutation(individual)
    print(f"Values: {mutated.chromosome.decode()}")
    print(f"Genes: {''.join(map(str, mutated.chromosome.genes))}")
    
    # Test two-point mutation
    print("\nTwo-point mutation:")
    mutated = MutationOperators.two_point_mutation(individual)
    print(f"Values: {mutated.chromosome.decode()}")
    print(f"Genes: {''.join(map(str, mutated.chromosome.genes))}")
    
    # Test inversion operator
    print("\nInversion operator:")
    inverted = InversionOperator.inversion(individual)
    print(f"Values: {inverted.chromosome.decode()}")
    print(f"Genes: {''.join(map(str, inverted.chromosome.genes))}")
    
    # Test with configuration
    print("\nTesting with GAConfig:")
    config = GAConfig()
    config.set_mutation_config(MutationMethod.ONE_POINT, 1.0)
    config.set_inversion_probability(1.0)
    
    population = [individual.copy() for _ in range(5)]
    
    print("Original population:")
    for i, ind in enumerate(population):
        print(f"Individual {i}: {ind.chromosome.decode()}")
    
    mutated_pop = MutationOperators.apply_mutation(population, config)
    print("\nAfter mutation:")
    for i, ind in enumerate(mutated_pop):
        print(f"Individual {i}: {ind.chromosome.decode()}")
    
    inverted_pop = InversionOperator.apply_inversion(mutated_pop, config)
    print("\nAfter inversion:")
    for i, ind in enumerate(inverted_pop):
        print(f"Individual {i}: {ind.chromosome.decode()}")