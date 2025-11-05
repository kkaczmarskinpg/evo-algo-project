import random
from typing import List, Tuple
from chromosome import Individual
from config import GAConfig, CrossoverMethod


class CrossoverOperators:
    """
    Collection of crossover operators for genetic algorithms.
    """
    
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
        if config.crossover_method == CrossoverMethod.ONE_POINT:
            return CrossoverOperators.one_point_crossover(parent1, parent2)
        
        elif config.crossover_method == CrossoverMethod.TWO_POINT:
            return CrossoverOperators.two_point_crossover(parent1, parent2)
        
        elif config.crossover_method == CrossoverMethod.UNIFORM:
            return CrossoverOperators.uniform_crossover(parent1, parent2)
        
        elif config.crossover_method == CrossoverMethod.DISCRETE:
            return CrossoverOperators.discrete_crossover(parent1, parent2)
        
        else:
            raise ValueError(f"Unknown crossover method: {config.crossover_method}")
    
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


if __name__ == "__main__":
    # Test crossover operators
    from chromosome import Chromosome, Individual
    from config import GAConfig, CrossoverMethod
    
    print("Testing Crossover Operators:")
    print("=" * 50)
    
    # Create test parents
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    precision = 8
    
    chromosome1 = Chromosome(bounds, precision)
    chromosome1.encode([2.0, -1.5])
    parent1 = Individual(chromosome1)
    
    chromosome2 = Chromosome(bounds, precision)
    chromosome2.encode([-3.0, 4.0])
    parent2 = Individual(chromosome2)
    
    print("Parent chromosomes:")
    print(f"Parent 1: {parent1.chromosome.decode()}")
    print(f"Parent 2: {parent2.chromosome.decode()}")
    print(f"Parent 1 genes: {''.join(map(str, parent1.chromosome.genes))}")
    print(f"Parent 2 genes: {''.join(map(str, parent2.chromosome.genes))}")
    
    # Test one-point crossover
    print("\nOne-point crossover:")
    child1, child2 = CrossoverOperators.one_point_crossover(parent1, parent2)
    print(f"Child 1: {child1.chromosome.decode()}")
    print(f"Child 2: {child2.chromosome.decode()}")
    print(f"Child 1 genes: {''.join(map(str, child1.chromosome.genes))}")
    print(f"Child 2 genes: {''.join(map(str, child2.chromosome.genes))}")
    
    # Test two-point crossover
    print("\nTwo-point crossover:")
    child1, child2 = CrossoverOperators.two_point_crossover(parent1, parent2)
    print(f"Child 1: {child1.chromosome.decode()}")
    print(f"Child 2: {child2.chromosome.decode()}")
    
    # Test uniform crossover
    print("\nUniform crossover:")
    child1, child2 = CrossoverOperators.uniform_crossover(parent1, parent2)
    print(f"Child 1: {child1.chromosome.decode()}")
    print(f"Child 2: {child2.chromosome.decode()}")
    
    # Test discrete crossover
    print("\nDiscrete crossover:")
    child1, child2 = CrossoverOperators.discrete_crossover(parent1, parent2)
    print(f"Child 1: {child1.chromosome.decode()}")
    print(f"Child 2: {child2.chromosome.decode()}")
    
    # Test with configuration
    print("\nTesting with GAConfig:")
    config = GAConfig()
    config.set_crossover_config(CrossoverMethod.TWO_POINT, 1.0)
    
    parents = [parent1, parent2, parent1.copy(), parent2.copy()]
    offspring = CrossoverOperators.apply_crossover(parents, config)
    
    print(f"Generated {len(offspring)} offspring from {len(parents)} parents")
    for i, child in enumerate(offspring):
        print(f"Offspring {i}: {child.chromosome.decode()}")