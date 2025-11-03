"""
Genetic Algorithm Framework - Chromosome Implementation
Author: kkaczmarski
Date: October 25, 2025

This module implements a binary chromosome representation for genetic algorithms
with configurable precision and support for multiple variables.
"""

import random
from typing import List, Tuple


class Chromosome:
    """
    Binary chromosome representation with configurable precision.
    
    Attributes:
        genes (List[int]): Binary genes representing the chromosome
        bounds (List[Tuple[float, float]]): Variable bounds for each dimension
        precision (int): Number of bits per variable
        num_variables (int): Number of variables (dimensions)
    """
    
    def __init__(self, bounds: List[Tuple[float, float]], precision: int = 10):
        """
        Initialize chromosome with given bounds and precision.
        
        Args:
            bounds: List of (min, max) tuples for each variable
            precision: Number of bits per variable (default: 10)
        """
        self.bounds = bounds
        self.precision = precision
        self.num_variables = len(bounds)
        self.chromosome_length = self.num_variables * self.precision
        self.genes = self._generate_random_genes()
        
    def _generate_random_genes(self) -> List[int]:
        """Generate random binary genes."""
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]
    
    def decode(self) -> List[float]:
        """
        Decode binary chromosome to real values.
        
        Returns:
            List of decoded real values
        """
        decoded_values = []
        
        for i in range(self.num_variables):
            start_idx = i * self.precision
            end_idx = (i + 1) * self.precision
            
            # Extract bits for current variable
            bits = self.genes[start_idx:end_idx]
            
            # Convert binary to decimal
            decimal_value = sum(bit * (2 ** (self.precision - 1 - j)) for j, bit in enumerate(bits))
            
            # Scale to bounds
            min_val, max_val = self.bounds[i]
            max_decimal = (2 ** self.precision) - 1
            real_value = min_val + (decimal_value / max_decimal) * (max_val - min_val)
            
            decoded_values.append(real_value)
            
        return decoded_values
    
    def encode(self, values: List[float]) -> None:
        """
        Encode real values to binary chromosome.
        
        Args:
            values: List of real values to encode
        """
        if len(values) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} values, got {len(values)}")
        
        self.genes = []
        
        for i, value in enumerate(values):
            min_val, max_val = self.bounds[i]
            
            # Clamp value to bounds
            value = max(min_val, min(max_val, value))
            
            # Scale to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            
            # Convert to decimal
            max_decimal = (2 ** self.precision) - 1
            decimal_value = int(normalized * max_decimal)
            
            # Convert to binary
            binary_str = format(decimal_value, f'0{self.precision}b')
            bits = [int(bit) for bit in binary_str]
            
            self.genes.extend(bits)
    
    def copy(self) -> 'Chromosome':
        """Create a deep copy of the chromosome."""
        new_chromosome = Chromosome(self.bounds, self.precision)
        new_chromosome.genes = self.genes.copy()
        return new_chromosome
    
    def __str__(self) -> str:
        """String representation of chromosome."""
        decoded = self.decode()
        return f"Chromosome(genes={''.join(map(str, self.genes))}, decoded={decoded})"
    
    def __len__(self) -> int:
        """Return chromosome length."""
        return self.chromosome_length


class Individual:
    """
    Individual in the population containing chromosome and fitness value.
    """
    
    def __init__(self, chromosome: Chromosome):
        """
        Initialize individual with chromosome.
        
        Args:
            chromosome: Chromosome instance
        """
        self.chromosome = chromosome
        self.fitness = None
        self.objective_value = None
        
    def evaluate(self, objective_function) -> float:
        """
        Evaluate individual using objective function.
        
        Args:
            objective_function: Function to evaluate chromosome
            
        Returns:
            Fitness value
        """
        decoded_values = self.chromosome.decode()
        self.objective_value = objective_function(decoded_values)
        self.fitness = self.objective_value  # For minimization problems
        return self.fitness
    
    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        new_individual = Individual(self.chromosome.copy())
        new_individual.fitness = self.fitness
        new_individual.objective_value = self.objective_value
        return new_individual
    
    def __str__(self) -> str:
        """String representation of individual."""
        return f"Individual(fitness={self.fitness}, values={self.chromosome.decode()})"


if __name__ == "__main__":
    # Test chromosome implementation
    bounds = [(-5.0, 5.0), (-10.0, 10.0), (0.0, 1.0)]
    precision = 8
    
    print("Testing Chromosome Implementation:")
    print("=" * 50)
    
    # Create and test chromosome
    chromosome = Chromosome(bounds, precision)
    print(f"Original chromosome: {chromosome}")
    print(f"Chromosome length: {len(chromosome)}")
    
    # Test encoding/decoding
    test_values = [2.5, -3.7, 0.8]
    print(f"\nEncoding values: {test_values}")
    chromosome.encode(test_values)
    decoded = chromosome.decode()
    print(f"Decoded values: {decoded}")
    
    # Test individual
    print(f"\nTesting Individual:")
    individual = Individual(chromosome)
    
    # Simple test function (sphere function)
    def sphere_function(x):
        return sum(xi**2 for xi in x)
    
    fitness = individual.evaluate(sphere_function)
    print(f"Individual: {individual}")
    print(f"Fitness: {fitness}")
