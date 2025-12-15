import random
import numpy as np
from typing import List, Tuple


class RealChromosome:
    """
    Real-valued chromosome representation.
    
    Attributes:
        genes (List[float]): Real-valued genes representing the chromosome
        bounds (List[Tuple[float, float]]): Variable bounds for each dimension
        num_variables (int): Number of variables (dimensions)
    """
    
    def __init__(self, bounds: List[Tuple[float, float]]):
        """
        Initialize real chromosome with given bounds.
        
        Args:
            bounds: List of (min, max) tuples for each variable
        """
        self.bounds = bounds
        self.num_variables = len(bounds)
        self.genes = self._generate_random_genes()
        self.fitness = None
        
    def _generate_random_genes(self) -> List[float]:
        """Generate random real-valued genes within bounds."""
        genes = []
        for min_val, max_val in self.bounds:
            genes.append(random.uniform(min_val, max_val))
        return genes
    
    def decode(self) -> List[float]:
        """
        Return the real values (genes are already in real form).
        
        Returns:
            List of real values
        """
        return self.genes.copy()
    
    def copy(self) -> 'RealChromosome':
        """
        Create a copy of this chromosome.
        
        Returns:
            New RealChromosome instance with same genes and bounds
        """
        new_chromosome = RealChromosome(self.bounds)
        new_chromosome.genes = self.genes.copy()
        new_chromosome.fitness = self.fitness
        return new_chromosome
    
    def set_genes(self, genes: List[float]) -> None:
        """
        Set genes ensuring they are within bounds.
        
        Args:
            genes: List of real values to set as genes
        """
        if len(genes) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} genes, got {len(genes)}")
        
        # Ensure genes are within bounds
        self.genes = []
        for i, gene in enumerate(genes):
            min_val, max_val = self.bounds[i]
            # Clip to bounds
            clipped_gene = max(min_val, min(max_val, gene))
            self.genes.append(clipped_gene)
        
        # Reset fitness when genes change
        self.fitness = None
    
    def mutate_uniform(self, probability: float, mutation_strength: float = 0.1) -> None:
        """
        Apply uniform mutation to the chromosome.
        
        Args:
            probability: Probability of mutating each gene
            mutation_strength: Strength of mutation (as fraction of range)
        """
        for i in range(self.num_variables):
            if random.random() < probability:
                min_val, max_val = self.bounds[i]
                range_val = max_val - min_val
                
                # Add uniform random noise
                noise = random.uniform(-mutation_strength * range_val, 
                                     mutation_strength * range_val)
                new_gene = self.genes[i] + noise
                
                # Ensure within bounds
                self.genes[i] = max(min_val, min(max_val, new_gene))
        
        # Reset fitness after mutation
        self.fitness = None
    
    def mutate_gaussian(self, probability: float, std_dev: float = 0.1) -> None:
        """
        Apply Gaussian (normal) mutation to the chromosome.
        
        Args:
            probability: Probability of mutating each gene
            std_dev: Standard deviation for Gaussian noise
        """
        for i in range(self.num_variables):
            if random.random() < probability:
                min_val, max_val = self.bounds[i]
                range_val = max_val - min_val
                
                # Add Gaussian random noise
                noise = random.gauss(0, std_dev * range_val)
                new_gene = self.genes[i] + noise
                
                # Ensure within bounds
                self.genes[i] = max(min_val, min(max_val, new_gene))
        
        # Reset fitness after mutation
        self.fitness = None
    
    def __str__(self) -> str:
        """String representation of chromosome."""
        genes_str = [f"{gene:.4f}" for gene in self.genes]
        fitness_str = f"{self.fitness:.6f}" if self.fitness is not None else "None"
        return f"RealChromosome(genes=[{', '.join(genes_str)}], fitness={fitness_str})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
    
    def __len__(self) -> int:
        """Return number of genes."""
        return len(self.genes)
    
    def __getitem__(self, index: int) -> float:
        """Get gene at index."""
        return self.genes[index]
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set gene at index (with bounds checking)."""
        if 0 <= index < len(self.genes):
            min_val, max_val = self.bounds[index]
            self.genes[index] = max(min_val, min(max_val, value))
            self.fitness = None  # Reset fitness
        else:
            raise IndexError("Gene index out of range")
