import random
import numpy as np
from typing import List, Tuple
from real_chromosome import RealChromosome


class RealCrossover:
    """Real-valued crossover operators for genetic algorithms."""
    
    @staticmethod
    def arithmetic_crossover(parent1: RealChromosome, parent2: RealChromosome, 
                           alpha: float = 0.5) -> Tuple[RealChromosome, RealChromosome]:
        """
        Arithmetic crossover: offspring = alpha * parent1 + (1-alpha) * parent2
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome  
            alpha: Blending factor (0.0 to 1.0)
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        genes1 = []
        genes2 = []
        
        for i in range(len(parent1)):
            # Arithmetic combination
            gene1 = alpha * parent1[i] + (1 - alpha) * parent2[i]
            gene2 = (1 - alpha) * parent1[i] + alpha * parent2[i]
            
            genes1.append(gene1)
            genes2.append(gene2)
        
        offspring1.set_genes(genes1)
        offspring2.set_genes(genes2)
        
        return offspring1, offspring2
    
    @staticmethod
    def linear_crossover(parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        """
        Linear crossover: creates three offspring and selects two best ones.
        offspring1 = 0.5 * parent1 + 0.5 * parent2
        offspring2 = 1.5 * parent1 - 0.5 * parent2  
        offspring3 = -0.5 * parent1 + 1.5 * parent2
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two best offspring chromosomes
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")
        
        # Create three potential offspring
        offspring1 = parent1.copy()
        offspring2 = parent1.copy() 
        offspring3 = parent1.copy()
        
        genes1, genes2, genes3 = [], [], []
        
        for i in range(len(parent1)):
            # Linear combinations
            gene1 = 0.5 * parent1[i] + 0.5 * parent2[i]
            gene2 = 1.5 * parent1[i] - 0.5 * parent2[i]
            gene3 = -0.5 * parent1[i] + 1.5 * parent2[i]
            
            genes1.append(gene1)
            genes2.append(gene2) 
            genes3.append(gene3)
        
        offspring1.set_genes(genes1)
        offspring2.set_genes(genes2)
        offspring3.set_genes(genes3)
        
        # Return the first two (in practice, we'd evaluate fitness to select best)
        return offspring1, offspring2
    
    @staticmethod
    def blend_alpha_crossover(parent1: RealChromosome, parent2: RealChromosome,
                            alpha: float = 0.5) -> Tuple[RealChromosome, RealChromosome]:
        """
        Blend-alpha crossover (BLX-α): offspring genes are uniformly distributed
        in extended range around parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            alpha: Extension factor for range
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        genes1, genes2 = [], []
        
        for i in range(len(parent1)):
            # Get parent genes
            p1_gene = parent1[i]
            p2_gene = parent2[i]
            
            # Calculate range
            min_gene = min(p1_gene, p2_gene)
            max_gene = max(p1_gene, p2_gene)
            range_val = max_gene - min_gene
            
            # Extend range by alpha
            extended_min = min_gene - alpha * range_val
            extended_max = max_gene + alpha * range_val
            
            # Ensure within chromosome bounds
            bound_min, bound_max = parent1.bounds[i]
            extended_min = max(bound_min, extended_min)
            extended_max = min(bound_max, extended_max)
            
            # Generate offspring genes
            gene1 = random.uniform(extended_min, extended_max)
            gene2 = random.uniform(extended_min, extended_max)
            
            genes1.append(gene1)
            genes2.append(gene2)
        
        offspring1.set_genes(genes1)
        offspring2.set_genes(genes2)
        
        return offspring1, offspring2
    
    @staticmethod
    def blend_alpha_beta_crossover(parent1: RealChromosome, parent2: RealChromosome,
                                 alpha: float = 0.5, beta: float = 0.5) -> Tuple[RealChromosome, RealChromosome]:
        """
        Blend-alpha-beta crossover (BLX-αβ): asymmetric extension of parent range.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            alpha: Extension factor for lower bound
            beta: Extension factor for upper bound
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        genes1, genes2 = [], []
        
        for i in range(len(parent1)):
            # Get parent genes
            p1_gene = parent1[i]
            p2_gene = parent2[i]
            
            # Calculate range with asymmetric extension
            min_gene = min(p1_gene, p2_gene)
            max_gene = max(p1_gene, p2_gene)
            range_val = max_gene - min_gene
            
            # Asymmetric extension
            extended_min = min_gene - alpha * range_val
            extended_max = max_gene + beta * range_val
            
            # Ensure within chromosome bounds
            bound_min, bound_max = parent1.bounds[i]
            extended_min = max(bound_min, extended_min)
            extended_max = min(bound_max, extended_max)
            
            # Generate offspring genes
            gene1 = random.uniform(extended_min, extended_max)
            gene2 = random.uniform(extended_min, extended_max)
            
            genes1.append(gene1)
            genes2.append(gene2)
        
        offspring1.set_genes(genes1)
        offspring2.set_genes(genes2)
        
        return offspring1, offspring2
    
    @staticmethod
    def averaging_crossover(parent1: RealChromosome, parent2: RealChromosome) -> Tuple[RealChromosome, RealChromosome]:
        """
        Averaging crossover: both offspring are the average of parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two identical offspring chromosomes (averages)
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        genes = []
        
        for i in range(len(parent1)):
            # Average of parents
            avg_gene = (parent1[i] + parent2[i]) / 2.0
            genes.append(avg_gene)
        
        offspring1.set_genes(genes)
        offspring2.set_genes(genes)
        
        return offspring1, offspring2
