from .chromosome import Chromosome, Individual
from .config import GAConfig, SelectionMethod, CrossoverMethod, MutationMethod
from .population import Population, ElitismStrategy
from .selection import SelectionOperators
from .crossover import CrossoverOperators
from .mutation import MutationOperators, InversionOperator
from .genetic_algorithm import GeneticAlgorithm, GARunner, GenerationResult

__version__ = "1.0.0"
__author__ = "kkaczmarski"

__all__ = [
    # Core classes
    'Chromosome',
    'Individual', 
    'GAConfig',
    'Population',
    'GeneticAlgorithm',
    'GARunner',
    'GenerationResult',
    
    # Enums
    'SelectionMethod',
    'CrossoverMethod', 
    'MutationMethod',
    
    # Operators
    'SelectionOperators',
    'CrossoverOperators',
    'MutationOperators',
    'InversionOperator',
    'ElitismStrategy',
]