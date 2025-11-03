"""
Genetic Algorithm Framework - Package Initializer
Author: kkaczmarski
Date: October 25, 2025

This module makes the src directory a proper Python package and provides
convenient imports for all genetic algorithm components.
"""

# Core components
from .chromosome import Chromosome, Individual
from .config import GAConfig, SelectionMethod, CrossoverMethod, MutationMethod
from .population import Population, ElitismStrategy
from .selection import SelectionOperators
from .crossover import CrossoverOperators
from .mutation import MutationOperators, InversionOperator
from .genetic_algorithm import GeneticAlgorithm, GARunner, GenerationResult

__version__ = "1.0.0"
__author__ = "kkaczmarski"

# Define what gets imported with "from src import *"
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