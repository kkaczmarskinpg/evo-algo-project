from typing import Dict, Any, List, Tuple
from enum import Enum


class SelectionMethod(Enum):
    """Selection methods available in the genetic algorithm."""
    BEST = "best"
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"


class CrossoverMethod(Enum):
    """Crossover methods available in the genetic algorithm."""
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    DISCRETE = "discrete"


class MutationMethod(Enum):
    """Mutation methods available in the genetic algorithm."""
    BOUNDARY = "boundary"
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"


class GAConfig:
    """
    Configuration class for Genetic Algorithm parameters.
    
    This class encapsulates all configurable parameters for the genetic algorithm
    including population size, number of epochs, selection methods, genetic operators,
    and their respective probabilities and parameters.
    """
    
    def __init__(self):
        """Initialize GA configuration with default values."""
        
        # Basic algorithm parameters
        self.population_size: int = 100
        self.num_epochs: int = 500
        self.chromosome_precision: int = 10
        
        # Problem definition
        self.bounds: List[Tuple[float, float]] = [(-5.0, 5.0)]  # Variable bounds
        
        # Selection configuration
        self.selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
        self.tournament_size: int = 3
        self.selection_pressure: float = 2.0  # For roulette wheel
        
        # Crossover configuration
        self.crossover_method: CrossoverMethod = CrossoverMethod.ONE_POINT
        self.crossover_probability: float = 0.8
        
        # Mutation configuration
        self.mutation_method: MutationMethod = MutationMethod.ONE_POINT
        self.mutation_probability: float = 0.1
        
        # Inversion operator configuration
        self.inversion_probability: float = 0.05
        
        # Elitism configuration
        self.elitism_enabled: bool = True
        self.elitism_count: int = 2  # Number of elite individuals
        self.elitism_percentage: float = 0.02  # Alternative: percentage of population
        self.use_elitism_percentage: bool = False  # Choose between count or percentage
        
    def set_population_size(self, size: int) -> None:
        """
        Set population size.
        
        Args:
            size: Population size (must be positive)
        """
        if size <= 0:
            raise ValueError("Population size must be positive")
        self.population_size = size
        
    def set_num_epochs(self, epochs: int) -> None:
        """
        Set number of epochs/generations.
        
        Args:
            epochs: Number of epochs (must be positive)
        """
        if epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        self.num_epochs = epochs
        
    def set_chromosome_precision(self, precision: int) -> None:
        """
        Set chromosome precision (bits per variable).
        
        Args:
            precision: Number of bits per variable (must be positive)
        """
        if precision <= 0:
            raise ValueError("Chromosome precision must be positive")
        self.chromosome_precision = precision
        
    def set_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Set variable bounds.
        
        Args:
            bounds: List of (min, max) tuples for each variable
        """
        for i, (min_val, max_val) in enumerate(bounds):
            if min_val >= max_val:
                raise ValueError(f"Invalid bounds for variable {i}: min >= max")
        self.bounds = bounds
        
    def set_selection_method(self, method: SelectionMethod, **kwargs) -> None:
        """
        Set selection method and its parameters.
        
        Args:
            method: Selection method
            **kwargs: Method-specific parameters
        """
        self.selection_method = method
        
        if method == SelectionMethod.TOURNAMENT:
            self.tournament_size = kwargs.get('tournament_size', self.tournament_size)
        elif method == SelectionMethod.ROULETTE_WHEEL:
            self.selection_pressure = kwargs.get('selection_pressure', self.selection_pressure)
            
    def set_crossover_config(self, method: CrossoverMethod, probability: float) -> None:
        """
        Set crossover method and probability.
        
        Args:
            method: Crossover method
            probability: Crossover probability [0, 1]
        """
        if not 0 <= probability <= 1:
            raise ValueError("Crossover probability must be between 0 and 1")
        
        self.crossover_method = method
        self.crossover_probability = probability
        
    def set_mutation_config(self, method: MutationMethod, probability: float) -> None:
        """
        Set mutation method and probability.
        
        Args:
            method: Mutation method
            probability: Mutation probability [0, 1]
        """
        if not 0 <= probability <= 1:
            raise ValueError("Mutation probability must be between 0 and 1")
        
        self.mutation_method = method
        self.mutation_probability = probability
        
    def set_inversion_probability(self, probability: float) -> None:
        """
        Set inversion operator probability.
        
        Args:
            probability: Inversion probability [0, 1]
        """
        if not 0 <= probability <= 1:
            raise ValueError("Inversion probability must be between 0 and 1")
        
        self.inversion_probability = probability
        
    def set_elitism_config(self, enabled: bool = True, count: int = None, 
                          percentage: float = None) -> None:
        """
        Configure elitism strategy.
        
        Args:
            enabled: Whether elitism is enabled
            count: Number of elite individuals (if using count-based elitism)
            percentage: Percentage of elite individuals (if using percentage-based elitism)
        """
        self.elitism_enabled = enabled
        
        if count is not None:
            if count < 0:
                raise ValueError("Elitism count must be non-negative")
            self.elitism_count = count
            self.use_elitism_percentage = False
            
        if percentage is not None:
            if not 0 <= percentage <= 1:
                raise ValueError("Elitism percentage must be between 0 and 1")
            self.elitism_percentage = percentage
            self.use_elitism_percentage = True
            
    def get_elitism_count(self) -> int:
        """
        Get the actual number of elite individuals based on configuration.
        
        Returns:
            Number of elite individuals
        """
        if not self.elitism_enabled:
            return 0
            
        if self.use_elitism_percentage:
            return max(1, int(self.population_size * self.elitism_percentage))
        else:
            return min(self.elitism_count, self.population_size)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'population_size': self.population_size,
            'num_epochs': self.num_epochs,
            'chromosome_precision': self.chromosome_precision,
            'bounds': self.bounds,
            'selection_method': self.selection_method.value,
            'tournament_size': self.tournament_size,
            'selection_pressure': self.selection_pressure,
            'crossover_method': self.crossover_method.value,
            'crossover_probability': self.crossover_probability,
            'mutation_method': self.mutation_method.value,
            'mutation_probability': self.mutation_probability,
            'inversion_probability': self.inversion_probability,
            'elitism_enabled': self.elitism_enabled,
            'elitism_count': self.elitism_count,
            'elitism_percentage': self.elitism_percentage,
            'use_elitism_percentage': self.use_elitism_percentage
        }
        
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == 'selection_method':
                    setattr(self, key, SelectionMethod(value))
                elif key == 'crossover_method':
                    setattr(self, key, CrossoverMethod(value))
                elif key == 'mutation_method':
                    setattr(self, key, MutationMethod(value))
                else:
                    setattr(self, key, value)
                    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "Genetic Algorithm Configuration:\n"
        config_str += f"  Population Size: {self.population_size}\n"
        config_str += f"  Number of Epochs: {self.num_epochs}\n"
        config_str += f"  Chromosome Precision: {self.chromosome_precision}\n"
        config_str += f"  Variable Bounds: {self.bounds}\n"
        config_str += f"  Selection Method: {self.selection_method.value}\n"
        config_str += f"  Tournament Size: {self.tournament_size}\n"
        config_str += f"  Crossover: {self.crossover_method.value} (p={self.crossover_probability})\n"
        config_str += f"  Mutation: {self.mutation_method.value} (p={self.mutation_probability})\n"
        config_str += f"  Inversion Probability: {self.inversion_probability}\n"
        config_str += f"  Elitism: {self.elitism_enabled}"
        if self.elitism_enabled:
            if self.use_elitism_percentage:
                config_str += f" ({self.elitism_percentage*100}%)"
            else:
                config_str += f" ({self.elitism_count} individuals)"
        return config_str
