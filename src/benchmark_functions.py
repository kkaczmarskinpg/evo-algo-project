"""
Benchmark functions for genetic algorithm testing.
Simple implementations of common optimization test functions.
"""
import math
import numpy as np


class Michalewicz:
    """Michalewicz function - multimodal optimization function."""
    
    def __init__(self, n_dimensions=2, m=10):
        """
        Initialize Michalewicz function.
        
        Args:
            n_dimensions: Number of dimensions
            m: Parameter controlling steepness (default: 10)
        """
        self.n_dimensions = n_dimensions
        self.m = m
    
    def __call__(self, x):
        """
        Evaluate Michalewicz function.
        
        Args:
            x: Input vector
            
        Returns:
            Function value (to be minimized, so we negate)
        """
        if isinstance(x, list):
            x = np.array(x)
        
        if len(x) != self.n_dimensions:
            # Pad or truncate to match expected dimensions
            if len(x) < self.n_dimensions:
                x = np.pad(x, (0, self.n_dimensions - len(x)), mode='constant', constant_values=0)
            else:
                x = x[:self.n_dimensions]
        
        result = 0
        for i in range(self.n_dimensions):
            result += math.sin(x[i]) * math.sin((i + 1) * x[i]**2 / math.pi)**(2 * self.m)
        
        return -result  # Negate for minimization


class Ackley:
    """Ackley function - multimodal optimization function."""
    
    def __init__(self, n_dimensions=2):
        """
        Initialize Ackley function.
        
        Args:
            n_dimensions: Number of dimensions
        """
        self.n_dimensions = n_dimensions
    
    def __call__(self, x):
        """
        Evaluate Ackley function.
        
        Args:
            x: Input vector
            
        Returns:
            Function value
        """
        if isinstance(x, list):
            x = np.array(x)
        
        if len(x) != self.n_dimensions:
            # Pad or truncate to match expected dimensions
            if len(x) < self.n_dimensions:
                x = np.pad(x, (0, self.n_dimensions - len(x)), mode='constant', constant_values=0)
            else:
                x = x[:self.n_dimensions]
        
        a = 20
        b = 0.2
        c = 2 * math.pi
        
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        
        result = -a * math.exp(-b * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + a + math.e
        return result


# Simple implementations for F16-2014 and F5-2014 (basic versions)
class F162014:
    """Simplified F16 CEC 2014 function."""
    
    def __init__(self, ndim=2):
        self.ndim = ndim
    
    def evaluate(self, x):
        """Simple sphere function as placeholder for F16-2014."""
        if isinstance(x, list):
            x = np.array(x)
        
        if len(x) != self.ndim:
            if len(x) < self.ndim:
                x = np.pad(x, (0, self.ndim - len(x)), mode='constant', constant_values=0)
            else:
                x = x[:self.ndim]
        
        return np.sum(x**2)


class F52014:
    """Simplified F5 CEC 2014 function."""
    
    def __init__(self, ndim=2):
        self.ndim = ndim
    
    def evaluate(self, x):
        """Simple Rosenbrock function as placeholder for F5-2014."""
        if isinstance(x, list):
            x = np.array(x)
        
        if len(x) != self.ndim:
            if len(x) < self.ndim:
                x = np.pad(x, (0, self.ndim - len(x)), mode='constant', constant_values=0)
            else:
                x = x[:self.ndim]
        
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        return result
