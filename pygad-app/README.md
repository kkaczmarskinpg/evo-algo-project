# PyGAD Genetic Algorithm Configuration Guide

This guide explains all configuration parameters available for the genetic algorithm implementation.

## Basic Configuration

### Core Parameters

| Parameter | Type | Description | Default | Allowed Values |
|-----------|------|-------------|---------|----------------|
| `experiment_name` | String | Name for the experiment (used in file names and plots) | "experiment" | Any string |
| `representation` | String | Type of chromosome encoding | "real" | "binary", "real" |
| `function_name` | String | Optimization function to use | "ackley" | "michalewicz", "f16_2014" |
| `num_variables` | Integer | Number of variables/dimensions for the function | 10 | 2-30 recommended |

### Population Parameters

| Parameter | Type | Description | Default | Allowed Values |
|-----------|------|-------------|---------|----------------|
| `num_generations` | Integer | Number of generations to evolve | 100 | 50-1000 typical |
| `population_size` | Integer | Number of individuals in population | 80 | 20-200 typical |
| `num_parents_mating` | Integer | Number of parents selected for mating | 50 | ≤ population_size, 50-80% recommended |
| `keep_elitism` | Integer | Number of best individuals to keep unchanged | 1 | 0-10% of population |

### Selection Parameters

| Parameter | Type | Description | Default | Allowed Values |
|-----------|------|-------------|---------|----------------|
| `parent_selection_type` | String | Parent selection method | "tournament" | "tournament", "rws", "sus", "rank", "random" |
| `K_tournament` | Integer | Tournament size (only for tournament selection) | 3 | 2-5 recommended |

### Crossover Parameters

| Parameter | Type | Description | Default | Allowed Values |
|-----------|------|-------------|---------|----------------|
| `crossover_type` | String | Crossover method | "single_point" | See crossover types below |
| `crossover_probability` | Float | Probability of crossover | null (uses PyGAD default) | 0.0-1.0, typically 0.6-0.9 |

#### Available Crossover Types

- **`single_point`** - Single-point crossover
- **`two_points`** - Two-point crossover  
- **`uniform`** - Uniform crossover
- **`scattered`** - Scattered crossover
- **`arithmetic`** - Arithmetic crossover (real-valued, custom implementation)
- **`blend_alpha`** - BLX-α crossover (real-valued, custom implementation)
- **`linear`** - Linear crossover (real-valued, custom implementation)

### Mutation Parameters

| Parameter | Type | Description | Default | Allowed Values |
|-----------|------|-------------|---------|----------------|
| `mutation_type` | String | Mutation method | "random" | See mutation types below |
| `mutation_probability` | Float | Probability of mutation per individual | null (uses PyGAD default) | 0.0-1.0, typically 0.01-0.1 |
| `mutation_num_genes` | Integer | Number of genes to mutate per individual | 1 | 1-10% of chromosome length |

#### Available Mutation Types

- **`random`** - Random mutation within bounds
- **`swap`** - Swap mutation (exchanges two genes)
- **`inversion`** - Inversion mutation
- **`scramble`** - Scramble mutation
- **`gaussian`** - Gaussian mutation (custom implementation with σ=0.1)

### Advanced Mutation Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `random_mutation_min_val` | Float | Minimum value for random mutation | Varies by representation | Auto-set based on bounds |
| `random_mutation_max_val` | Float | Maximum value for random mutation | Varies by representation | Auto-set based on bounds |
| `gaussian_mutation_mean` | Float | Mean for Gaussian mutation | 0.0 | Only used with custom gaussian mutation |
| `gaussian_mutation_std` | Float | Standard deviation for Gaussian mutation | 0.1 | Only used with custom gaussian mutation |

## Binary Representation Parameters

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `bits_per_variable` | Integer | Bits per variable in binary encoding | 20 | 10-20 recommended, higher = better precision |

## Advanced Parameters

### Population Initialization

| Parameter | Type | Description | Default | Notes |
|-----------|------|-------------|---------|-------|
| `init_range_low` | Float | Lower bound for initial population | Auto-set | Determined by function bounds |
| `init_range_high` | Float | Upper bound for initial population | Auto-set | Determined by function bounds |
| `gene_type` | String/List | Data type of genes | Auto-set | "int" for binary, "float" for real |

## Experiment Control

| Parameter | Type | Description | Default | Allowed Values |
|-----------|------|-------------|---------|----------------|
| `num_runs` | Integer | Number of independent runs to perform | 1 | 1-30 typical for statistics |
| `output_directory` | String | Directory to save results | "results" | Any valid directory name |
| `save_plot` | Boolean | Whether to save fitness evolution plots | true | true, false |

## Function-Specific Information

### Michalewicz Function
- **Domain**: [0, π] for each variable
- **Optimum**: For 2D: ~-1.8013 at (2.20, 1.57)
- **Nature**: Multimodal, difficult

### F16 2014 (CEC2014)
- **Domain**: [-100, 100] for each variable  
- **Nature**: Composition function, very difficult

## Example Configurations

### Binary Representation Example
```json
{
  "experiment_name": "binary_test",
  "representation": "binary",
  "function_name": "michalewicz",
  "num_variables": 2,
  "bits_per_variable": 15,
  "num_generations": 300,
  "population_size": 50,
  "crossover_type": "uniform",
  "mutation_type": "random"
}
```

### Real Representation Example
```json
{
  "experiment_name": "real_test", 
  "representation": "real",
  "function_name": "f16_2014",
  "num_variables": 10,
  "num_generations": 500,
  "population_size": 100,
  "crossover_type": "blend_alpha",
  "mutation_type": "gaussian"
}
```

## Tips for Parameter Tuning

1. **Population Size**: Larger populations explore better but are slower
2. **Generations**: More generations = better convergence but longer runtime
3. **Mutation Rate**: Lower for fine-tuning, higher for exploration
4. **Crossover Rate**: Usually 0.7-0.9 works well
5. **Elitism**: Keep 1-5% of population to preserve good solutions
6. **Binary Precision**: More bits = higher precision but larger search space

## Output Files

The experiment generates:
- `experiment_log.txt` - Detailed execution log
- `results_summary.txt` - Final statistics
- `generation_data.csv` - Per-generation fitness data
- `fitness_evolution.png` - Fitness plots (if save_plot=true)