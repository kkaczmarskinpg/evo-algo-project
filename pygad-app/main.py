import logging
import pygad
import numpy as np
import json
import sys
import os
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt


# ========================= FUNKCJE DEKODOWANIA =========================

def decode_binary_individual(individual, num_vars, bits_per_var, bounds):
    """
    Dekoduje binarnego osobnika na wektor wartości rzeczywistych.
    
    Args:
        individual: Osobnik binarny (lista/array z 0 i 1)
        num_vars: Liczba zmiennych
        bits_per_var: Liczba bitów na zmienną
        bounds: Lista krotek [(min1, max1), (min2, max2), ...]
        
    Returns:
        Lista wartości rzeczywistych
    """
    decoded = []
    
    for i in range(num_vars):
        # Wyciągnij bity dla i-tej zmiennej
        start_idx = i * bits_per_var
        end_idx = start_idx + bits_per_var
        bits = individual[start_idx:end_idx]
        
        # Konwertuj bity na liczbę całkowitą
        int_value = 0
        for j, bit in enumerate(bits):
            int_value += int(bit) * (2 ** (bits_per_var - 1 - j))
        
        # Normalizuj do przedziału [0, 1]
        max_int = 2 ** bits_per_var - 1
        normalized = int_value / max_int
        
        # Skaluj do docelowego przedziału
        min_bound, max_bound = bounds[i]
        real_value = min_bound + normalized * (max_bound - min_bound)
        
        decoded.append(real_value)
    
    return np.array(decoded)


# ========================= FUNKCJE TESTOWE =========================

def create_fitness_function(function_name, representation, config):
    """
    Tworzy funkcję fitness na podstawie konfiguracji.
    
    Args:
        function_name: Nazwa funkcji testowej
        representation: 'binary' lub 'real'
        config: Słownik konfiguracyjny
        
    Returns:
        Funkcja fitness dla PyGAD
    """
    num_vars = config['num_variables']
    
    # Importuj odpowiednią funkcję z gotowych bibliotek
    if function_name == 'michalewicz':
        import benchmark_functions as bf
        func_obj = bf.Michalewicz(n_dimensions=num_vars)
        bounds = [(0, np.pi)] * num_vars
        
        def evaluate_function(x):
            return func_obj(x)
            
    elif function_name == 'f16_2014':
        from opfunu.cec_based import cec2014
        func_obj = cec2014.F162014(ndim=num_vars)
        bounds = [(-100, 100)] * num_vars
        
        def evaluate_function(x):
            # Próbuj różne metody wywołania funkcji opfunu
            if hasattr(func_obj, 'evaluate'):
                return func_obj.evaluate(x)
            elif hasattr(func_obj, '__call__'):
                return func_obj(x)
            else:
                # Fallback - użyj prostej funkcji
                return np.sum(np.array(x)**2)
                
    else:
        raise ValueError(f"Unknown function: {function_name}. Available functions: 'michalewicz', 'f16_2014'")
    
    # Dla reprezentacji binarnej
    if representation == 'binary':
        bits_per_var = config.get('bits_per_variable', 20)
        
        def fitness_func_binary(ga_instance, solution, solution_idx):
            # Dekoduj binarnego osobnika
            decoded = decode_binary_individual(solution, num_vars, bits_per_var, bounds)
            # Oblicz wartość funkcji
            result = evaluate_function(decoded)
            # Dla funkcji minimalizacyjnych używamy stałej transformacji:
            # fitness = -result (im mniejszy result, tym większy fitness)
            return -result
        
        return fitness_func_binary, bounds
    
    # Dla reprezentacji rzeczywistej
    else:
        def fitness_func_real(ga_instance, solution, solution_idx):
            result = evaluate_function(solution)
            # Dla funkcji minimalizacyjnych używamy stałej transformacji:
            # fitness = -result (im mniejszy result, tym większy fitness)
            return -result
        
        return fitness_func_real, bounds


# ========================= KRZYŻOWANIE DLA REPREZENTACJI RZECZYWISTEJ =========================

def arithmetic_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie arytmetyczne: offspring = alpha * parent1 + (1-alpha) * parent2
    """
    offspring = []
    idx = 0
    alpha = 0.5
    
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        
        # Losuj alpha dla każdego genu
        if np.random.random() < ga_instance.crossover_probability:
            child = alpha * parent1 + (1 - alpha) * parent2
        else:
            child = parent1
        
        offspring.append(child)
        idx += 1
    
    return np.array(offspring)


def blend_alpha_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie BLX-α: offspring w rozszerzonym przedziale rodziców
    """
    offspring = []
    idx = 0
    alpha = 0.5
    
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        
        child = parent1.copy()
        
        if np.random.random() < ga_instance.crossover_probability:
            for i in range(len(parent1)):
                min_val = min(parent1[i], parent2[i])
                max_val = max(parent1[i], parent2[i])
                range_val = max_val - min_val
                
                extended_min = min_val - alpha * range_val
                extended_max = max_val + alpha * range_val
                
                child[i] = np.random.uniform(extended_min, extended_max)
        
        offspring.append(child)
        idx += 1
    
    return np.array(offspring)


def linear_crossover(parents, offspring_size, ga_instance):
    """
    Krzyżowanie liniowe: tworzy trzy potomstwa i wybiera dwa najlepsze
    """
    offspring = []
    idx = 0
    
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        
        if np.random.random() < ga_instance.crossover_probability:
            # Trzy warianty
            child1 = 0.5 * parent1 + 0.5 * parent2
            child2 = 1.5 * parent1 - 0.5 * parent2
            
            # Zwróć jeden z nich (lub można wybrać najlepszy)
            if len(offspring) < offspring_size[0]:
                offspring.append(child1)
                idx += 1
            if len(offspring) < offspring_size[0]:
                offspring.append(child2)
                idx += 1
        else:
            offspring.append(parent1)
            idx += 1
    
    return np.array(offspring[:offspring_size[0]])


# ========================= MUTACJA =========================

def gaussian_mutation(offspring, ga_instance):
    """
    Mutacja Gaussa: dodaje szum z rozkładu normalnego
    """
    mutation_probability = 1.0 / offspring.shape[1]  # Prawdopodobieństwo mutacji na gen
    sigma = 0.1  # Odchylenie standardowe
    
    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < mutation_probability:
                # Dodaj szum gaussowski
                offspring[chromosome_idx, gene_idx] += np.random.normal(0, sigma)
    
    return offspring


# ========================= LOGOWANIE =========================

def setup_logger(log_file='experiment.log'):
    """Konfiguruje logger do pliku i konsoli."""
    logger = logging.getLogger('pygad_experiment')
    logger.setLevel(logging.DEBUG)
    
    # Usuń istniejące handlery
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def on_generation(ga_instance):
    """Callback wywoływany po każdej generacji."""
    generation = ga_instance.generations_completed
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness
    )
    
    # Konwertuj fitness z powrotem do rzeczywistych wartości funkcji (minimalizacja)
    # Ponieważ fitness = -result, to result = -fitness
    def fitness_to_function_value(fitness_val):
        return -fitness_val
    
    tmp = [fitness_to_function_value(x) for x in ga_instance.last_generation_fitness]
    best_value = fitness_to_function_value(solution_fitness)
    mean_fitness = np.average(tmp)
    std_fitness = np.std(tmp)
    
    # Zapisz statystyki do historii (używamy rzeczywistych wartości funkcji)
    if not hasattr(ga_instance, 'stats_history'):
        ga_instance.stats_history = []
    
    ga_instance.stats_history.append({
        'generation': generation,
        'best_fitness': best_value,  # Rzeczywista wartość f(x) - im mniejsza tym lepiej
        'best_individual': solution.tolist(),
        'mean_fitness': mean_fitness,
        'std_dev': std_fitness
    })
    
    if generation % 10 == 0:  # Co 10 generacji
        ga_instance.logger.info(f"Generation {generation}")
        ga_instance.logger.info(f"  Best: {best_value:.6f}")
        ga_instance.logger.info(f"  Min: {np.min(tmp):.6f}")
        ga_instance.logger.info(f"  Max: {np.max(tmp):.6f}")
        ga_instance.logger.info(f"  Avg: {mean_fitness:.6f}")
        ga_instance.logger.info(f"  Std: {std_fitness:.6f}")


# ========================= GŁÓWNA FUNKCJA =========================

def run_experiment(config_file='experiment_configuration.json', run_index=1):
    """
    Uruchamia eksperyment na podstawie pliku konfiguracyjnego.
    
    Args:
        config_file: Ścieżka do pliku JSON z konfiguracją
        run_index: Numer uruchomienia (dla wielokrotnych eksperymentów)
    """
    # Wczytaj konfigurację
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.get('experiment_name', 'Unnamed')}")
    print(f"{'='*70}\n")
    
    # Pobierz parametry
    representation = config.get('representation', 'real')
    function_name = config.get('function_name', 'ackley')
    num_vars = config.get('num_variables', 10)
    num_generations = config.get('num_generations', 100)
    sol_per_pop = config.get('population_size', 80)
    num_parents_mating = config.get('num_parents_mating', 50)
    parent_selection_type = config.get('parent_selection_type', 'tournament')
    crossover_type = config.get('crossover_type', 'single_point')
    mutation_type = config.get('mutation_type', 'random')
    mutation_probability = config.get('mutation_probability', None)
    crossover_probability = config.get('crossover_probability', None)
    
    # Utwórz folder wynikowy
    output_dir = config.get('output_directory', 'results')
    exp_name = config.get('experiment_name', 'experiment').replace(' ', '_').replace('-', '_')
    run_dir = os.path.join(output_dir, exp_name, f'run_{run_index}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(run_dir, f'run_{run_index}.log')
    logger = setup_logger(log_file)
    
    # Utwórz funkcję fitness
    fitness_func, bounds = create_fitness_function(function_name, representation, config)
    
    # Parametry specyficzne dla reprezentacji
    if representation == 'binary':
        bits_per_var = config.get('bits_per_variable', 20)
        num_genes = num_vars * bits_per_var
        gene_type = int
        init_range_low = 0
        init_range_high = 2
        
        logger.info(f"Representation: Binary")
        logger.info(f"Bits per variable: {bits_per_var}")
        logger.info(f"Total genes: {num_genes}")
    else:
        num_genes = num_vars
        gene_type = float
        init_range_low = bounds[0][0]
        init_range_high = bounds[0][1]
        
        logger.info(f"Representation: Real")
        logger.info(f"Number of genes: {num_genes}")
        logger.info(f"Bounds: [{init_range_low}, {init_range_high}]")
    
    logger.info(f"Function: {function_name}")
    logger.info(f"Population size: {sol_per_pop}")
    logger.info(f"Generations: {num_generations}")
    logger.info(f"Selection: {parent_selection_type}")
    logger.info(f"Crossover: {crossover_type}")
    logger.info(f"Mutation: {mutation_type}")
    logger.info("")
    
    # Przygotuj parametry dla PyGAD
    ga_params = {
        'num_generations': num_generations,
        'sol_per_pop': sol_per_pop,
        'num_parents_mating': num_parents_mating,
        'num_genes': num_genes,
        'fitness_func': fitness_func,
        'init_range_low': init_range_low,
        'init_range_high': init_range_high,
        'gene_type': gene_type,
        'parent_selection_type': parent_selection_type,
        'keep_elitism': config.get('keep_elitism', 1),
        'K_tournament': config.get('K_tournament', 3),
        'logger': logger,
        'on_generation': on_generation,
    }
    
    # Dla reprezentacji rzeczywistej użyj własnych operatorów krzyżowania
    if representation == 'real' and crossover_type in ['arithmetic', 'blend_alpha', 'linear']:
        if crossover_type == 'arithmetic':
            ga_params['crossover_type'] = arithmetic_crossover
        elif crossover_type == 'blend_alpha':
            ga_params['crossover_type'] = blend_alpha_crossover
        elif crossover_type == 'linear':
            ga_params['crossover_type'] = linear_crossover
    else:
        ga_params['crossover_type'] = crossover_type
    
    # Mutacja
    if mutation_type == 'gaussian':
        ga_params['mutation_type'] = gaussian_mutation
    else:
        ga_params['mutation_type'] = mutation_type
        if mutation_type == 'random':
            ga_params['random_mutation_min_val'] = init_range_low
            ga_params['random_mutation_max_val'] = init_range_high
        ga_params['mutation_num_genes'] = config.get('mutation_num_genes', 1)
    
    # Prawdopodobieństwa
    if crossover_probability is not None:
        ga_params['crossover_probability'] = crossover_probability
    if mutation_probability is not None:
        ga_params['mutation_probability'] = mutation_probability
    
    # Utwórz instancję GA
    print("Starting genetic algorithm...\n")
    ga_instance = pygad.GA(**ga_params)
    
    # Inicjalizuj historię statystyk
    ga_instance.stats_history = []
    
    # Uruchom algorytm
    start_time = time.time()
    ga_instance.run()
    execution_time = time.time() - start_time
    
    # Wyniki
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_value = 1.0 / (solution_fitness + 1e-10)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    if representation == 'binary':
        decoded_solution = decode_binary_individual(
            solution, num_vars, bits_per_var, bounds
        )
        print(f"Best solution (decoded): {decoded_solution}")
    else:
        print(f"Best solution: {solution}")
    
    print(f"Best fitness value: {best_value:.10f}")
    print(f"{'='*70}\n")
    
    logger.info(f"\nFinal best fitness: {best_value:.10f}")
    logger.info(f"Execution time: {execution_time:.2f}s")
    
    # Zapisz dane do CSV
    csv_file = os.path.join(run_dir, f'run_{run_index}.csv')
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best_fitness', 'best_individual', 'mean_fitness', 'std_dev'])
            writer.writeheader()
            for stat in ga_instance.stats_history:
                writer.writerow(stat)
        print(f"CSV data saved to: {csv_file}")
    except Exception as e:
        print(f"Could not save CSV: {e}")
    
    # Zapisz wyniki do pliku tekstowego
    results_file = os.path.join(run_dir, f'run_{run_index}_results.txt')
    try:
        with open(results_file, 'w') as f:
            f.write("=== GENETIC ALGORITHM RESULTS ===\n")
            f.write(f"Run Index: {run_index}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Representation: {representation}\n")
            f.write(f"Function: {function_name}\n")
            f.write(f"Population Size: {sol_per_pop}\n")
            f.write(f"Generations: {num_generations}\n")
            f.write(f"Selection: {parent_selection_type}\n")
            f.write(f"Crossover: {crossover_type}\n")
            f.write(f"Mutation: {mutation_type}\n")
            f.write(f"Execution Time: {execution_time:.2f}s\n")
            f.write("="*40 + "\n\n")
            f.write("=== FINAL RESULTS ===\n")
            f.write(f"Best Fitness: {best_value:.6f}\n")
            if representation == 'binary':
                decoded_sol = decode_binary_individual(solution, num_vars, bits_per_var, bounds)
                f.write(f"Best Solution (decoded): {decoded_sol.tolist()}\n")
                f.write(f"Best Solution (binary): {solution.tolist()}\n")
            else:
                f.write(f"Best Solution: {solution.tolist()}\n")
            f.write(f"Total Time: {execution_time:.2f}s\n")
        print(f"Results saved to: {results_file}")
    except Exception as e:
        print(f"Could not save results file: {e}")
    
    # Generuj i zapisz wykres
    if config.get('save_plot', True):
        try:
            plot_file = os.path.join(run_dir, f'run_{run_index}.png')
            
            # Przygotuj dane
            generations = [s['generation'] for s in ga_instance.stats_history]
            best_fitness = [s['best_fitness'] for s in ga_instance.stats_history]
            mean_fitness = [s['mean_fitness'] for s in ga_instance.stats_history]
            std_dev = [s['std_dev'] for s in ga_instance.stats_history]
            
            # Utwórz wykres z dwoma subplot'ami
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Górny wykres - Best Fitness (minimalizacja)
            ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
            ax1.set_xlabel('Generation', fontsize=12)
            ax1.set_ylabel('Fitness Value (lower is better)', fontsize=12)
            ax1.set_title(f'Best Fitness vs Generation - {config.get("experiment_name", "Run 1")}', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Dolny wykres - Average Fitness i Standard Deviation
            mean_fitness_array = np.array(mean_fitness)
            std_dev_array = np.array(std_dev)
            
            ax2.plot(generations, mean_fitness_array, 'g-', linewidth=2, label='Average Fitness')
            ax2.fill_between(generations, 
                            mean_fitness_array - std_dev_array, 
                            mean_fitness_array + std_dev_array, 
                            alpha=0.3, color='green', label='+/-1 Standard Deviation')
            ax2.set_xlabel('Generation', fontsize=12)
            ax2.set_ylabel('Fitness Value (lower is better)', fontsize=12)
            ax2.set_title(f'Average Fitness and Standard Deviation vs Generation - {config.get("experiment_name", "Run 1")}', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved to: {plot_file}")
        except Exception as e:
            print(f"Could not save plot: {e}")
            import traceback
            traceback.print_exc()
    
    return ga_instance, best_value


def run_multiple_experiments(config_file='experiment_configuration.json'):
    """
    Uruchamia wiele eksperymentów zgodnie z konfiguracją.
    
    Args:
        config_file: Ścieżka do pliku JSON z konfiguracją
    """
    # Wczytaj konfigurację
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    num_runs = config.get('num_runs', 1)
    
    print(f"\n{'='*80}")
    print(f"RUNNING {num_runs} EXPERIMENT(S): {config.get('experiment_name', 'Unnamed')}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx}/{num_runs}")
        print(f"{'='*80}")
        
        try:
            ga_instance, best_value = run_experiment(config_file, run_index=run_idx)
            all_results.append({
                'run': run_idx,
                'best_fitness': best_value,
                'success': True
            })
        except Exception as e:
            print(f"\nError in run {run_idx}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'run': run_idx,
                'error': str(e),
                'success': False
            })
    
    # Podsumowanie wszystkich uruchomień
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL RUNS")
    print(f"{'='*80}\n")
    
    successful_runs = [r for r in all_results if r['success']]
    failed_runs = [r for r in all_results if not r['success']]
    
    print(f"Total runs: {num_runs}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")
    
    if successful_runs:
        best_fitnesses = [r['best_fitness'] for r in successful_runs]
        print(f"\nBest fitness values:")
        print(f"  Min: {np.min(best_fitnesses):.10f}")
        print(f"  Max: {np.max(best_fitnesses):.10f}")
        print(f"  Mean: {np.mean(best_fitnesses):.10f}")
        print(f"  Std: {np.std(best_fitnesses):.10f}")
    
    print(f"\n{'='*80}\n")
    
    return all_results


# ========================= MAIN =========================

if __name__ == "__main__":
    config_file = 'experiment_configuration.json'
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found!")
        print(f"Usage: python main.py [config_file.json]")
        sys.exit(1)
    
    try:
        run_multiple_experiments(config_file)
    except Exception as e:
        print(f"\nError during experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
