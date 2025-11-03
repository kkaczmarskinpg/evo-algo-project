"""
Minimal GUI Application for Genetic Algorithm Framework
Author: kkaczmarski
Date: October 25, 2025

A simple tkinter-based GUI for configuring and running genetic algorithms
with real-time plotting and execution time monitoring.
"""
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
import json
from typing import Callable, Optional, Dict, Any
import sys
import os
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import GA components
from genetic_algorithm import GeneticAlgorithm, GenerationResult
from config import GAConfig, SelectionMethod, CrossoverMethod, MutationMethod
# Import test functions
from benchmark_functions import Michalewicz, Ackley
from opfunu.cec_based.cec2014 import F162014, F52014

class ObjectiveFunctions:
    """Collection of test objective functions."""
    
    @staticmethod
    def FuncMichalewicz(x, dim=None):
        try:
            if dim is None:
                dim = len(x)
            return Michalewicz(n_dimensions=dim)(x)
        except Exception as e:
            raise ValueError(f"Michalewicz function error: {str(e)}. Check dimension compatibility.")
    
    @staticmethod
    def FuncAckley(x, dim=None):
        try:
            if dim is None:
                dim = len(x)
            return Ackley(n_dimensions=dim)(x)
        except Exception as e:
            raise ValueError(f"Ackley library function error: {str(e)}. Check dimension compatibility.")
    
    @staticmethod
    def FuncF162014(x, dim=None):
        try:
            if dim is None:
                dim = len(x)
            
            # F16-2014 only supports specific dimensions: [10, 20, 30, 50, 100]
            supported_dims = [10, 20, 30, 50, 100]
            if dim not in supported_dims:
                # Use closest supported dimension
                closest_dim = min(supported_dims, key=lambda d: abs(d - dim))
                # Pad or truncate input to match supported dimension
                if len(x) < closest_dim:
                    x_adjusted = list(x) + [0.0] * (closest_dim - len(x))
                else:
                    x_adjusted = x[:closest_dim]
                return F162014(ndim=closest_dim).evaluate(x_adjusted)
            else:
                return F162014(ndim=dim).evaluate(x)
        except Exception as e:
            raise ValueError(f"F16-2014 function error: {str(e)}. Supported dimensions: [10, 20, 30, 50, 100].")
    
    @staticmethod
    def FuncF52014(x, dim=None):
        try:
            if dim is None:
                dim = len(x)
            
            # F5-2014 only supports specific dimensions: [10, 20, 30, 50, 100]
            supported_dims = [10, 20, 30, 50, 100]
            if dim not in supported_dims:
                # Use closest supported dimension
                closest_dim = min(supported_dims, key=lambda d: abs(d - dim))
                # Pad or truncate input to match supported dimension
                if len(x) < closest_dim:
                    x_adjusted = list(x) + [0.0] * (closest_dim - len(x))
                else:
                    x_adjusted = x[:closest_dim]
                return F52014(ndim=closest_dim).evaluate(x_adjusted)
            else:
                return F52014(ndim=dim).evaluate(x)
        except Exception as e:
            raise ValueError(f"F5-2014 function error: {str(e)}. Supported dimensions: [10, 20, 30, 50, 100].")


class GAConfigPanel:
    """Panel for configuring GA parameters."""
    
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="GA Configuration", padding="5")
        
        # Configuration variables
        self.config_vars = {
            'population_size': tk.IntVar(value=100),
            'num_epochs': tk.IntVar(value=200),
            'chromosome_precision': tk.IntVar(value=12),
            'selection_method': tk.StringVar(value="tournament"),
            'tournament_size': tk.IntVar(value=3),
            'crossover_method': tk.StringVar(value="one_point"),
            'crossover_probability': tk.DoubleVar(value=0.8),
            'mutation_method': tk.StringVar(value="one_point"),
            'mutation_probability': tk.DoubleVar(value=0.1),
            'inversion_probability': tk.DoubleVar(value=0.05),
            'elitism_enabled': tk.BooleanVar(value=True),
            'elitism_count': tk.IntVar(value=2),
            'minimize': tk.BooleanVar(value=True),
            'function_name': tk.StringVar(value=""),
            'num_variables': tk.IntVar(value=2),
            'bound_min': tk.DoubleVar(value=-5.0),
            'bound_max': tk.DoubleVar(value=5.0)
        }
        
        self._create_widgets()
    
    def _on_function_change(self, event=None):
        """Auto-adjust variables when function selection changes."""
        function_name = self.config_vars['function_name'].get()
        
        # Set appropriate default number of variables and bounds
        if function_name in ['michalewicz', 'ackley']:
            # Default to 2D but allow user to change
            if self.config_vars['num_variables'].get() < 2:
                self.config_vars['num_variables'].set(2)
            # Set appropriate bounds for these functions
            self.config_vars['bound_min'].set(0.0)
            self.config_vars['bound_max'].set(3.14159)  # pi for Michalewicz
        elif function_name in ['f16_2014', 'f5_2014']:
            # CEC functions support specific dimensions: [10, 20, 30, 50, 100]
            # Default to 30D (most common test dimension)
            current_vars = self.config_vars['num_variables'].get()
            supported_dims = [10, 20, 30, 50, 100]
            
            if current_vars not in supported_dims:
                # Set to closest supported dimension, defaulting to 30
                closest_dim = min(supported_dims, key=lambda d: abs(d - current_vars)) if current_vars > 0 else 30
                self.config_vars['num_variables'].set(closest_dim)
            
            # Set appropriate bounds for CEC functions
            self.config_vars['bound_min'].set(-100.0)
            self.config_vars['bound_max'].set(100.0)
    
    def _create_widgets(self):
        row = 0
        
        # Basic parameters
        ttk.Label(self.frame, text="Basic Parameters").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Label(self.frame, text="Population Size:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['population_size'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Epochs:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['num_epochs'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Precision:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['chromosome_precision'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Problem definition
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(self.frame, text="Problem Definition").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Label(self.frame, text="Function:").grid(row=row, column=0, sticky="w")
        function_combo = ttk.Combobox(self.frame, textvariable=self.config_vars['function_name'], 
                                     values=["michalewicz", "ackley", "f16_2014", "f5_2014"], 
                                     state="readonly", width=15)
        function_combo.grid(row=row, column=1, sticky="w")
        
        # Add callback to auto-adjust variables when function changes
        function_combo.bind('<<ComboboxSelected>>', self._on_function_change)
        row += 1
        
        ttk.Label(self.frame, text="Variables:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['num_variables'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Bounds Min:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['bound_min'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Bounds Max:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['bound_max'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Checkbutton(self.frame, text="Minimize", variable=self.config_vars['minimize']).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        
        # Selection
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(self.frame, text="Selection").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Label(self.frame, text="Method:").grid(row=row, column=0, sticky="w")
        selection_combo = ttk.Combobox(self.frame, textvariable=self.config_vars['selection_method'],
                                      values=["best", "tournament", "roulette_wheel"], 
                                      state="readonly", width=12)
        selection_combo.grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Tournament Size:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['tournament_size'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Crossover
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(self.frame, text="Crossover").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Label(self.frame, text="Method:").grid(row=row, column=0, sticky="w")
        crossover_combo = ttk.Combobox(self.frame, textvariable=self.config_vars['crossover_method'],
                                      values=["one_point", "two_point", "uniform", "discrete"], 
                                      state="readonly", width=12)
        crossover_combo.grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Probability:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['crossover_probability'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Mutation
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(self.frame, text="Mutation").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Label(self.frame, text="Method:").grid(row=row, column=0, sticky="w")
        mutation_combo = ttk.Combobox(self.frame, textvariable=self.config_vars['mutation_method'],
                                     values=["boundary", "one_point", "two_point"], 
                                     state="readonly", width=12)
        mutation_combo.grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Probability:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['mutation_probability'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Inversion Prob:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['inversion_probability'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
        
        # Elitism
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        
        ttk.Label(self.frame, text="Elitism").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        
        ttk.Checkbutton(self.frame, text="Enabled", variable=self.config_vars['elitism_enabled']).grid(row=row, column=0, sticky="w")
        row += 1
        
        ttk.Label(self.frame, text="Count:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.frame, textvariable=self.config_vars['elitism_count'], width=10).grid(row=row, column=1, sticky="w")
        row += 1
    
    def get_config(self) -> GAConfig:
        """Get GAConfig object from current settings."""
        config = GAConfig()
        
        # Set basic parameters
        config.set_population_size(self.config_vars['population_size'].get())
        config.set_num_epochs(self.config_vars['num_epochs'].get())
        config.set_chromosome_precision(self.config_vars['chromosome_precision'].get())
        
        # Set bounds
        num_vars = self.config_vars['num_variables'].get()
        min_bound = self.config_vars['bound_min'].get()
        max_bound = self.config_vars['bound_max'].get()
        bounds = [(min_bound, max_bound) for _ in range(num_vars)]
        config.set_bounds(bounds)
        
        # Set selection method
        selection_method = SelectionMethod(self.config_vars['selection_method'].get())
        config.set_selection_method(selection_method, tournament_size=self.config_vars['tournament_size'].get())
        
        # Set crossover
        crossover_method = CrossoverMethod(self.config_vars['crossover_method'].get())
        config.set_crossover_config(crossover_method, self.config_vars['crossover_probability'].get())
        
        # Set mutation
        mutation_method = MutationMethod(self.config_vars['mutation_method'].get())
        config.set_mutation_config(mutation_method, self.config_vars['mutation_probability'].get())
        
        # Set inversion
        config.set_inversion_probability(self.config_vars['inversion_probability'].get())
        
        # Set elitism
        config.set_elitism_config(
            enabled=self.config_vars['elitism_enabled'].get(),
            count=self.config_vars['elitism_count'].get()
        )
        
        return config
    
    def get_objective_function(self) -> Callable:
        """Get objective function from selection."""
        function_name = self.config_vars['function_name'].get()
        num_variables = self.config_vars['num_variables'].get()
        
        # Create a wrapper function that passes the dimension parameter
        if function_name == 'michalewicz':
            return lambda x: ObjectiveFunctions.FuncMichalewicz(x, dim=num_variables)
        elif function_name == 'ackley':
            return lambda x: ObjectiveFunctions.FuncAckley(x, dim=num_variables)
        elif function_name == 'f16_2014':
            return lambda x: ObjectiveFunctions.FuncF162014(x, dim=num_variables)
        elif function_name == 'f5_2014':
            return lambda x: ObjectiveFunctions.FuncF52014(x, dim=num_variables)
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    def is_minimize(self) -> bool:
        """Get optimization direction."""
        return self.config_vars['minimize'].get()


class PlotPanel:
    """Panel for displaying convergence plots."""
    
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="Convergence Plots", padding="5")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        
        self.clear_plots()
        
    def clear_plots(self):
        """Clear all plots."""
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.set_title("Best Fitness vs Generation")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness Value")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("Average Fitness and Standard Deviation vs Generation")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Fitness Value")
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_plots(self, convergence_data: Dict):
        """Update plots with new data."""
        generations = convergence_data['generations']
        best_fitness = convergence_data['best_fitness']
        avg_fitness = convergence_data['average_fitness']
        std_fitness = convergence_data['std_fitness']
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Best fitness over generations
        self.ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        self.ax1.set_title("Best Fitness vs Generation")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness Value")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot 2: Average fitness and standard deviation
        avg_fitness_array = np.array(avg_fitness)
        std_fitness_array = np.array(std_fitness)
        
        self.ax2.plot(generations, avg_fitness, 'g-', linewidth=2, label='Average Fitness')
        self.ax2.fill_between(generations, 
                             avg_fitness_array - std_fitness_array,
                             avg_fitness_array + std_fitness_array,
                             alpha=0.3, color='green', label='±1 Standard Deviation')
        self.ax2.set_title("Average Fitness and Standard Deviation vs Generation")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Fitness Value")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        self.fig.tight_layout()
        self.canvas.draw()


class GAApp:
    """Main GUI application for Genetic Algorithm."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Algorithm")
        self.root.geometry("1400x800")

        # List for Saved results
        self.saved_results = []
        
        # GA instance
        self.ga = None
        self.is_running = False
        self.results = None
        
        # Execution time tracking
        self.start_time = None
        self.execution_time_var = tk.StringVar(value="Execution Time: 0.00s")
        
        # Progress tracking
        self.progress_var = tk.StringVar(value="Ready")
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        
        # Main container
        main_container = ttk.PanedWindow(self.root, orient="horizontal")
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left panel (configuration)
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=1)
        
        # Configuration panel
        self.config_panel = GAConfigPanel(left_frame)
        self.config_panel.frame.pack(fill="both", expand=True, padx=(0, 5))
        
        # Right panel (plots and controls)
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=3)
        
        # Control panel
        control_frame = ttk.LabelFrame(right_frame, text="Controls", padding="5")
        control_frame.pack(fill="x", pady=(0, 5))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x")
        
        self.run_button = ttk.Button(button_frame, text="Run GA", command=self._run_ga)
        self.run_button.pack(side="left", padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop_ga, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 5))
        
        self.clear_button = ttk.Button(button_frame, text="Clear", command=self._clear_results)
        self.clear_button.pack(side="left", padx=(0, 5))

        self.save_csv_button = ttk.Button(button_frame, text="Save results to csv", command=self._save_results_to_csv)
        self.save_csv_button.pack(side="left", padx=5)
        
        # Save/Load buttons
        ttk.Button(button_frame, text="Save Config", command=self._save_config).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Load Config", command=self._load_config).pack(side="left", padx=(0, 5))
        
        # Status panel
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill="x", pady=(5, 0))
        
        ttk.Label(status_frame, textvariable=self.progress_var).pack(side="left")
        ttk.Label(status_frame, textvariable=self.execution_time_var).pack(side="right")
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(control_frame, mode='determinate')
        self.progress_bar.pack(fill="x", pady=(5, 0))
        
        # Results panel
        results_frame = ttk.LabelFrame(right_frame, text="Results", padding="5")
        results_frame.pack(fill="x", pady=(0, 5))
        
        self.results_text = tk.Text(results_frame, height=4, font=("Courier", 9))
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Plot panel
        self.plot_panel = PlotPanel(right_frame)
        self.plot_panel.frame.pack(fill="both", expand=True)
    
    def _setup_layout(self):
        """Setup the layout and initial state."""
        self._clear_results()

    def _save_results_to_csv(self):
        """Saves the current generation results to a CSV file."""
        if not self.saved_results:
            messagebox.showwarning("No Data", "No results to save — please run the algorithm first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            title="Save results as..."
        )

        if not file_path:
            return  # user cancelled

        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["generation", "best_fitness", "best_individual", "mean_fitness", "std_dev"])
                writer.writeheader()
                writer.writerows(self.saved_results)
            messagebox.showinfo("Saved", f"Results saved to file:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{e}")
    
    def _run_ga(self):
        """Run the genetic algorithm in a separate thread."""
        if self.is_running:
            return
            
        try:
            # Get configuration and objective function
            config = self.config_panel.get_config()
            objective_function = self.config_panel.get_objective_function()
            minimize = self.config_panel.is_minimize()
            
            # Validate configuration
            if config.population_size <= 0 or config.num_epochs <= 0:
                messagebox.showerror("Error", "Population size and epochs must be positive!")
                return
            
            # Validate function-specific requirements
            function_name = self.config_panel.config_vars['function_name'].get()
            num_variables = self.config_panel.config_vars['num_variables'].get()
            
            # Basic validation - all functions need at least 1 variable
            if num_variables < 1:
                messagebox.showerror("Error", "Number of variables must be at least 1!")
                return
            
            # Function-specific validation
            if function_name in ['michalewicz'] and num_variables < 2:
                messagebox.showerror("Error", "Michalewicz function requires at least 2 variables!")
                return
            
            # Update UI state
            self.is_running = True
            self.run_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.progress_var.set("Initializing...")
            self.progress_bar['value'] = 0
            self.progress_bar['maximum'] = config.num_epochs
            
            # Clear previous results
            self.plot_panel.clear_plots()
            self.results_text.delete(1.0, tk.END)
            
            # Create GA instance
            self.ga = GeneticAlgorithm(config)
            self.ga.set_objective(minimize)
            
            # Start execution time tracking
            self.start_time = time.time()
            
            # Run GA in separate thread
            def run_ga_thread():
                try:
                    # Progress callback
                    def progress_callback(result: GenerationResult):
                        if not self.is_running:
                            return
                            
                        # Save results to variable
                        self.saved_results.append({
                            "generation": result.generation,
                            "best_fitness": result.best_fitness,
                            "best_individual": result.best_individual,
                            "mean_fitness": result.average_fitness,
                            "std_dev": result.std_fitness
                        })
                        # Update progress in GUI thread
                        self.root.after(0, self._update_progress, result)
                    
                    # Run GA
                    self.results = self.ga.run(objective_function, progress_callback)
                    
                    # Update final results in GUI thread
                    self.root.after(0, self._finish_execution)
                    
                except Exception as e:
                    # Handle errors in GUI thread
                    error_msg = str(e)
                    self.root.after(0, lambda: self._handle_error(error_msg))
            
            # Start thread
            thread = threading.Thread(target=run_ga_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self._handle_error(str(e))
    
    def _update_progress(self, result: GenerationResult):
        """Update progress during GA execution."""
        if not self.is_running:
            return
            
        # Update progress bar
        self.progress_bar['value'] = result.generation + 1
        
        # Update status
        elapsed_time = time.time() - self.start_time
        self.execution_time_var.set(f"Execution Time: {elapsed_time:.2f}s")
        self.progress_var.set(f"Generation {result.generation + 1}/{self.ga.config.num_epochs}")
        
        # Update plots every few generations to avoid slowdown
        if (result.generation + 1) % max(1, self.ga.config.num_epochs // 50) == 0 or result.generation == 0:
            convergence_data = self.ga._extract_convergence_data()
            self.plot_panel.update_plots(convergence_data)
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Current Best: {result.best_fitness:.6f}\n")
        self.results_text.insert(tk.END, f"Average: {result.average_fitness:.6f}\n")
        self.results_text.insert(tk.END, f"Std Dev: {result.std_fitness:.6f}\n")
        self.results_text.insert(tk.END, f"Generation Time: {result.execution_time:.4f}s")
    
    def _finish_execution(self):
        """Finish GA execution and display final results."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        
        if self.results:
            # Final time update
            total_time = time.time() - self.start_time
            self.execution_time_var.set(f"Total Time: {total_time:.2f}s")
            self.progress_var.set("Completed")
            
            # Final plot update
            convergence_data = self.results['convergence_data']
            self.plot_panel.update_plots(convergence_data)
            
            # Display final results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"=== FINAL RESULTS ===\n")
            self.results_text.insert(tk.END, f"Best Fitness: {self.results['best_fitness']:.6f}\n")
            self.results_text.insert(tk.END, f"Best Solution: {self.results['best_solution']}\n")
            self.results_text.insert(tk.END, f"Total Time: {total_time:.2f}s\n")
            self.results_text.insert(tk.END, f"Generations: {self.results['total_generations']}")
    
    def _stop_ga(self):
        """Stop the GA execution."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_var.set("Stopped")
    
    def _clear_results(self):
        """Clear all results and plots."""
        self.plot_panel.clear_plots()
        self.results_text.delete(1.0, tk.END)
        self.execution_time_var.set("Execution Time: 0.00s")
        self.progress_var.set("Ready")
        self.progress_bar['value'] = 0
        self.results = None
    
    def _handle_error(self, error_message: str):
        """Handle errors during execution."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_var.set("Error occurred")
        messagebox.showerror("Error", f"An error occurred:\n{error_message}")
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            config = self.config_panel.get_config()
            config_dict = config.to_dict()
            
            # Add GUI-specific settings
            config_dict['function_name'] = self.config_panel.config_vars['function_name'].get()
            config_dict['minimize'] = self.config_panel.config_vars['minimize'].get()
            config_dict['num_variables'] = self.config_panel.config_vars['num_variables'].get()
            config_dict['bound_min'] = self.config_panel.config_vars['bound_min'].get()
            config_dict['bound_max'] = self.config_panel.config_vars['bound_max'].get()
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                messagebox.showinfo("Success", "Configuration saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config_dict = json.load(f)
                
                # Update configuration variables
                for key, value in config_dict.items():
                    if key in self.config_panel.config_vars:
                        var = self.config_panel.config_vars[key]
                        if isinstance(var, (tk.IntVar, tk.DoubleVar, tk.BooleanVar, tk.StringVar)):
                            var.set(value)
                
                messagebox.showinfo("Success", "Configuration loaded successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = GAApp()
    app.run()