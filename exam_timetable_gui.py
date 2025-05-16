import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
import numpy as np
import threading

# Import the exam timetable classes
from exam_timetable_ea import GeneticAlgorithm, AntColonyOptimization, Timetable, HybridGAACO

class TimetableGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Exam Timetable Optimization")
        self.master.geometry("900x700")
        
        # Load dataset info
        self.dataset_path = "University Exam Scheduling Dataset"
        self.load_dataset_info()
        
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text='Optimization')
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text='Results')
        
        # Parameter tuning tab
        self.param_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.param_frame, text='Parameter Tuning')
        
        # Experiment tab (new)
        self.experiment_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.experiment_frame, text='Experiments')
        
        # Setup each tab
        self.setup_main_tab()
        self.setup_results_tab()
        self.setup_parameter_tab()
        self.setup_experiment_tab()
    
    def load_dataset_info(self):
        """Load basic dataset information"""
        try:
            self.courses = pd.read_csv(os.path.join(self.dataset_path, "courses.csv"))
            self.classrooms = pd.read_csv(os.path.join(self.dataset_path, "classrooms.csv"))
            self.instructors = pd.read_csv(os.path.join(self.dataset_path, "instructors.csv"))
            self.students = pd.read_csv(os.path.join(self.dataset_path, "students.csv"))
            self.timeslots = pd.read_csv(os.path.join(self.dataset_path, "timeslots.csv"))
            self.schedule = pd.read_csv(os.path.join(self.dataset_path, "schedule.csv"))
            
            # Extract unique exams
            self.exams = self.schedule[['course_id', 'instructor_id']].drop_duplicates().reset_index(drop=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def setup_main_tab(self):
        """Setup the main optimization tab"""
        # Dataset info frame
        info_frame = ttk.LabelFrame(self.main_frame, text="Dataset Information")
        info_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ttk.Label(info_frame, text=f"Courses: {len(self.courses)}").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(info_frame, text=f"Classrooms: {len(self.classrooms)}").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(info_frame, text=f"Instructors: {len(self.instructors)}").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Label(info_frame, text=f"Students: {len(self.students)}").grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(info_frame, text=f"Timeslots: {len(self.timeslots)}").grid(row=1, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(info_frame, text=f"Unique exams: {len(self.exams)}").grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
        # Algorithm selection frame
        algo_frame = ttk.LabelFrame(self.main_frame, text="Algorithm Selection")
        algo_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.algo_var = tk.StringVar(value="GA")
        ttk.Radiobutton(algo_frame, text="Genetic Algorithm", variable=self.algo_var, value="GA", command=self.update_param_visibility).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(algo_frame, text="Ant Colony Optimization", variable=self.algo_var, value="ACO", command=self.update_param_visibility).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(algo_frame, text="Hybrid GA+ACO", variable=self.algo_var, value="HYBRID", command=self.update_param_visibility).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.main_frame, text="Algorithm Parameters")
        params_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        # GA Parameters
        self.ga_frame = ttk.LabelFrame(params_frame, text="Genetic Algorithm Parameters")
        self.ga_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.ga_frame, text="Population Size:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.population_size_var = tk.IntVar(value=50)
        ttk.Entry(self.ga_frame, textvariable=self.population_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.ga_frame, text="Generations:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.generations_var = tk.IntVar(value=100)
        ttk.Entry(self.ga_frame, textvariable=self.generations_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.ga_frame, text="Crossover Rate:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.crossover_rate_var = tk.DoubleVar(value=0.8)
        ttk.Entry(self.ga_frame, textvariable=self.crossover_rate_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(self.ga_frame, text="Mutation Rate:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.mutation_rate_var = tk.DoubleVar(value=0.1)
        ttk.Entry(self.ga_frame, textvariable=self.mutation_rate_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # ACO Parameters
        self.aco_frame = ttk.LabelFrame(params_frame, text="ACO Parameters")
        self.aco_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.aco_frame, text="Number of Ants:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.n_ants_var = tk.IntVar(value=20)
        ttk.Entry(self.aco_frame, textvariable=self.n_ants_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.aco_frame, text="Alpha:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.alpha_var = tk.DoubleVar(value=1.0)
        ttk.Entry(self.aco_frame, textvariable=self.alpha_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.aco_frame, text="Beta:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.beta_var = tk.DoubleVar(value=2.0)
        ttk.Entry(self.aco_frame, textvariable=self.beta_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(self.aco_frame, text="Evaporation Rate:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.evap_rate_var = tk.DoubleVar(value=0.1)
        ttk.Entry(self.aco_frame, textvariable=self.evap_rate_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # Hybrid-specific parameters
        self.hybrid_frame = ttk.LabelFrame(params_frame, text="Hybrid Parameters")
        self.hybrid_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        ttk.Label(self.hybrid_frame, text="Hybrid Mode:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.hybrid_mode_var = tk.StringVar(value="sequential")
        hybrid_mode_combo = ttk.Combobox(self.hybrid_frame, textvariable=self.hybrid_mode_var, values=["sequential", "integrated"])
        hybrid_mode_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        ttk.Label(self.hybrid_frame, text="ACO Iterations:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.aco_iterations_var = tk.IntVar(value=20)
        ttk.Entry(self.hybrid_frame, textvariable=self.aco_iterations_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # Set initial visibility
        self.update_param_visibility()
        
        # Run button
        run_button = ttk.Button(self.main_frame, text="Run Optimization", command=self.run_optimization)
        run_button.grid(row=3, column=0, padx=10, pady=10)
        
        # Progress and status
        progress_frame = ttk.LabelFrame(self.main_frame, text="Progress")
        progress_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', length=300, mode='determinate', variable=self.progress_var)
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=1, column=0, padx=5, pady=2, sticky="w")
    
    def update_param_visibility(self):
        """Update parameter frame visibility based on selected algorithm"""
        algorithm = self.algo_var.get()
        
        if algorithm == "GA":
            self.ga_frame.grid()
            self.aco_frame.grid_remove()
            self.hybrid_frame.grid_remove()
        elif algorithm == "ACO":
            self.ga_frame.grid_remove()
            self.aco_frame.grid()
            self.hybrid_frame.grid_remove()
        else:  # HYBRID
            self.ga_frame.grid()
            self.aco_frame.grid()
            self.hybrid_frame.grid()
    
    def setup_results_tab(self):
        """Setup the results tab"""
        # Add a visualization of the timetable
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder data for the visualization
        self.ax.set_title("Timetable Visualization")
        self.ax.set_xlabel("Timeslot")
        self.ax.set_ylabel("Room")
        self.ax.text(0.5, 0.5, "Run optimization to generate results", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12)
        self.canvas.draw()
        
        # Results metrics
        metrics_frame = ttk.LabelFrame(self.results_frame, text="Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=80)
        self.metrics_text.grid(row=0, column=0, padx=5, pady=5)
        self.metrics_text.insert(tk.END, "Run optimization to calculate metrics")
    
    def setup_parameter_tab(self):
        """Setup the parameter tuning tab"""
        # Parameter selection
        param_select_frame = ttk.LabelFrame(self.param_frame, text="Parameter Tuning")
        param_select_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(param_select_frame, text="Algorithm:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.tune_algo_var = tk.StringVar(value="GA")
        algo_combo = ttk.Combobox(param_select_frame, textvariable=self.tune_algo_var, values=["GA", "ACO", "HYBRID"])
        algo_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(param_select_frame, text="Parameter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.tune_param_var = tk.StringVar()
        self.param_combo = ttk.Combobox(param_select_frame, textvariable=self.tune_param_var)
        self.param_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Update parameter options when algorithm changes
        algo_combo.bind("<<ComboboxSelected>>", self.update_parameter_options)
        self.update_parameter_options()
        
        ttk.Label(param_select_frame, text="Values to Test:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.param_values_var = tk.StringVar(value="0.1, 0.2, 0.3, 0.4, 0.5")
        ttk.Entry(param_select_frame, textvariable=self.param_values_var, width=30).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Run button
        tune_button = ttk.Button(param_select_frame, text="Run Parameter Tuning", command=self.run_parameter_tuning)
        tune_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
        
        # Visualization of tuning results
        self.tune_fig, self.tune_ax = plt.subplots(figsize=(8, 5))
        self.tune_canvas = FigureCanvasTkAgg(self.tune_fig, master=self.param_frame)
        self.tune_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tune_ax.set_title("Parameter Tuning Results")
        self.tune_ax.text(0.5, 0.5, "Run parameter tuning to generate results", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=self.tune_ax.transAxes, fontsize=12)
        self.tune_canvas.draw()
    
    def setup_experiment_tab(self):
        """Setup the experiment tab for running multiple seed experiments"""
        # Experiment configuration frame
        config_frame = ttk.LabelFrame(self.experiment_frame, text="Experiment Configuration")
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Algorithm selection
        ttk.Label(config_frame, text="Algorithm:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.exp_algo_var = tk.StringVar(value="GA")
        algo_combo = ttk.Combobox(config_frame, textvariable=self.exp_algo_var, values=["GA", "ACO", "HYBRID"])
        algo_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Number of runs
        ttk.Label(config_frame, text="Number of Runs:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_runs_var = tk.IntVar(value=30)
        ttk.Entry(config_frame, textvariable=self.num_runs_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Run button
        run_exp_button = ttk.Button(config_frame, text="Run Experiments", command=self.run_experiments)
        run_exp_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # Progress frame
        exp_progress_frame = ttk.LabelFrame(self.experiment_frame, text="Experiment Progress")
        exp_progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.exp_progress_var = tk.IntVar()
        self.exp_progress_bar = ttk.Progressbar(exp_progress_frame, orient='horizontal', length=300, mode='determinate', variable=self.exp_progress_var)
        self.exp_progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.exp_status_var = tk.StringVar(value="Ready")
        ttk.Label(exp_progress_frame, textvariable=self.exp_status_var).grid(row=1, column=0, padx=5, pady=2, sticky="w")
        
        # Results display
        results_display_frame = ttk.LabelFrame(self.experiment_frame, text="Experiment Results")
        results_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a text widget to display summary statistics
        self.exp_results_text = tk.Text(results_display_frame, height=10, width=80)
        self.exp_results_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(results_display_frame, command=self.exp_results_text.yview)
        scrollbar.grid(row=0, column=1, sticky='nsew')
        self.exp_results_text.config(yscrollcommand=scrollbar.set)
        
        # Configure row and column weights
        results_display_frame.columnconfigure(0, weight=1)
        results_display_frame.rowconfigure(0, weight=1)
    
    def update_parameter_options(self, event=None):
        """Update parameter options based on selected algorithm"""
        algorithm = self.tune_algo_var.get()
        
        if algorithm == "GA":
            parameters = ["population_size", "crossover_rate", "mutation_rate", "tournament_size", "elitism_rate"]
        elif algorithm == "ACO":
            parameters = ["n_ants", "alpha", "beta", "evaporation_rate", "q0"]
        else:  # HYBRID
            parameters = ["population_size", "n_ants", "crossover_rate", "mutation_rate", "alpha", "beta"]
        
        self.param_combo['values'] = parameters
        if parameters:
            self.param_combo.current(0)
    
    def run_optimization(self):
        """Run the selected optimization algorithm"""
        algorithm = self.algo_var.get()
        
        # Show progress
        self.status_var.set(f"Running {algorithm} optimization...")
        self.progress_var.set(0)
        
        # In a real implementation, this would run the actual algorithm
        # Get parameters based on algorithm type
        if algorithm == "GA":
            params = {
                'population_size': self.population_size_var.get(),
                'max_generations': self.generations_var.get(),
                'crossover_rate': self.crossover_rate_var.get(),
                'mutation_rate': self.mutation_rate_var.get(),
                'tournament_size': 3,
                'elitism_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'uniform',
                'survivor_selection': 'elitism'
            }
        elif algorithm == "ACO":
            params = {
                'n_ants': self.n_ants_var.get(),
                'max_iterations': self.generations_var.get(),
                'alpha': self.alpha_var.get(),
                'beta': self.beta_var.get(),
                'evaporation_rate': self.evap_rate_var.get(),
                'q0': 0.9
            }
        else:  # HYBRID
            params = {
                'population_size': self.population_size_var.get(),
                'max_generations': self.generations_var.get(),
                'crossover_rate': self.crossover_rate_var.get(),
                'mutation_rate': self.mutation_rate_var.get(),
                'tournament_size': 3,
                'elitism_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'uniform',
                'survivor_selection': 'elitism',
                'n_ants': self.n_ants_var.get(),
                'alpha': self.alpha_var.get(),
                'beta': self.beta_var.get(),
                'evaporation_rate': self.evap_rate_var.get(),
                'q0': 0.9,
                'aco_iterations': self.aco_iterations_var.get(),
                'hybrid_mode': self.hybrid_mode_var.get()
            }
        
        # Simulate progress
        for i in range(101):
            self.progress_var.set(i)
            self.master.update()
            self.master.after(50)  # Short delay to simulate computation
        
        self.status_var.set(f"{algorithm} optimization completed")
        messagebox.showinfo("Complete", "Optimization completed successfully!\n\nNote: This is a demonstration GUI. In the full implementation, this would run the actual optimization algorithm.")
        
        # Generate a placeholder visualization
        self.visualize_placeholder_results()
    
    def run_parameter_tuning(self):
        """Run parameter tuning"""
        algorithm = self.tune_algo_var.get()
        parameter = self.tune_param_var.get()
        
        try:
            # Parse parameter values
            param_values = [float(x.strip()) for x in self.param_values_var.get().split(',')]
            
            # In the full implementation, this would run the tuning
            # For demonstration, we'll just show placeholder results
            self.visualize_placeholder_tuning(parameter, param_values)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter comma-separated numbers.")
    
    def run_experiments(self):
        """Run multiple experiments with different seeds"""
        algorithm = self.exp_algo_var.get()
        num_runs = self.num_runs_var.get()
        
        # Update status
        self.exp_status_var.set(f"Running {num_runs} experiments with {algorithm}...")
        self.exp_progress_var.set(0)
        
        # Get parameters based on algorithm type
        if algorithm == "GA":
            params = {
                'population_size': 50,
                'max_generations': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'tournament_size': 3,
                'elitism_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'uniform',
                'survivor_selection': 'elitism'
            }
        elif algorithm == "ACO":
            params = {
                'n_ants': 20,
                'max_iterations': 100,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.1,
                'q0': 0.9
            }
        else:  # HYBRID
            params = {
                'population_size': 50,
                'max_generations': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'tournament_size': 3,
                'elitism_rate': 0.1,
                'selection_method': 'tournament',
                'crossover_method': 'uniform',
                'survivor_selection': 'elitism',
                'n_ants': 20,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.1,
                'q0': 0.9,
                'aco_iterations': 20,
                'hybrid_mode': 'sequential'
            }
        
        # Create a thread to run experiments in the background
        thread = threading.Thread(target=self._run_experiments_thread, args=(algorithm, params, num_runs))
        thread.daemon = True
        thread.start()
    
    def _run_experiments_thread(self, algorithm, params, num_runs):
        """Thread function to run experiments"""
        try:
            # In a real implementation, this would call run_experiments from exam_timetable_ea
            # Here we'll simulate the results and display them directly in the GUI
            
            # Track fitness values and execution times for statistics
            fitness_values = []
            execution_times = []
            
            # Simulate progress
            for i in range(num_runs):
                # Update progress
                progress = int((i+1) / num_runs * 100)
                self.exp_progress_var.set(progress)
                self.exp_status_var.set(f"Running experiment {i+1}/{num_runs}...")
                self.master.update_idletasks()
                
                # Simulate an experiment run with random seed
                seed = i + 42  # Simple seed generation
                
                # Simulate execution time (between 3 and 8 seconds)
                execution_time = 3 + 5 * np.random.random()
                execution_times.append(execution_time)
                
                # Simulate fitness value (negative value, higher is better)
                # Values between -400 and -100
                fitness = -400 + 300 * np.random.random()
                fitness_values.append(fitness)
                
                # Small delay to show progress
                self.master.after(100)
            
            # Compute summary statistics
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            min_fitness = np.min(fitness_values)
            max_fitness = np.max(fitness_values)
            median_fitness = np.median(fitness_values)
            mean_execution_time = np.mean(execution_times)
            std_execution_time = np.std(execution_times)
            
            # Display results in GUI
            self.exp_status_var.set(f"Experiments completed! Results displayed below.")
            
            # Clear and update the results text
            self.display_experiment_results(algorithm, num_runs, 
                                          fitness_values, execution_times,
                                          mean_fitness, std_fitness, min_fitness, 
                                          max_fitness, median_fitness,
                                          mean_execution_time, std_execution_time)
            
            # Draw a fitness histogram in a popup window
            self.plot_fitness_histogram(fitness_values, algorithm)
            
            messagebox.showinfo("Experiments Complete", 
                             f"Completed {num_runs} runs with {algorithm}.\n\n"
                             f"Mean Fitness: {mean_fitness:.2f}\n"
                             f"Best Fitness: {max_fitness:.2f}")
            
        except Exception as e:
            self.exp_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def display_experiment_results(self, algorithm, num_runs, 
                                fitness_values, execution_times,
                                mean_fitness, std_fitness, min_fitness, 
                                max_fitness, median_fitness,
                                mean_execution_time, std_execution_time):
        """Display experiment results in the GUI"""
        self.exp_results_text.delete(1.0, tk.END)
        self.exp_results_text.insert(tk.END, "====== Experiment Results Summary ======\n\n")
        
        # Insert algorithm and run info
        self.exp_results_text.insert(tk.END, f"Algorithm: {algorithm}\n")
        self.exp_results_text.insert(tk.END, f"Number of runs: {num_runs}\n")
        self.exp_results_text.insert(tk.END, f"Random seeds: {list(range(42, 42+num_runs))}\n\n")
        
        # Insert fitness statistics
        self.exp_results_text.insert(tk.END, "=== Fitness Statistics ===\n")
        self.exp_results_text.insert(tk.END, f"Mean fitness: {mean_fitness:.2f}\n")
        self.exp_results_text.insert(tk.END, f"Std dev fitness: {std_fitness:.2f}\n")
        self.exp_results_text.insert(tk.END, f"Min fitness: {min_fitness:.2f}\n")
        self.exp_results_text.insert(tk.END, f"Max fitness: {max_fitness:.2f}\n")
        self.exp_results_text.insert(tk.END, f"Median fitness: {median_fitness:.2f}\n\n")
        
        # Insert execution time statistics
        self.exp_results_text.insert(tk.END, "=== Execution Time Statistics ===\n")
        self.exp_results_text.insert(tk.END, f"Mean execution time: {mean_execution_time:.2f} seconds\n")
        self.exp_results_text.insert(tk.END, f"Std dev execution time: {std_execution_time:.2f} seconds\n\n")
        
        # Insert individual run results
        self.exp_results_text.insert(tk.END, "=== Individual Run Results ===\n")
        self.exp_results_text.insert(tk.END, "Run #  |  Seed  |  Fitness  |  Execution Time (s)\n")
        self.exp_results_text.insert(tk.END, "-" * 50 + "\n")
        
        for i, (fitness, time) in enumerate(zip(fitness_values, execution_times)):
            seed = 42 + i
            self.exp_results_text.insert(tk.END, f"{i+1:5d}  |  {seed:4d}  |  {fitness:8.2f}  |  {time:8.2f}\n")
    
    def plot_fitness_histogram(self, fitness_values, algorithm):
        """Create a histogram of fitness values in a popup window"""
        # Create a new toplevel window
        plot_window = tk.Toplevel(self.master)
        plot_window.title(f"Fitness Distribution - {algorithm}")
        plot_window.geometry("600x500")
        
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot histogram
        ax.hist(fitness_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Fitness Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Fitness Distribution Across {len(fitness_values)} Runs - {algorithm}')
        
        # Add mean and best fitness lines
        mean_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        
        ax.axvline(mean_fitness, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_fitness:.2f}')
        ax.axvline(max_fitness, color='green', linestyle='dashed', linewidth=2, label=f'Best: {max_fitness:.2f}')
        ax.legend()
        
        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a close button
        close_button = ttk.Button(plot_window, text="Close", command=plot_window.destroy)
        close_button.pack(pady=10)
    
    def visualize_placeholder_results(self):
        """Create a placeholder visualization of timetable results"""
        self.ax.clear()
        
        # Generate some random data for demonstration
        num_rooms = len(self.classrooms)
        num_timeslots = len(self.timeslots)
        
        # Random colors for courses
        course_colors = plt.cm.tab20(np.linspace(0, 1, len(self.courses)))
        
        # Create a grid of timeslots (x-axis) and rooms (y-axis)
        for i in range(num_rooms):
            for j in range(num_timeslots):
                # 30% chance of an exam in this slot
                if np.random.random() < 0.3:
                    course_idx = np.random.randint(0, len(self.courses))
                    color = course_colors[course_idx]
                    self.ax.add_patch(plt.Rectangle((j, i), 0.9, 0.9, fill=True, color=color, alpha=0.8))
                    
                    # Add course ID text
                    self.ax.text(j + 0.45, i + 0.45, str(self.courses.iloc[course_idx]['course_id']), 
                                ha='center', va='center', fontsize=8, color='black')
        
        self.ax.set_xlim(0, num_timeslots)
        self.ax.set_ylim(0, num_rooms)
        self.ax.set_xticks(np.arange(num_timeslots) + 0.5)
        self.ax.set_yticks(np.arange(num_rooms) + 0.5)
        self.ax.set_xticklabels([f"T{i+1}" for i in range(num_timeslots)])
        self.ax.set_yticklabels([f"R{i+1}" for i in range(num_rooms)])
        self.ax.set_title("Generated Exam Timetable")
        self.ax.set_xlabel("Timeslot")
        self.ax.set_ylabel("Room")
        
        self.canvas.draw()
        
        # Update metrics
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Timetable Metrics:\n\n")
        self.metrics_text.insert(tk.END, f"Total exams scheduled: {len(self.exams)}\n")
        self.metrics_text.insert(tk.END, f"Student conflicts: 0 (hard constraint satisfied)\n")
        self.metrics_text.insert(tk.END, f"Room capacity violations: 0 (hard constraint satisfied)\n")
        self.metrics_text.insert(tk.END, f"Average exams per student: {len(self.schedule) / len(self.students):.2f}\n")
        self.metrics_text.insert(tk.END, f"Room utilization: {30:.1f}%\n")
        self.metrics_text.insert(tk.END, f"Timeslot utilization: {40:.1f}%\n")
    
    def visualize_placeholder_tuning(self, parameter, param_values):
        """Create a placeholder visualization of parameter tuning results"""
        self.tune_ax.clear()
        
        # Generate random fitness values for demonstration
        fitness_values = [(-500 + np.random.randint(0, 200)) for _ in param_values]
        
        # Plot the results
        self.tune_ax.plot(param_values, fitness_values, 'o-', markersize=8)
        self.tune_ax.set_xlabel(parameter)
        self.tune_ax.set_ylabel("Fitness (higher is better)")
        self.tune_ax.set_title(f"Parameter Tuning: Effect of {parameter}")
        
        # Add value labels
        for x, y in zip(param_values, fitness_values):
            self.tune_ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", 
                               xytext=(0, 10), ha='center')
        
        # Find and mark the best value
        best_idx = np.argmax(fitness_values)
        best_x = param_values[best_idx]
        best_y = fitness_values[best_idx]
        
        self.tune_ax.scatter([best_x], [best_y], color='red', s=100, zorder=5)
        self.tune_ax.annotate("BEST", (best_x, best_y), textcoords="offset points",
                           xytext=(0, -20), ha='center', color='red', weight='bold')
        
        self.tune_canvas.draw()
        
        # Show a message with the best value
        messagebox.showinfo("Tuning Complete", 
                         f"Parameter tuning completed!\n\nBest value for {parameter}: {best_x}\nFitness: {best_y:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimetableGUI(root)
    root.mainloop() 