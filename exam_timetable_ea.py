# =========================
# Block 1: Imports and Setup
# This block imports required libraries and sets up file paths and constants.
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import copy
import time
import os
import json
import csv
from datetime import datetime

# https://www.kaggle.com/datasets/smrezwanulazad/exam-schedule/data  (dataset used for this code)

# Set random seed for reproducibility
RANDOM_SEEDS = [42, 123, 256, 389, 512, 645, 778, 891, 934, 1045,
                1156, 1267, 1378, 1489, 1590, 1601, 1712, 1823, 1934, 2045,
                2156, 2267, 2378, 2489, 2590, 2601, 2712, 2823, 2934, 3045]

# File paths
DATASET_PATH = "University Exam Scheduling Dataset"
COURSES_FILE = os.path.join(DATASET_PATH, "courses.csv")
CLASSROOMS_FILE = os.path.join(DATASET_PATH, "classrooms.csv")
INSTRUCTORS_FILE = os.path.join(DATASET_PATH, "instructors.csv")
STUDENTS_FILE = os.path.join(DATASET_PATH, "students.csv")
TIMESLOTS_FILE = os.path.join(DATASET_PATH, "timeslots.csv")
SCHEDULE_FILE = os.path.join(DATASET_PATH, "schedule.csv")

# Results directory
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)






# =========================
# Block 2: Data Loading
# Loads all required CSV files into pandas DataFrames.
# =========================

courses = pd.read_csv(COURSES_FILE)
classrooms = pd.read_csv(CLASSROOMS_FILE)
instructors = pd.read_csv(INSTRUCTORS_FILE)
students = pd.read_csv(STUDENTS_FILE)
timeslots = pd.read_csv(TIMESLOTS_FILE)
schedule = pd.read_csv(SCHEDULE_FILE)

print(f"Loaded {len(courses)} courses")
print(f"Loaded {len(classrooms)} classrooms")
print(f"Loaded {len(instructors)} instructors")
print(f"Loaded {len(students)} students")
print(f"Loaded {len(timeslots)} timeslots")
print(f"Loaded {len(schedule)} schedule entries")






# =========================
# Block 3: Exam Extraction
# Extracts unique exams and builds student-exam registration matrix.
# =========================

# Extract unique exams (course_id, instructor_id combinations)
exams = schedule[['course_id', 'instructor_id']].drop_duplicates().reset_index(drop=True)
print(f"Found {len(exams)} unique exams to schedule")

# Create student-exam registration matrix
# For each student, which exams are they taking?
student_exam_registrations = {}
for student_id in students['student_id'].unique():
    student_exams = []
    student_schedule = schedule[schedule['student_id'] == student_id]
    for _, row in student_schedule.iterrows():
        course_id = row['course_id']
        instructor_id = row['instructor_id']
        # Find the exam index in our exams dataframe
        exam_idx = exams[(exams['course_id'] == course_id) & 
                          (exams['instructor_id'] == instructor_id)].index.tolist()
        if exam_idx:
            student_exams.append(exam_idx[0])
    student_exam_registrations[student_id] = student_exams

# Get room capacities as a list
room_capacities = classrooms['capacity'].tolist()






# =========================
# Block 4: Parameter Ranges and Results Tracking
# Sets up parameter ranges for tuning and initializes results tracking.
# =========================

PARAMETER_RANGES = {
    'population_size': [20, 50, 100, 200],
    'max_generations': [50, 100, 200, 500],
    'tournament_size': [2, 3, 5, 7],
    'crossover_rate': [0.7, 0.8, 0.9],
    'mutation_rate': [0.01, 0.05, 0.1, 0.2],
    'elitism_rate': [0.05, 0.1, 0.2],
    # ACO parameters
    'n_ants': [5, 10, 20, 30],
    'alpha': [0.5, 1.0, 2.0],
    'beta': [1.0, 2.0, 5.0],
    'evaporation_rate': [0.1, 0.3, 0.5, 0.7],
    'q0': [0.0, 0.5, 0.9]  # Exploitation probability
}

# Results tracking
experiment_results = {}






# =========================
# Block 5: Timetable Class
# Defines the Timetable class representing a solution and its fitness/diversity.
# =========================

class Timetable:
    """
    Class representing a timetable solution
    """
    def __init__(self, num_exams, num_timeslots, num_rooms):
        self.num_exams = num_exams
        self.num_timeslots = num_timeslots
        self.num_rooms = num_rooms
        
        # Initialize the chromosome as a list of (timeslot, room) tuples for each exam
        self.chromosome = [(random.randint(0, num_timeslots-1), 
                            random.randint(0, num_rooms-1)) 
                           for _ in range(num_exams)]
        
        self.fitness = 0  # Will be calculated later
        self.diversity_score = 0  # For diversity preservation
    
    def calculate_fitness(self, student_exam_registrations, room_capacities):
        """
        Calculate fitness based on constraints:
        1. Hard constraint: No student has two exams at the same time
        2. Hard constraint: Room capacities are not exceeded
        3. Soft constraint: Exams are well spread out for students
        """
        # Initialize counters for constraint violations
        student_conflicts = 0
        capacity_violations = 0
        proximity_penalty = 0
        
        # Check student conflicts (hard constraint)
        for student_id, exams in student_exam_registrations.items():
            # Group exams by timeslot
            timeslot_exams = {}
            for exam_idx in exams:
                timeslot = self.chromosome[exam_idx][0]
                if timeslot not in timeslot_exams:
                    timeslot_exams[timeslot] = []
                timeslot_exams[timeslot].append(exam_idx)
            
            # Count conflicts (more than one exam in the same timeslot)
            for timeslot, timeslot_exam_list in timeslot_exams.items():
                if len(timeslot_exam_list) > 1:
                    student_conflicts += len(timeslot_exam_list) - 1
            
            # Check proximity (soft constraint) - exams too close together
            timeslots_with_exams = sorted(timeslot_exams.keys())
            for i in range(len(timeslots_with_exams) - 1):
                time_gap = timeslots_with_exams[i+1] - timeslots_with_exams[i]
                if time_gap <= 1:  # Adjacent timeslots
                    proximity_penalty += 1
                elif time_gap <= 2:  # One timeslot gap
                    proximity_penalty += 0.5
        
        # Check room capacity violations (hard constraint)
        room_usage = {}
        for i, (timeslot, room) in enumerate(self.chromosome):
            key = (timeslot, room)
            if key not in room_usage:
                room_usage[key] = []
            room_usage[key].append(i)
        
        # Count students in each room-timeslot and check against capacity
        for (timeslot, room), exams_in_room in room_usage.items():
            # Count unique students with exams in this room & timeslot
            students_in_room = set()
            for exam_idx in exams_in_room:
                for student_id, student_exams in student_exam_registrations.items():
                    if exam_idx in student_exams:
                        students_in_room.add(student_id)
            
            # Check if room capacity is exceeded
            if len(students_in_room) > room_capacities[room]:
                capacity_violations += len(students_in_room) - room_capacities[room]
        
        # Calculate fitness (higher is better)
        # Hard constraints are heavily penalized
        hard_penalty = 1000 * (student_conflicts + capacity_violations)
        soft_penalty = 10 * proximity_penalty
        
        # Raw fitness is negative of penalties (higher values = better solutions)
        raw_fitness = -(hard_penalty + soft_penalty)
        
        self.fitness = raw_fitness
        return raw_fitness
    
    def calculate_diversity(self, population):
        """Calculate how different this timetable is from others in the population"""
        if not population:
            self.diversity_score = 0
            return 0
            
        total_distance = 0
        for other in population:
            if other is self:
                continue
                
            # Calculate Hamming distance between chromosomes
            distance = sum(1 for a, b in zip(self.chromosome, other.chromosome) if a != b)
            total_distance += distance
        
        avg_distance = total_distance / max(1, len(population) - 1)
        self.diversity_score = avg_distance
        return avg_distance
    
    def mutate(self, mutation_rate):
        """Mutate the chromosome by changing timeslots or rooms"""
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                # 50% chance to mutate timeslot, 50% chance to mutate room
                if random.random() < 0.5:
                    # Mutate timeslot
                    timeslot, room = self.chromosome[i]
                    new_timeslot = random.randint(0, self.num_timeslots - 1)
                    self.chromosome[i] = (new_timeslot, room)
                else:
                    # Mutate room
                    timeslot, room = self.chromosome[i]
                    new_room = random.randint(0, self.num_rooms - 1)
                    self.chromosome[i] = (timeslot, new_room)
    
    def __str__(self):
        return f"Timetable(fitness={self.fitness:.2f}, diversity={self.diversity_score:.2f})"






# =========================
# Block 6: Genetic Algorithm Class
# Implements the Genetic Algorithm for exam timetabling.
# =========================

class GeneticAlgorithm:
    """
    Genetic Algorithm implementation for exam timetabling
    """
    def __init__(self, 
                population_size=100,
                max_generations=200,
                crossover_rate=0.8,
                mutation_rate=0.1,
                tournament_size=3,
                elitism_rate=0.1,
                selection_method='tournament',
                crossover_method='uniform',
                survivor_selection='elitism'):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.survivor_selection = survivor_selection
        
        # Stats tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.diversity_history = []
        
    def initialize_population(self, num_exams, num_timeslots, num_rooms):
        """Create an initial population of timetables"""
        return [Timetable(num_exams, num_timeslots, num_rooms) 
                for _ in range(self.population_size)]
    
    def evaluate_population(self, population, student_exam_registrations, room_capacities):
        """Calculate fitness for all individuals in the population"""
        for individual in population:
            individual.calculate_fitness(student_exam_registrations, room_capacities)
            individual.calculate_diversity(population)
            
        # Update stats
        fitness_values = [ind.fitness for ind in population]
        self.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
        best_individual = max(population, key=lambda x: x.fitness)
        self.best_fitness_history.append(best_individual.fitness)
        
        # Update best solution if we found a better one
        if self.best_solution is None or best_individual.fitness > self.best_solution.fitness:
            self.best_solution = copy.deepcopy(best_individual)
        
        # Track diversity
        diversity_values = [ind.diversity_score for ind in population]
        self.diversity_history.append(sum(diversity_values) / len(diversity_values))

    def tournament_selection(self, population, tournament_size):
        """Select parent using tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def roulette_wheel_selection(self, population):
        """Select parent using fitness-proportionate selection"""
        # Adjust fitness to be positive for all individuals
        min_fitness = min(ind.fitness for ind in population)
        adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in population]
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_fitness)
        selection_probs = [fit/total_fitness for fit in adjusted_fitness]
        
        # Select based on probabilities
        return np.random.choice(population, p=selection_probs)
    
    def select_parent(self, population):
        """Select a parent based on the chosen selection method"""
        if self.selection_method == 'tournament':
            return self.tournament_selection(population, self.tournament_size)
        elif self.selection_method == 'roulette':
            return self.roulette_wheel_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def uniform_crossover(self, parent1, parent2):
        """Create child using uniform crossover"""
        child = Timetable(parent1.num_exams, parent1.num_timeslots, parent1.num_rooms)
        
        for i in range(len(parent1.chromosome)):
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child.chromosome[i] = parent1.chromosome[i]
            else:
                child.chromosome[i] = parent2.chromosome[i]
        
        return child
    
    def ordered_crossover(self, parent1, parent2):
        """Create child using ordered crossover (adapted for timetabling)"""
        child = Timetable(parent1.num_exams, parent1.num_timeslots, parent1.num_rooms)
        
        # Choose two crossover points
        cx_points = sorted(random.sample(range(len(parent1.chromosome)), 2))
        
        # Copy a segment from parent1
        for i in range(cx_points[0], cx_points[1]):
            child.chromosome[i] = parent1.chromosome[i]
        
        # Fill the rest from parent2, preserving order but avoiding duplicates
        for i in range(len(parent1.chromosome)):
            if i < cx_points[0] or i >= cx_points[1]:
                child.chromosome[i] = parent2.chromosome[i]
        
        return child
    
    def crossover(self, parent1, parent2):
        """Perform crossover based on chosen method"""
        if random.random() > self.crossover_rate:
            # No crossover, return a copy of parent1
            child = copy.deepcopy(parent1)
            return child
        
        if self.crossover_method == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_method == 'ordered':
            return self.ordered_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def elitism_selection(self, population, offspring, elitism_count):
        """Select survivors using elitism"""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        sorted_offspring = sorted(offspring, key=lambda x: x.fitness, reverse=True)
        
        # Take the best from both populations
        new_pop = sorted_pop[:elitism_count]
        
        # Fill the rest with the best offspring
        remaining = self.population_size - elitism_count
        new_pop.extend(sorted_offspring[:remaining])
        
        return new_pop
    
    def crowding_selection(self, population, offspring):
        """Select survivors using crowding (diversity preservation)"""
        # Combine populations
        combined = population + offspring
        
        # Calculate diversity scores
        for ind in combined:
            ind.calculate_diversity(combined)
        
        # Sort by fitness and diversity (weighted sum)
        combined.sort(key=lambda x: x.fitness + 0.2 * x.diversity_score, reverse=True)
        
        # Take the best individuals
        return combined[:self.population_size]
    
    def select_survivors(self, population, offspring):
        """Select survivors based on chosen method"""
        if self.survivor_selection == 'elitism':
            elitism_count = int(self.elitism_rate * self.population_size)
            return self.elitism_selection(population, offspring, elitism_count)
        elif self.survivor_selection == 'crowding':
            return self.crowding_selection(population, offspring)
        else:
            raise ValueError(f"Unknown survivor selection method: {self.survivor_selection}")
    
    def run(self, num_exams, num_timeslots, num_rooms, student_exam_registrations, room_capacities):
        """Run the genetic algorithm"""
        # Initialize population
        population = self.initialize_population(num_exams, num_timeslots, num_rooms)
        
        # Evaluate initial population
        self.evaluate_population(population, student_exam_registrations, room_capacities)
        
        # Main loop
        for generation in range(self.max_generations):
            # Create offspring
            offspring = []
            
            while len(offspring) < self.population_size:
                # Select parents
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)
                
                # Create child
                child = self.crossover(parent1, parent2)
                
                # Mutate child
                child.mutate(self.mutation_rate)
                
                # Add to offspring
                offspring.append(child)
            
            # Evaluate offspring
            for individual in offspring:
                individual.calculate_fitness(student_exam_registrations, room_capacities)
            
            # Select survivors for next generation
            population = self.select_survivors(population, offspring)
            
            # Evaluate new population
            self.evaluate_population(population, student_exam_registrations, room_capacities)
            
            # Print progress
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_generations}, "
                      f"Best Fitness: {self.best_fitness_history[-1]:.2f}, "
                      f"Average Fitness: {self.avg_fitness_history[-1]:.2f}, "
                      f"Diversity: {self.diversity_history[-1]:.2f}")
        
        return self.best_solution






# =========================
# Block 7: Ant Colony Optimization Class
# Implements the Ant Colony Optimization (ACO) algorithm.
# =========================

class AntColonyOptimization:
    """
    Ant Colony Optimization for exam timetabling
    """
    def __init__(self, 
                n_ants=20,
                max_iterations=100,
                alpha=1.0,
                beta=2.0,
                evaporation_rate=0.1,
                q0=0.9):  # Exploitation probability for ACS
        
        self.n_ants = n_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q0 = q0
        
        # Stats tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        
    def initialize_pheromones(self, num_exams, num_timeslots, num_rooms):
        """Initialize pheromone matrix with small values"""
        # Create a 3D pheromone matrix: exam -> (timeslot, room)
        pheromones = np.ones((num_exams, num_timeslots, num_rooms))
        return pheromones
    
    def calculate_heuristic(self, exam, timeslot, room, assigned_exams, 
                           student_exam_registrations, room_capacities):
        """
        Calculate heuristic value for assigning an exam to a timeslot and room.
        Lower value = better assignment
        """
        # Count potential student conflicts
        conflicts = 0
        # Count how many students are in this room for this timeslot
        students_in_room = set()
        
        # Check each exam already assigned to this timeslot
        for other_exam, (other_timeslot, other_room) in assigned_exams.items():
            if other_timeslot == timeslot:
                # Check for student conflicts (students who have both exams)
                for student_id, exams in student_exam_registrations.items():
                    if exam in exams and other_exam in exams:
                        conflicts += 1
                
                # Count students in this room
                if other_room == room:
                    for student_id, exams in student_exam_registrations.items():
                        if other_exam in exams:
                            students_in_room.add(student_id)
        
        # For the current exam, count how many more students would be added to the room
        for student_id, exams in student_exam_registrations.items():
            if exam in exams:
                students_in_room.add(student_id)
        
        # Check room capacity
        capacity_violation = max(0, len(students_in_room) - room_capacities[room])
        
        # Calculate total penalty (higher penalty = lower heuristic value)
        penalty = 1000 * conflicts + 500 * capacity_violation
        
        # Return the inverse (higher is better)
        return 1.0 / (penalty + 1)
    
    def acs_state_transition(self, exam, pheromones, assigned_exams, 
                           student_exam_registrations, room_capacities,
                           num_timeslots, num_rooms):
        """ACS state transition rule to choose next timeslot and room for an exam"""
        # With probability q0, choose the best option (exploitation)
        if random.random() < self.q0:
            best_value = -1
            best_combination = (0, 0)
            
            for timeslot in range(num_timeslots):
                for room in range(num_rooms):
                    # Calculate heuristic value
                    heuristic = self.calculate_heuristic(
                        exam, timeslot, room, assigned_exams, 
                        student_exam_registrations, room_capacities
                    )
                    
                    # Calculate value based on pheromone and heuristic
                    value = pheromones[exam, timeslot, room]**self.alpha * heuristic**self.beta
                    
                    if value > best_value:
                        best_value = value
                        best_combination = (timeslot, room)
            
            return best_combination
        
        # Otherwise, use the proportional rule (exploration)
        else:
            # Calculate probabilities for all combinations
            probabilities = []
            combinations = []
            
            for timeslot in range(num_timeslots):
                for room in range(num_rooms):
                    # Calculate heuristic value
                    heuristic = self.calculate_heuristic(
                        exam, timeslot, room, assigned_exams, 
                        student_exam_registrations, room_capacities
                    )
                    
                    # Calculate value based on pheromone and heuristic
                    value = pheromones[exam, timeslot, room]**self.alpha * heuristic**self.beta
                    
                    probabilities.append(value)
                    combinations.append((timeslot, room))
            
            # Normalize probabilities
            total = sum(probabilities)
            if total == 0:
                # If all have zero probability, choose randomly
                return (random.randint(0, num_timeslots-1), random.randint(0, num_rooms-1))
                
            probabilities = [p/total for p in probabilities]
            
            # Choose based on probabilities
            return random.choices(combinations, weights=probabilities, k=1)[0]
    
    def construct_solution(self, pheromones, num_exams, num_timeslots, num_rooms, 
                         student_exam_registrations, room_capacities):
        """Construct a solution by assigning exams one by one"""
        # Start with an empty solution
        assigned_exams = {}
        
        # Randomly order exams for assignment (diversity enhancement)
        exam_order = list(range(num_exams))
        random.shuffle(exam_order)
        
        # Assign each exam
        for exam in exam_order:
            # Choose timeslot and room using ACS rule
            timeslot, room = self.acs_state_transition(
                exam, pheromones, assigned_exams, 
                student_exam_registrations, room_capacities,
                num_timeslots, num_rooms
            )
            
            # Assign the exam
            assigned_exams[exam] = (timeslot, room)
            
            # Local pheromone update (ACS specific)
            pheromones[exam, timeslot, room] = (1 - self.evaporation_rate) * pheromones[exam, timeslot, room] + \
                                              self.evaporation_rate * 0.1
        
        # Convert to Timetable object for fitness calculation
        solution = Timetable(num_exams, num_timeslots, num_rooms)
        for exam, (timeslot, room) in assigned_exams.items():
            solution.chromosome[exam] = (timeslot, room)
        
        return solution
    
    def update_pheromones(self, pheromones, solutions):
        """Update pheromone matrix based on the quality of solutions"""
        # Evaporate pheromones
        pheromones *= (1 - self.evaporation_rate)
        
        # Add new pheromones based on solution quality
        for solution in solutions:
            # Calculate quality (higher fitness = more pheromone)
            quality = max(0, solution.fitness)  # Ensure positive
            
            # Add pheromones based on solution
            for exam, (timeslot, room) in enumerate(solution.chromosome):
                pheromones[exam, timeslot, room] += quality
        
        return pheromones
    
    def run(self, num_exams, num_timeslots, num_rooms, student_exam_registrations, room_capacities):
        """Run the ACO algorithm"""
        # Initialize pheromone matrix
        pheromones = self.initialize_pheromones(num_exams, num_timeslots, num_rooms)
        
        # Main loop
        for iteration in range(self.max_iterations):
            # Construct solutions
            solutions = []
            for ant in range(self.n_ants):
                solution = self.construct_solution(
                    pheromones, num_exams, num_timeslots, num_rooms,
                    student_exam_registrations, room_capacities
                )
                solution.calculate_fitness(student_exam_registrations, room_capacities)
                solutions.append(solution)
            
            # Update stats
            fitness_values = [sol.fitness for sol in solutions]
            self.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
            best_in_iteration = max(solutions, key=lambda x: x.fitness)
            self.best_fitness_history.append(best_in_iteration.fitness)
            
            # Update best solution
            if self.best_solution is None or best_in_iteration.fitness > self.best_solution.fitness:
                self.best_solution = copy.deepcopy(best_in_iteration)
            
            # Update pheromones
            pheromones = self.update_pheromones(pheromones, solutions)
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best Fitness: {self.best_fitness_history[-1]:.2f}, "
                      f"Average Fitness: {self.avg_fitness_history[-1]:.2f}")
        
        return self.best_solution 






# =========================
# Block 8: Hybrid GA+ACO Class
# Implements the Hybrid approach combining GA and ACO.
# =========================

class HybridGAACO:
    """
    Hybrid GA and ACO implementation for exam timetabling
    """
    def __init__(self, 
                population_size=50,
                max_generations=100,
                crossover_rate=0.8,
                mutation_rate=0.1,
                tournament_size=3,
                elitism_rate=0.1,
                selection_method='tournament',
                crossover_method='uniform',
                survivor_selection='elitism',
                n_ants=20,
                alpha=1.0,
                beta=2.0,
                evaporation_rate=0.1,
                q0=0.9,
                aco_iterations=20,
                hybrid_mode='sequential'):
        
        # GA parameters
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.survivor_selection = survivor_selection
        
        # ACO parameters
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q0 = q0
        self.aco_iterations = aco_iterations
        
        # Hybrid parameters
        self.hybrid_mode = hybrid_mode  # 'sequential' or 'integrated'
        
        # Stats tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.diversity_history = []
    
    def run(self, num_exams, num_timeslots, num_rooms, student_exam_registrations, room_capacities):
        """Run the hybrid GA+ACO algorithm"""
        
        if self.hybrid_mode == 'sequential':
            # Run GA first, then use its best solution to initialize ACO
            return self.run_sequential(num_exams, num_timeslots, num_rooms, 
                                      student_exam_registrations, room_capacities)
        else:
            # Run GA and ACO in an integrated manner
            return self.run_integrated(num_exams, num_timeslots, num_rooms, 
                                      student_exam_registrations, room_capacities)
    
    def run_sequential(self, num_exams, num_timeslots, num_rooms, 
                      student_exam_registrations, room_capacities):
        """
        Run GA first, then use its best solution to initialize ACO.
        The ACO will refine the GA solution.
        """
        print("Running Sequential Hybrid GA+ACO...")
        
        # Step 1: Run GA to get initial solution
        ga = GeneticAlgorithm(
            population_size=self.population_size,
            max_generations=self.max_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            tournament_size=self.tournament_size,
            elitism_rate=self.elitism_rate,
            selection_method=self.selection_method,
            crossover_method=self.crossover_method,
            survivor_selection=self.survivor_selection
        )
        
        ga_solution = ga.run(num_exams, num_timeslots, num_rooms, 
                          student_exam_registrations, room_capacities)
        
        # Copy GA stats for tracking
        self.best_fitness_history = ga.best_fitness_history.copy()
        self.avg_fitness_history = ga.avg_fitness_history.copy()
        self.diversity_history = ga.diversity_history.copy()
        self.best_solution = copy.deepcopy(ga_solution)
        
        print(f"GA phase completed. Best fitness: {ga_solution.fitness:.2f}")
        
        # Step 2: Initialize ACO with pheromones biased towards GA solution
        aco = AntColonyOptimization(
            n_ants=self.n_ants,
            max_iterations=self.aco_iterations,
            alpha=self.alpha,
            beta=self.beta,
            evaporation_rate=self.evaporation_rate,
            q0=self.q0
        )
        
        # Initialize pheromone matrix with a bias towards the GA solution
        pheromones = aco.initialize_pheromones(num_exams, num_timeslots, num_rooms)
        
        # Increase pheromone levels at positions from GA solution
        for exam, (timeslot, room) in enumerate(ga_solution.chromosome):
            pheromones[exam, timeslot, room] *= 10.0  # Significant boost to pheromone level
        
        # Run ACO with the biased pheromones
        # We'll modify the ACO run method slightly to accept initial pheromones
        aco_solution = self.run_aco_with_pheromones(aco, pheromones, num_exams, num_timeslots, 
                                                   num_rooms, student_exam_registrations, room_capacities)
        
        # Extend our tracking history with ACO results
        self.best_fitness_history.extend(aco.best_fitness_history)
        self.avg_fitness_history.extend(aco.avg_fitness_history)
        
        # Check if ACO improved the solution
        if aco_solution.fitness > self.best_solution.fitness:
            self.best_solution = copy.deepcopy(aco_solution)
            print(f"ACO improved the solution! New best fitness: {aco_solution.fitness:.2f}")
        else:
            print(f"ACO did not improve the solution. Keeping GA solution with fitness: {self.best_solution.fitness:.2f}")
        
        return self.best_solution
    
    def run_aco_with_pheromones(self, aco, initial_pheromones, num_exams, num_timeslots, 
                               num_rooms, student_exam_registrations, room_capacities):
        """Run ACO with a given initial pheromone matrix"""
        # This is essentially a modified version of the ACO run method
        
        # Use the provided pheromone matrix
        pheromones = initial_pheromones
        
        # Main loop
        for iteration in range(aco.max_iterations):
            # Construct solutions
            solutions = []
            for ant in range(aco.n_ants):
                solution = aco.construct_solution(
                    pheromones, num_exams, num_timeslots, num_rooms,
                    student_exam_registrations, room_capacities
                )
                solution.calculate_fitness(student_exam_registrations, room_capacities)
                solutions.append(solution)
            
            # Update stats
            fitness_values = [sol.fitness for sol in solutions]
            aco.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
            best_in_iteration = max(solutions, key=lambda x: x.fitness)
            aco.best_fitness_history.append(best_in_iteration.fitness)
            
            # Update best solution
            if aco.best_solution is None or best_in_iteration.fitness > aco.best_solution.fitness:
                aco.best_solution = copy.deepcopy(best_in_iteration)
            
            # Update pheromones
            pheromones = aco.update_pheromones(pheromones, solutions)
            
            # Print progress
            if (iteration + 1) % 5 == 0:
                print(f"ACO Iteration {iteration + 1}/{aco.max_iterations}, "
                      f"Best Fitness: {aco.best_fitness_history[-1]:.2f}, "
                      f"Average Fitness: {aco.avg_fitness_history[-1]:.2f}")
        
        return aco.best_solution
    
    def run_integrated(self, num_exams, num_timeslots, num_rooms, 
                      student_exam_registrations, room_capacities):
        """
        Run GA and ACO in an integrated manner.
        After each GA generation, run ACO for a few iterations to improve solutions.
        """
        print("Running Integrated Hybrid GA+ACO...")
        
        # Initialize GA components
        ga = GeneticAlgorithm(
            population_size=self.population_size,
            max_generations=self.max_generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            tournament_size=self.tournament_size,
            elitism_rate=self.elitism_rate,
            selection_method=self.selection_method,
            crossover_method=self.crossover_method,
            survivor_selection=self.survivor_selection
        )
        
        # Initialize ACO components
        aco = AntColonyOptimization(
            n_ants=self.n_ants,
            max_iterations=5,  # Few iterations each time
            alpha=self.alpha,
            beta=self.beta,
            evaporation_rate=self.evaporation_rate,
            q0=self.q0
        )
        
        # Initialize population
        population = ga.initialize_population(num_exams, num_timeslots, num_rooms)
        
        # Evaluate initial population
        ga.evaluate_population(population, student_exam_registrations, room_capacities)
        
        # Initialize pheromone matrix
        pheromones = aco.initialize_pheromones(num_exams, num_timeslots, num_rooms)
        
        # Copy initial stats
        self.best_fitness_history = ga.best_fitness_history.copy()
        self.avg_fitness_history = ga.avg_fitness_history.copy()
        self.diversity_history = ga.diversity_history.copy()
        self.best_solution = copy.deepcopy(max(population, key=lambda x: x.fitness))
        
        # Main loop - alternate between GA and ACO
        for generation in range(self.max_generations):
            # Run one generation of GA
            
            # Create offspring
            offspring = []
            
            while len(offspring) < ga.population_size:
                # Select parents
                parent1 = ga.select_parent(population)
                parent2 = ga.select_parent(population)
                
                # Create child
                child = ga.crossover(parent1, parent2)
                
                # Mutate child
                child.mutate(ga.mutation_rate)
                
                # Add to offspring
                offspring.append(child)
            
            # Evaluate offspring
            for individual in offspring:
                individual.calculate_fitness(student_exam_registrations, room_capacities)
            
            # Select survivors for next generation
            population = ga.select_survivors(population, offspring)
            
            # Evaluate new population
            ga.evaluate_population(population, student_exam_registrations, room_capacities)
            
            # Update best solution if found a better one
            best_individual = max(population, key=lambda x: x.fitness)
            if best_individual.fitness > self.best_solution.fitness:
                self.best_solution = copy.deepcopy(best_individual)
            
            # Update tracking history
            self.best_fitness_history.append(best_individual.fitness)
            self.avg_fitness_history.append(sum(ind.fitness for ind in population) / len(population))
            self.diversity_history.append(sum(ind.diversity_score for ind in population) / len(population))
            
            # Every few generations, apply ACO to improve the population
            if (generation + 1) % 5 == 0:
                # Update pheromones based on current population
                for individual in population:
                    quality = max(0, individual.fitness)
                    for exam, (timeslot, room) in enumerate(individual.chromosome):
                        pheromones[exam, timeslot, room] += quality
                
                # Run ACO for a few iterations
                aco_solutions = []
                for _ in range(5):  # Short ACO run
                    for ant in range(aco.n_ants):
                        solution = aco.construct_solution(
                            pheromones, num_exams, num_timeslots, num_rooms,
                            student_exam_registrations, room_capacities
                        )
                        solution.calculate_fitness(student_exam_registrations, room_capacities)
                        aco_solutions.append(solution)
                    
                    # Update pheromones
                    pheromones = aco.update_pheromones(pheromones, aco_solutions)
                
                # Select best ACO solutions to inject into GA population
                aco_solutions.sort(key=lambda x: x.fitness, reverse=True)
                inject_count = min(len(aco_solutions), int(ga.population_size * 0.2))
                
                # Replace worst individuals in GA population with best ACO solutions
                population.sort(key=lambda x: x.fitness)
                for i in range(inject_count):
                    if i < len(population) and i < len(aco_solutions):
                        population[i] = copy.deepcopy(aco_solutions[i])
                
                # Ensure diversity in the population
                for ind in population:
                    ind.calculate_diversity(population)
            
            # Print progress
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_generations}, "
                      f"Best Fitness: {self.best_fitness_history[-1]:.2f}, "
                      f"Average Fitness: {self.avg_fitness_history[-1]:.2f}, "
                      f"Diversity: {self.diversity_history[-1]:.2f}")
        
        return self.best_solution






# =========================
# Block 9: Main Function
# Provides a simple main function for demonstration and instructions.
# =========================

def main():
    print("Exam Timetabling Optimization Demo")
    print("----------------------------------")
    
    # Get problem dimensions
    num_exams = len(exams)
    num_timeslots = len(timeslots)
    num_rooms = len(classrooms)
    
    print(f"Problem dimensions: {num_exams} exams, {num_timeslots} timeslots, {num_rooms} rooms")
    
    # To run the full implementation, you would instantiate and run one of the algorithms:
    # ga = GeneticAlgorithm(population_size=50, max_generations=100)
    # best_solution = ga.run(num_exams, num_timeslots, num_rooms, student_exam_registrations, room_capacities)
    # 
    # Or for ACO:
    # aco = AntColonyOptimization(n_ants=20, max_iterations=100)
    # best_solution = aco.run(num_exams, num_timeslots, num_rooms, student_exam_registrations, room_capacities)
    #
    # Or for Hybrid GA+ACO:
    # hybrid = HybridGAACO(population_size=50, max_generations=100, n_ants=20, hybrid_mode='sequential')
    # best_solution = hybrid.run(num_exams, num_timeslots, num_rooms, student_exam_registrations, room_capacities)
    
    print("\nThis is a placeholder for the full implementation.")
    print("The complete code includes:")
    print("- Genetic Algorithm with multiple selection, crossover, and survivor selection methods")
    print("- Ant Colony Optimization with ACS variant")
    print("- Hybrid GA+ACO combination for enhanced performance")
    print("- Multi-seed experimentation with statistical analysis")
    print("- GUI for visualization and parameter tuning")
    print("- Comprehensive fitness evaluation considering hard and soft constraints")
    print("- Diversity preservation mechanisms")
    print("- Parameter tuning capabilities")
    print("- Results tracking and visualization")

if __name__ == "__main__":
    main() 






# =========================
# Block 10: Experiment Runner
# Defines a function to run multiple experiments and save results.
# =========================

def run_experiments(algorithm_type='GA', params=None, num_runs=30, seeds=None):
    """
    Run multiple experiments with different random seeds and save the results
    
    Parameters:
    -----------
    algorithm_type : str
        'GA', 'ACO', or 'HYBRID'
    params : dict
        Parameters for the algorithm
    num_runs : int
        Number of runs to perform
    seeds : list
        List of random seeds to use. If None, RANDOM_SEEDS will be used
    
    Returns:
    --------
    results : dict
        Dictionary containing experiment results
    """
    if seeds is None:
        seeds = RANDOM_SEEDS[:num_runs]
    
    if params is None:
        if algorithm_type == 'GA':
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
        elif algorithm_type == 'ACO':
            params = {
                'n_ants': 20,
                'max_iterations': 100,
                'alpha': 1.0,
                'beta': 2.0,
                'evaporation_rate': 0.1,
                'q0': 0.9
            }
        elif algorithm_type == 'HYBRID':
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
                'hybrid_mode': 'sequential'  # 'sequential' or 'integrated'
            }
    
    # Get problem dimensions
    num_exams = len(exams)
    num_timeslots = len(timeslots)
    num_rooms = len(classrooms)
    
    # Prepare results storage
    all_results = {
        'algorithm': algorithm_type,
        'parameters': params,
        'seeds': seeds,
        'runs': []
    }
    
    # Run experiments
    for i, seed in enumerate(seeds):
        print(f"Running experiment {i+1}/{len(seeds)} with seed {seed}")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Start timer
        start_time = time.time()
        
        # Run the appropriate algorithm
        if algorithm_type == 'GA':
            algorithm = GeneticAlgorithm(**params)
            solution = algorithm.run(num_exams, num_timeslots, num_rooms, 
                                     student_exam_registrations, room_capacities)
            
            # Record results
            run_results = {
                'seed': seed,
                'best_fitness': solution.fitness,
                'best_fitness_history': algorithm.best_fitness_history,
                'avg_fitness_history': algorithm.avg_fitness_history,
                'diversity_history': algorithm.diversity_history,
                'execution_time': time.time() - start_time
            }
            
        elif algorithm_type == 'ACO':
            algorithm = AntColonyOptimization(**params)
            solution = algorithm.run(num_exams, num_timeslots, num_rooms, 
                                     student_exam_registrations, room_capacities)
            
            # Record results
            run_results = {
                'seed': seed,
                'best_fitness': solution.fitness,
                'best_fitness_history': algorithm.best_fitness_history,
                'avg_fitness_history': algorithm.avg_fitness_history,
                'execution_time': time.time() - start_time
            }
            
        elif algorithm_type == 'HYBRID':
            algorithm = HybridGAACO(**params)
            solution = algorithm.run(num_exams, num_timeslots, num_rooms, 
                                     student_exam_registrations, room_capacities)
            
            # Record results
            run_results = {
                'seed': seed,
                'best_fitness': solution.fitness,
                'best_fitness_history': algorithm.best_fitness_history,
                'avg_fitness_history': algorithm.avg_fitness_history,
                'diversity_history': algorithm.diversity_history,
                'execution_time': time.time() - start_time
            }
        
        all_results['runs'].append(run_results)
    
    # Calculate summary statistics
    fitness_values = [run['best_fitness'] for run in all_results['runs']]
    execution_times = [run['execution_time'] for run in all_results['runs']]
    
    all_results['summary'] = {
        'mean_fitness': np.mean(fitness_values),
        'std_fitness': np.std(fitness_values),
        'min_fitness': np.min(fitness_values),
        'max_fitness': np.max(fitness_values),
        'median_fitness': np.median(fitness_values),
        'mean_execution_time': np.mean(execution_times),
        'std_execution_time': np.std(execution_times)
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_str = "_".join([f"{k}-{v}" for k, v in list(params.items())[:3]])
    filename = f"{algorithm_type}_{params_str}_{timestamp}"
    
    # Save detailed results as JSON
    json_path = os.path.join(RESULTS_DIR, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary as CSV
    csv_path = os.path.join(RESULTS_DIR, f"{filename}_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in all_results['summary'].items():
            writer.writerow([key, value])
    
    # Create and save summary plots
    # Fitness history plot
    plt.figure(figsize=(12, 6))
    
    # Plot best fitness across all runs
    best_fitness_curves = [run['best_fitness_history'] for run in all_results['runs']]
    max_length = max(len(curve) for curve in best_fitness_curves)
    
    # Pad shorter curves to match the longest one
    padded_curves = []
    for curve in best_fitness_curves:
        if len(curve) < max_length:
            padded = curve + [curve[-1]] * (max_length - len(curve))
        else:
            padded = curve
        padded_curves.append(padded)
    
    best_fitness_matrix = np.array(padded_curves)
    
    # Plot mean and std
    mean_curve = np.mean(best_fitness_matrix, axis=0)
    std_curve = np.std(best_fitness_matrix, axis=0)
    
    x = np.arange(max_length)
    plt.plot(x, mean_curve, 'b-', label='Mean Best Fitness')
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                     alpha=0.2, color='b', label='±1 Std Dev')
    
    plt.xlabel('Generation/Iteration')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Evolution over {num_runs} Runs - {algorithm_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_fitness.png"), dpi=300)
    
    print(f"Experiment results saved to {RESULTS_DIR}")
    return all_results