# Exam Timetable Optimization Using Evolutionary Algorithms

An academic project implementing and comparing various evolutionary algorithms for solving the University Exam Timetabling Problem.

## Project Overview

This project implements three optimization approaches to solve the complex university exam scheduling problem:

1. **Genetic Algorithm (GA)** - Uses evolutionary principles with selection, crossover, and mutation
2. **Ant Colony Optimization (ACO)** - Uses stigmergy and pheromone trails to construct solutions
3. **Hybrid GA-ACO** - Combines both approaches for potentially better results

The algorithms schedule exams considering multiple constraints including:
- No student having multiple exams in the same timeslot (hard constraint)
- Room capacity limits (hard constraint)
- Exams well spread out for students (soft constraint)

## Dataset

The project uses a university exam scheduling dataset with the following components:
- Courses information
- Classroom details with capacities
- Instructor information
- Student enrollment data
- Available timeslots
- Schedule of student-course-instructor relationships
- The link of it https://www.kaggle.com/datasets/smrezwanulazad/exam-schedule

## Features

- **Multiple Algorithm Implementation** - GA, ACO, and Hybrid approaches
- **Customizable Parameters** - Adjust population size, generations, crossover rate, mutation rate, etc.
- **GUI Interface** - Visual interface for running optimizations and viewing results
- **Parameter Tuning** - Tools to find optimal parameter settings
- **Experiment Features** - Run multiple trials with statistical analysis
- **Result Visualization** - Graphical display of optimized timetables and metrics

## Requirements

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - tkinter

## Usage

1. **Run the GUI application:**
   ```
   python exam_timetable_gui.py
   ```

2. **Select Algorithm:**
   - Choose between GA, ACO, or Hybrid approaches

3. **Configure Parameters:**
   - Adjust algorithm-specific parameters to optimize performance

4. **Run Optimization:**
   - Click "Run Optimization" to generate an optimized exam timetable

5. **View Results:**
   - Visualize the generated timetable
   - Review statistics and metrics on solution quality

## Project Structure

- `exam_timetable_ea.py` - Core evolutionary algorithm implementations
- `exam_timetable_gui.py` - GUI interface for the application
- `University Exam Scheduling Dataset/` - Dataset files
- `experiment_results/` - Directory for storing experiment results

## Future Improvements

- Implementation of additional evolutionary approaches
- Multi-objective optimization to handle competing constraints
- Improved visualization of exam conflicts
- Performance optimizations for larger datasets 
