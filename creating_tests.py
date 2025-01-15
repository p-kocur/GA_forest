from algorithm.algorithm_body import GeneticAlgorithm
from algorithm.genetic_functions import mutate_to_legal, basic_mutation, permutation_mutation, territorial_mutation, max_reward_mutation, expansion_mutation
from data_structures.problem_structure import Problem
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
#import pickle
import dill as pickle


    
# Odkomentuj tylko za pierwszym razem, żeby stworzyć problem
'''   
problem = Problem(size=[15, 15], mutation_probs=[1, 0, 0, 0, 0],
            mutation_functions=[basic_mutation, permutation_mutation, territorial_mutation, max_reward_mutation, expansion_mutation],
            mutate_to_legal=mutate_to_legal, basic_mutation=basic_mutation, n=20)
with open("problem.pkl", 'wb') as f:
        pickle.dump(problem, f)
'''
            
            
with open("problem.pkl", 'rb') as f:
    problem = pickle.load(f)
            
algorithm = GeneticAlgorithm(
    crossover_probs=[1, 0 , 0, 0],
    selection_types="elitism_selection",
    mutation_probs=[1, 0, 0, 0, 0],
    population_size = 100,
    max_iter = 1000,
    mutation_probability = 0.1,
    elite_percentage = 20,
    load_population = False, # W kolejnych True
    problem=problem,
)

# Uruchomienie algorytmu
best_sol, bests, avgs, worsts = algorithm.run()
