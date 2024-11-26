import random
import numpy as np

from data_structures.problem_structure import Problem, Solution, Tile

def stop_condition(i, max_iter):
    '''Warunek stopu osiągniecie zadanej ilości iteracji i = max_iter'''
    return i >= max_iter # jedyny warunek stopu to osiągniecie zadanej ilości wywołań

def first_population_generator(n, generation_size):
    '''Generowanie n macierzy wielkości [generation_size x generation_size]'''

    return [Forrest(np.random.randint(0, 20, (generation_size, generation_size))) for _ in range(n)]

    '''Generowanie n macierzy wielkości [generation_size x generation_size] z posortowanymi wartościami'''
    # ordered_array = n * np.arange(1, generation_size*generation_size+1).reshape(generation_size, generation_size)
    # print(ordered_array)
    # return first_population



class GeneticAlgorithm:

    def __init__(self, first_population_generator: callable,
                selection_model: callable, stop_condition: callable, 
                pupulation_size: int = 100, generation_size: int = 15,
                mutation_probability: float = 0.1, max_iter: int = 100):
        
        self.first_generation_func = first_population_generator # funkcja generująca pierwsze pokolenie
        self.selection_model = selection_model # funkcja selekcji
        self.stop_condition = stop_condition # warunek stopu
        self.pupulation_size = pupulation_size # wielkość populacji
        self.generation_size = generation_size # wielkość macierzy
        self.mutation_probability = mutation_probability # prawdopodobienstwo mutacji
        self.max_iter = max_iter # maksymalna ilosc wywolan

    def run(self):
        population = self.first_generation_func(self.generation_size, self.pupulation_size)
        population.sort(key=lambda x: x.fitness)
        population_len = len(population)
        i = 0
        while True:
            selected = self.selection_model(population)
            new_population = selected.copy()
            while len(new_population) != population_len:
                child = choice(population).crossover(choice(population))
                if random() <= self.mutation_probability:
                    child.mutation()
                new_population.append(child)

            population = new_population
            the_best_match = min(population, key=lambda x: x.fitness)
            print("Generation: {} S: {} fitness: {}".format(i, the_best_match, the_best_match.fitness))
            i += 1
            if self.stop_condition(i, self.max_iter):
                break


def simple_selection_model(population):
    """
    Select the top 50% of the population based on fitness.
    """
    return population[:len(population) // 2]