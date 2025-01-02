import random
import numpy as np

from data_structures.problem_structure import Problem, Solution, Tile
from algorithm.genetic_functions import (
    naive_crossover,
    single_point_crossover_vector,
    single_point_crossover_random,
    single_point_matrix_crossover,
    multi_point_crossover_vector,
    uniform_crossover_naive,
    elitism_selection,
    roulette_selection,
    tournament_selection,
    elitist_roulette_selection,
    elitist_tournament_selection,
    basic_mutation,
    territorial_mutation,
    permutation_mutation,
    max_reward_mutation,
    expansion_mutation,
    mutate_to_legal
)


class GeneticAlgorithm:
    """
    #### Klasa główna algorytmu genetycznego.
    ----------
    #### Parametry
    ----------
    crossover_probabilities: list = None, lista dostepnych metod krzyżowania wraz z ich prawdopodobienstwem
    selection_probabilities: list = None, lista dostepnych metod selekcji wraz z ich prawdopodobienstwem
    mutation_probability: list = None, lista dostepnych metod mutacji wraz z ich prawdopodobienstwem
    first_population: list = None, przekazanie startowej populacji, None: losowa generacja pierwszej generacji
    population_size: int = 100, wielkosc generacji (populacji), tj. ilość Solution
    max_iter: int = 100, maksymalna ilosc iteracji
    problem: Problem = None, problem który rozwiazujemy
    leave_parents: bool = True, jezeli True wszystkie selekcje działają na zasadzie elity tj. populacja pozostała po selekcji jest przekazywana do kolejnej generacji  

    Schemat rozwiązania:
    """

    def __init__(self,
                crossover_probabilities: list = None,
                selection_probabilities: list = None,
                mutation_probabilities: list = None,
                mutation_probability: float = 0.1,
                first_population: list = None, 
                population_size: int = 1000,
                max_iter: int = 1000,
                problem: Problem = None,
                leave_parents: bool = True):
        

        # listy crossover_probabilities oraz selection_proability
        # jest tylko po to że taka forma jest bardziej czytelna 
        # niz cztery odzielne listy faktycznie wykorzystywane do algorytmu  

        if not crossover_probabilities:
            crossover_probabilities = [
            {"crossover": naive_crossover, "probability": 1.0}
            #{"crossover": single_point_crossover_random, "probability": 0.80},
            #{"crossover": single_point_matrix_crossover, "probability": 0.60},
            #{"crossover": single_point_crossover_vector, "probability": 0.75},
            #{"crossover": multi_point_crossover_vector, "probability": 0.65},
            #{"crossover": uniform_crossover_naive, "probability": 0.50}
            ]
        self.crossover_types = [entry["crossover"] for entry in crossover_probabilities]
        self.crossover_weights = [entry["probability"] for entry in crossover_probabilities]
            
        if not selection_probabilities:
            selection_probabilities = [
            {"selection": elitism_selection, "probability": 0.80},
            {"selection": roulette_selection, "probability": 0.60},
            {"selection": elitist_roulette_selection, "probability": 0.75},
            {"selection": tournament_selection, "probability": 0.65},
            {"selection": elitist_tournament_selection, "probability": 0.50}
            ]
        self.selection_types = [entry["selection"] for entry in selection_probabilities]
        self.selection_weights = [entry["probability"] for entry in selection_probabilities]
        
        if not mutation_probabilities:
            mutation_probabilities = [
            {"mutation": basic_mutation, "probability": 0.2},
            {"mutation": territorial_mutation, "probability": 0.2},
            {"mutation": permutation_mutation, "probability": 0.2},
            {"mutation": max_reward_mutation, "probability": 0.2},
            {"mutation": expansion_mutation, "probability": 0.2}
            ]
        self.mutation_types = [entry["mutation"] for entry in mutation_probabilities]
        self.mutation_weights = [entry["probability"] for entry in mutation_probabilities]
  
        if problem is None:
            self.problem = Problem(self.mutation_types, self.mutation_weights, basic_mutation=basic_mutation, mutate_to_legal=mutate_to_legal)
        else:
            self.problem = problem
  
        self.mutation_probability = mutation_probability
        self.first_population = first_population 
        self.population_size = population_size
        self.max_iter = max_iter
        self.leave_parents = leave_parents

    def run(self):
        if not self.first_population:
            population = self.generate_first_population(self.problem.n, self.population_size)
        
        # Aktualizujemy "wartość" każdego rozwiązania w populacji
        for solution in population:
            solution.evaluate_function()

        generation = 0

        
        bests = []
        avgs = []
        worsts = []
        # Dopóki nie osiągniemy maksymalnej liczby iteracji
        while generation < self.max_iter:
            # Selekcja
            selected_population = self.selection_method(population)

            # Generowanie nowej populacji przez krzyżowanie i mutację
            new_population = self.new_generation(selected_population, population)

            for solution in new_population:
                solution.evaluate_function()

            # Aktualizacja populacji
            population = new_population

            # Znajdź najlepsze rozwiązanie w generacji
            best_solution = max(population, key=lambda x: x.fitness)
            bests.append(best_solution.fitness)
            
            # Najgorsze
            worst_solution = min(population, key=lambda x: x.fitness)
            worsts.append(worst_solution.fitness)
             
            # Średnie
            avg_solution = np.mean([solution.fitness for solution in population])
            avgs.append(avg_solution)
            
            print("Generation: {:d}, Best fitness: {:.2f}, Worst fitness: {:.2f}, Avg fitness: {:.2f}".format(generation, best_solution.fitness, worst_solution.fitness, avg_solution))

            generation += 1

        return np.array(bests), np.array(avgs), np.array(worsts)
    

    def generate_first_population(self, solution_size, generation_size):
        '''Generowanie generation_size rozwiązań dla podanego problemu.'''
        population = []
        
        for _ in range(generation_size):
            vector = []
            for _ in range(solution_size):
                x = random.randint(0, self.problem.size[0]-1)
                y = random.randint(0, self.problem.size[1]-1)
                
                while (x, y) in vector:
                    x = random.randint(0, self.problem.size[0]-1)
                    y = random.randint(0, self.problem.size[1]-1)
                vector.append((x, y))
                
            sol = Solution(vector=vector, problem=self.problem)
            while not sol.is_legal():
                sol.mutation()
            population.append(sol)
                
        return population
    
    def new_generation(self, selected_population: list, old_population: list) -> list:
        """
        ----------
        #### Parametry
        ----------
        self
        selected_population: list - populacja otrzymana z procesu selekcji
        """
        new_population = []

        # Proces krzyżowania
        if self.leave_parents:
            new_population = selected_population

            while len(new_population) < self.population_size:
                parent_1, parent_2 = random.sample(old_population, 2)
 
                child = self.crossover(parent_1, parent_2)

                new_population.append(child)
        else:
            pass
        
        # Proces mutacji
        to_mutate = random.choices(new_population, k=int(self.mutation_probability * self.population_size))
        for solution in to_mutate:
            solution.mutation()

        return new_population


    def crossover(self, parent_1, parent_2):
        """
        #### Funkcja krzyzująca
        ----------
        """
        
        
        # operation = random.choices(self.crossover_types, weights=self.crossover_weights, k=100)[0]
        # Czy nie powinniśmy jednak wybrać tylko jednej operacji krzyżowania?
        # Tak jak poniżej
        
        operation = random.choices(self.crossover_types, weights=self.crossover_weights, k=1)[0]
        
        return(operation(parent_1, parent_2))
    
    def selection_method(self, population):
        """
        #### Funkcja selekcji
        ----------
        """
        
        # operation = random.choices(self.selection_types, weights=self.selection_weights, k=100)[0]
        # Czy nie powinniśmy jednak wybrać tylko jednej operacji selekcji?
        # Tak jak poniżej
        
        operation = random.choices(self.selection_types, weights=self.selection_weights, k=1)[0]
        
        return(operation(population))

