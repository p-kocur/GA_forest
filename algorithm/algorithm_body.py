import random
import numpy as np

from data_structures.problem_structure import Problem, Solution, Tile
from genetic_functions import (
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
                mutation_probability: list = None,
                first_population: list = None, 
                population_size: int = 100,
                max_iter: int = 100,
                problem: Problem = None,
                leave_parents: bool = True):
        

        # listy crossover_probabilities oraz selection_proability
        # jest tylko po to że taka forma jest bardziej czytelna 
        # niz cztery odzielne listy faktycznie wykorzystywane do algorytmu  

        if not crossover_probabilities:
            crossover_probabilities = [
            {"crossover": single_point_crossover_random, "probability": 0.80},
            {"crossover": single_point_matrix_crossover, "probability": 0.60},
            {"crossover": single_point_crossover_vector, "probability": 0.75},
            {"crossover": multi_point_crossover_vector, "probability": 0.65},
            {"crossover": uniform_crossover_naive, "probability": 0.50}
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
  
        self.mutation_probability = mutation_probability
        self.first_population = first_population 
        self.population_size = population_size
        self.max_iter = max_iter
        self.problem = problem 
        self.leave_parents = leave_parents

    def run(self):
        if not self.first_population:
            population = self.generate_first_population()
        
        for solution in population:
            solution.evaluate_function()

        generation = 0

        while generation < self.max_iter:
            # Selekcja
            selected_population = self.selection_method(population)

            # Generowanie nowej populacji przez krzyżowanie i mutację
            new_population = self.new_generation(self, selected_population)

            for solution in new_population:
                solution.evaluate_function()

            # Aktualizacja populacji
            population = new_population

            # Znajdź najlepsze rozwiązanie w generacji
            best_solution = min(population, key=lambda x: x.fitness)
            
            print(f"Generation: {generation}, Best fitness: {best_solution.fitness}")

            generation += 1

        return min(population, key=lambda x: x.fitness)
    

    def generate_first_population(self, n, generation_size):
        '''Generowanie n rozwiązań dla podanego problemu.'''
        #### TODO Lepsze rozwiązanie niż teraz
        return [Solution(vector=[(random.randint(0, self.problem.size[0]-1), random.randint(0, self.problem.size[1]-1)) 
                                for _ in range(self.problem.n)], 
                        problem=self.problem) 
                for _ in range(n)]
    
    def new_generation(self, selected_population: list) -> list:
        """
        #### TODO Rozwiązanie uwzględniające fitness rodziców, teraz losuje z populacji
        #### Funkcja tworząca nową generacje (populacje)
        ----------
        #### Parametry
        ----------
        self
        selected_population: list - populacja otrzymana z procesu selekcji
        """
        new_population = []

        if self.leave_parents:
            new_population = selected_population

            while len(new_population) < self.population_size:
                parent_1, parent_2 = random.sample(selected_population, 2)

                child = self.crossover(parent_1, parent_2)

                new_population.append(child)

        else:
            pass

        return new_population


    def crossover(self, parent_1, parent_2):
        """
        #### Funkcja krzyzująca
        ----------
        """
        operation = random.choices(self.crossover_types, weights=self.crossover_weights, k=100)[0]
        
        return(operation(parent_1, parent_2))

