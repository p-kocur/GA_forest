import random
import numpy as np

from data_structures.problem_structure import Problem, Solution, Tile
from algorithm.genetic_functions import (
    naive_crossover,
    single_point_crossover_vector,
    multi_point_crossover_vector,
    elitism_selection,
    roulette_selection,
    tournament_selection,
    basic_mutation,
    territorial_mutation,
    permutation_mutation,
    max_reward_mutation,
    expansion_mutation,
    mutate_to_legal
)

# Nie wiem jak inaczej zrobić to tak żeby z main przekazywać liste funkcji

function_map = {
    'naive_crossover': naive_crossover,
    'single_point_crossover_vector': single_point_crossover_vector,
    'multi_point_crossover_vector': multi_point_crossover_vector,
    'elitism_selection': elitism_selection,
    'roulette_selection': roulette_selection,
    'tournament_selection': tournament_selection,
    'basic_mutation': basic_mutation,
    'territorial_mutation': territorial_mutation,
    'permutation_mutation': permutation_mutation,
    'max_reward_mutation': max_reward_mutation,
    'expansion_mutation': expansion_mutation,
    'mutate_to_legal': mutate_to_legal
}

def get_function_list(function_names, function_map):
    returns = [function_map[name] for name in function_names if name in function_map]
    return returns

class GeneticAlgorithm:
    """
    #### Klasa główna algorytmu genetycznego.
    ----------
    #### Parametry
    ----------
    crossover_types: list = None, lista dostepnych metod krzyżowania
    selection_types: list = None, lista dostepnych metod selekcji
    mutation_types: list = None, lista dostepnych metod mutacji
    mutation_probability: float = 0.1, prawdopopobienstwo wystapienia mutacji
    first_population: list = None, przekazanie startowej populacji, None: losowa generacja pierwszej generacji
    population_size: int = 100, wielkosc generacji (populacji), tj. ilość Solution
    max_iter: int = 1000, maksymalna ilosc iteracji
    problem: Problem = None, problem który rozwiazujemy
    leave_parents: bool = True, jezeli True wszystkie selekcje działają na zasadzie elity tj. populacja pozostała po selekcji jest przekazywana do kolejnej generacji  
    elite_percentage: int = 10, procent pozostawianych rozwiązań
    interrupt_flag: callable = None, używane do przerwania algorytmu w związku z działaniem TKinter
    
    Schemat rozwiązania:
    """

    def __init__(self,
                crossover_types: list = None, #
                selection_types: list = None, #
                mutation_types: list = None,
                mutation_probability: float = 0.1,
                first_population: list = None, 
                population_size: int = 1000,
                max_iter: int = 1000,
                problem: Problem = None,
                leave_parents: bool = True,
                elite_percentage: int = 10,
                interrupt_flag= False):
        
        if not crossover_types:
            self.crossover_types = [naive_crossover]
        else:
            self.crossover_types = get_function_list(crossover_types, function_map)
            
        if not selection_types:
            self.selection_types = [elitism_selection]
        else:
            self.selection_types = get_function_list(selection_types, function_map)
        
        if not mutation_types:
            self.mutation_types = [basic_mutation]
        else:            
            self.mutation_types = get_function_list(mutation_types, function_map)
  
        if problem is None:
            self.problem = Problem(self.mutation_types, basic_mutation=basic_mutation, mutate_to_legal=mutate_to_legal)
        else:
            self.problem = problem

        if mutation_probability > 1:
            mutation_probability = 1
  
        self.mutation_probability = mutation_probability
        self.first_population = first_population 
        self.population_size = population_size
        self.max_iter = max_iter
        self.leave_parents = leave_parents
        self.elite_percentage = elite_percentage
        self.interrupt_flag = interrupt_flag

    def run(self):
        global should_interrupt

        if not self.first_population:
            self.population = self.generate_first_population(self.problem.n, self.population_size)
        else:
            self.population = self.first_population
        
        # Aktualizujemy "wartość" każdego rozwiązania w populacji
        for solution in self.population:
            solution.evaluate_function()

        generation = 0

        # Tworzymy listy, w których będziemy zapisywać pośrednie wyniki w trakcie pracy algorytmu
        bests = []
        avgs = []
        worsts = []


        # Dopóki nie osiągniemy maksymalnej liczby iteracji
        while generation < self.max_iter:

            if self.interrupt_flag and self.interrupt_flag():
                print(f"Algorytm zatrzymany na {generation} generacji.")
                break

            # Selekcja
            selected_population = self.selection_method()

            # Generowanie nowej populacji przez krzyżowanie i mutację
            new_population = self.new_generation(selected_population, self.population)

            # Aktualizujemy "wartość" każdego rozwiązania w populacji
            for solution in new_population:
                solution.evaluate_function()

            # Aktualizacja populacji
            self.population = new_population

            # Znajdź najlepsze rozwiązanie w generacji
            best_solution = max(self.population, key=lambda x: x.fitness)
            bests.append(best_solution.fitness)
            
            # Najgorsze
            worst_solution = min(self.population, key=lambda x: x.fitness)
            worsts.append(worst_solution.fitness)
             
            # Średnie
            avg_solution = np.mean([solution.fitness for solution in self.population])
            avgs.append(avg_solution)

            potential_solution = max(self.population, key=lambda x: x.fitness)
            if best_solution is None or potential_solution.fitness > best_solution.fitness:
                best_solution = potential_solution

            print("Generation: {:d}, Best fitness: {:.2f}, Worst fitness: {:.2f}, Avg fitness: {:.2f}".format(generation, best_solution.fitness, worst_solution.fitness, avg_solution))

            generation += 1


        return best_solution, np.array(bests), np.array(avgs), np.array(worsts)
    

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
            
            # Mutujemy dopóki rozwiązanie nie będzie legalne
            while not sol.is_legal():
                sol.problem.basic_mutation(sol)
                
            # Następnie dodajemy je do populacji
            population.append(sol)
                
        return population
    
    def new_generation(self, selected_population: list, old_population: list) -> list:
        """
        #### Tworzenie nowej populacji na podstawie rozwiązań wybranych w procesie selekcji i całości poprzedniej populacji
        ----------
        #### Parametry
        ----------
        self
        selected_population: list - populacja otrzymana z procesu selekcji
        old_population: list - populacja poprzednia
        """
        new_population = []

        # Proces krzyżowania
        if self.leave_parents:
            new_population = selected_population.copy()
        while len(new_population) < self.population_size:
            parent_1, parent_2 = random.sample(old_population, 2)
            child = self.crossover(parent_1, parent_2)
            new_population.append(child)
        
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
        operation = random.choices(self.crossover_types, k=1)[0]
        return(operation(parent_1, parent_2))
    
    def selection_method(self):
        """
        #### Funkcja selekcji
        ----------
        """
        operation = random.choices(self.selection_types, k=1)[0]
        return(operation(self.population, self.elite_percentage))

