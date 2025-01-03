import numpy as np
import random
from collections import defaultdict
from typing import Union

class Tile():
    '''
    Klasa reprezentująca część lasu, inaczej: pole w macierzy problemu.
    ----------
    Parametry
    ---------- 
    self.transport_cost: - koszt przetransportowania ściętego drewna do bazy ponoszony jednorazowo,
    self.workers_required: - liczba pracowników potrzebna by ściąć wszystkie drzewa w punkcie,
    self.reward: - cena całego drewna zebranego z pola.
    '''
    def __init__(self, transport_cost: float, workers_required: int, reward: float):
        self.transport_cost = transport_cost
        self.workers_required = workers_required
        self.reward = reward
        # Jak przeszkadza pogoda
        self.weather_affection = random.choice([1, 2, 4, 5])
 
class Problem():
    '''
    Klasa reprezentująca problem.
    ----------
    Parametry
    ---------- 
    self.matrix:  macierz, której każde pole jest instancją klasy Tile,
    self.n:  liczba etapów problemu
    self.bad_weather_prob:  rozkład prawdopodobieństwa wystąpienia złej pogody
    self.penalty:  wartość dodatkowego kosztu spowodowanego warunkami pogodowymi
    self.our_workers:  liczba "naszych" pracowników,
    self.a_workers:  liczba pracowników z zewnątrz klasy A,
    self.b_workers:  liczba pracowników z zewnątrz klasy B,
    self.wage:  wypłata na dzień, jaką płacimy "naszemu" pracownikowi,
    self.a_wage:  wypłata na dzień, jaką płacimy pracownikom z zewnątrz klasy A,
    self.b_wage: wypłata na dzień, jaką płacimy pracownikom z zewnątrz klasy B,
    self.mutation_functions: lista funkcji mutacji,
    self.mutation_probs: lista prawdopodobieństw wyboru funkcji mutacji.
    '''
    def __init__(self, mutation_functions: list, mutation_probs: list, basic_mutation: callable, mutate_to_legal: callable, size: tuple[int] = (30, 30), n: int = 40, wage: float = 600, our_workers: int = 50, weather_prob: callable = lambda i: i * 1/20, penalty: float = 5000):

        self.n = n
        self.size = size
        self.weather_prob = weather_prob
        self.penalty = penalty
        self.our_workers = our_workers
        self.wage = wage
        self.a_wage = wage*1.25
        self.b_wage = wage*1.6
        self.a_workers = int(our_workers * 0.3)
        self.b_workers = int(our_workers * 0.2)
        self.mutation_functions = mutation_functions
        self.mutation_probs = mutation_probs
        self.basic_mutation = basic_mutation
        self.mutate_to_legal = mutate_to_legal

        self.matrix = []

        for i in range(size[0]):
            self.matrix.append([])
            for j in range(size[1]):
                workers_required = random.choice(range(our_workers, int(our_workers*1.5)))
                reward = workers_required * random.choice(range(600, 750, 10))
                # Koszt transportu jest proporcjonalny do odległości pola od lewego górnego rogu macierzy lasu.
                self.matrix[i].append(Tile(transport_cost=50*(i+j+1), workers_required=workers_required, reward=reward))
            
            
  
class Solution():
    '''
    Klasa reprezentująca rozwiązanie.
    ----------
    Parametry
    ---------- 
    self.vector: wektor z rozwiązaniami,
    self.problem: problem który rozwiązujemy
    '''   
    def __init__(self, vector: list[tuple], problem: 'Problem'):
        self.size = len(vector)
        self.problem = problem
        self.vector = vector 
        self.fitness = self.evaluate_function()
        # self.crossover_functions, self.crossover_probs = map(list, zip(*crossover_strategies))
        # self.mutation_functions, self.mutation_probs = map(list, zip(*mutation_strategies))

    def mutation(self):
        self._perform_mutation()

    def _perform_mutation(self):
        mutation_function = random.choices(self.problem.mutation_functions, weights=self.problem.mutation_probs, k=1)[0]
        mutation_function(self)
        
        counter = 0
        while not self.is_legal():
            if counter < 50:
                self.problem.basic_mutation(self)
            else:
                self.problem.mutate_to_legal(self)


    # def crossover(self, solution2: 'Solution' ) -> 'Solution':
    #     """
    #     # TODO Wykorzystanie funkcji z genetic_functions
    #     # TODO Wybór z dostepnych metod krzyżowania (losowy/deterministyczny) 
    #     """
    #     self._perform_crossover(solution2)

    # def _perform_crossover(self, solution2: 'Solution') -> 'Solution':
    #     """
    #     # TODO Jak przekazywać wybór dodatkowych parametrów do funkcji krzyzowania np. single_point_crossover
    #     # może otrzymywać 'strone' z której dzielimy  
    #     """
    #     crossover_function = random.choice(self.crossover_functions, 1, p=self.crossover_probs)
    #     child = crossover_function(self, solution2)
    #     while not child.is_legal():
    #         child.mutation()
    #     return child

    def evaluate_function(self):
        j = 0
        for xy, i in zip(self.vector, range(len(self.vector))):
            x, y = xy
            xy_tile = self.problem.matrix[x][y]
            # Ilu wykorzystamy naszych pracowników
            our_workers = min(self.problem.our_workers, xy_tile.workers_required)
            # Ilu wykorzystamy pracowników klasy A
            a_workers = min(xy_tile.workers_required % self.problem.our_workers, self.problem.a_workers) 
            # Ilu wykorzystamy pracowników klasy B
            b_workers = xy_tile.workers_required % (self.problem.our_workers + a_workers)
            
            # Koszt poniesiony przy wypłatach
            paid_wages = our_workers*self.problem.wage + a_workers*self.problem.a_wage + b_workers*self.problem.b_wage
            
            # Koszty poniesione z powodu warunków pogodowych 
            w_cost = self.problem.weather_prob(i)*(self.problem.penalty+xy_tile.weather_affection)
            
            # Ostatecznie wyliczamy wartość funkcji celu
            j += xy_tile.reward - xy_tile.transport_cost - w_cost - paid_wages
            
        self.fitness = j
        return j
    
    
    def is_legal(self):
        """
        Metoda sprawdzająca czy rozwiązanie jest legalne - tj. czy spełnia warunki
        Naszym warunkiem jest: maksymalnie 2 sąsiadujące ze sobą pola (pionowo lub poziomo) mogą zostać ścięte.
        """

        rows = defaultdict(list)
        cols = defaultdict(list)
        
        for xy in self.vector:
            rows[xy[0]].append(xy[1])
            cols[xy[1]].append(xy[0])
            
        for x, ys in rows.items():
            if len(ys) > 2:
                if self._check_lines(ys):
                    return False
                
        for y, xs in cols.items():
            if len(xs) > 2:
                if self._check_lines(xs):
                    return False
                
        return True

    def _check_lines(self, ls: list[int]) -> bool:
        """
        Funkcja sprawdzająca, czy w jednej lini znajdują się 3 sąsaidujące ze sobą pola
        """

        ls.sort() 
        counter = 0
        for i in range(1, len(ls)):
            if ls[i] == ls[i-1] + 1:
                counter += 1
                if counter == 2:
                    return True
            else:
                counter = 0
        return False