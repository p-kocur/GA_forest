import numpy as np
import random

'''
Klasa reprezentująca część lasu, inaczej: pole w macierzy problemu.
Jej pola to:
self.transport_cost - koszt przetransportowania ściętego drewna do bazy ponoszony jednorazowo,
self.workers_required - liczba pracowników potrzebna by ściąć wszystkie drzewa w punkcie,
self.reward - cena całego drewna zebranego z pola.
'''
class Tile():
    def __init__(self, transport_cost: float, workers_required: int, reward: float):
        self.transport_cost = transport_cost
        self.workers_required = workers_required
        self.reward = reward

'''
Klasa reprezentująca problem.
Jej pola to:
self.matrix - macierz, której każde pole jest instancją klasy Tile,
self.n - liczba etapów problemu
self.bad_weather_prob - rozkład prawdopodobieństwa wystąpienia złej pogody
self.penalty - wartość dodatkowego kosztu spowodowanego warunkami pogodowymi
self.our_workers - liczba "naszych" pracowników,
self.a_workers - liczba pracowników z zewnątrz klasy A,
self.b_workers - liczba pracowników z zewnątrz klasy B,
self.wage - wypłata na dzień, jaką płacimy "naszemu" pracownikowi,
self.a_wage - wypłata na dzień, jaką płacimy pracownikom z zewnątrz klasy A,
self.b_wage - wypłata na dzień, jaką płacimy pracownikom z zewnątrz klasy B,
'''
class Problem():
    def __init__(self, size: tuple[int], n: int, wage: float, our_workers: int, weather_prob: callable, penalty: float):
        self.n = n
        self.weather_prob = weather_prob
        self.penalty = penalty
        self.our_workers = our_workers
        self.wage = wage
        self.a_wage = wage*1.25
        self.b_wage = wage*1.6
        self.a_workers = int(our_workers * 0.3)
        self.b_workers = int(our_workers * 0.2)
        self.matrix = []
        for i in range(size[0]):
            self.matrix.append([])
            for j in range(size[1]):
                workers_required = random.choice(range(our_workers, int(our_workers*1.5)))
                reward = workers_required * random.choice(range(1, 1.5, 0.05)) 
                # Koszt transportu jest proporcjonalny do odległości pola od lewego górnego rogu macierzy lasu.
                self.matrix[i].append(Tile(transport_cost=50*(i+j+1), workers_required=workers_required, reward=reward))
            
'''
Klasa reprezentująca rozwiązanie.
'''     
class Solution():
    def __init__(self, size: int, vector: list[tuple], problem: 'Problem'):
        self.generation_size = len(vector)
        self.problem = problem
        self.vector = vector 
        self.fitness = self.evaluate_function()

    def mutation(self):
        self._perform_mutation()
        self.fitness = self.evaluate_function()

    def _perform_mutation(self):
        '''
        Choose two random indices and swap their respective values
        '''
        
        source_index = (np.random.randint(0, self.size))

        while True:
            target_index = (np.random.randint(0, self.size))
            if target_index != source_index:
                break

        self.matrix[source_index], self.matrix[target_index] = (
            self.matrix[target_index],
            self.matrix[source_index],
        )

    def crossover(self, solution2: 'Solution' ) -> 'Solution':
        pass

    def evaluate_function(self):
        j = 0
        for x, y, i in zip(self.vector, range(len(self.vector))):
            xy_tile = self.problem[x][y]
            
            j += xy_tile.reward - xy_tile.transport_cost - self.problem.weather_prob(i)*self.problem.penalty - 