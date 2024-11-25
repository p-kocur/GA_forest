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
self.our_workers - liczba "naszych" pracowników,
self.a_workers - liczba pracowników z zewnątrz klasy A,
self.b_workers - liczba pracowników z zewnątrz klasy B,
self.wage - wypłata na dzień, jaką płacimy "naszemu" pracownikowi,
self.a_wage - wypłata na dzień, jaką płacimy pracownikom z zewnątrz klasy A,
self.b_wage - wypłata na dzień, jaką płacimy pracownikom z zewnątrz klasy B,
'''
class Problem():
    def __init__(self, size: tuple[int], wage: float, our_workers: int):
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
                self.matrix[i].append(Tile(transport_cost=50*(i+j+1), workers_required=workers_required, reward=reward))
            
        

class Solution():
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.fitness = self.evaluate_function()

    def mutation(self):
        self._perform_mutation()
        self.fitness = self.evaluate_function()

    def _perform_mutation(self):
        pass

    def crossover(self, solution2: 'Solution' ) -> 'Solution':
        pass

    def evaluate_function(self):
        pass