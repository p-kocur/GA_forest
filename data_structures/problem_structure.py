import numpy as np
import random
from collections import defaultdict

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
        # Jak przeszkadza pogoda
        self.weather_affection = random.choice([1, 2, 4, 5])

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
        self.size = size
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
                reward = workers_required * random.choice(range(100, 150, 5)) / 100
                # Koszt transportu jest proporcjonalny do odległości pola od lewego górnego rogu macierzy lasu.
                self.matrix[i].append(Tile(transport_cost=50*(i+j+1), workers_required=workers_required, reward=reward))
            
'''
Klasa reprezentująca rozwiązanie.
'''     
class Solution():
    def __init__(self, vector: list[tuple], problem: 'Problem'):
        self.size = len(vector)
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
            
        return j
    
    # Metoda sprawdzająca czy rozwiązanie jest legalne - tj. czy spełnia warunki
    # Naszym warunkiem jest: maksymalnie 2 sąsiadujące ze sobą pola (pionowo lub poziomo) mogą zostać ścięte.
    def is_legal(self):
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

    # Funkcja sprawdzająca, czy w jednej lini znajdują się 3 sąsaidujące ze sobą pola
    def _check_lines(self, ls: list[int]) -> bool:
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