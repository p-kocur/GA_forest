import numpy as np
import random

from data_structures.problem_structure import Solution, Problem, Tile

'''
Zamienia element rozwiązania z losowym elementem macierzy problemu
'''
def basic_mutation(sol: 'Solution', index=None) -> None:
    # Losowo wybieramy element rozwiązania chyba, że przekazano index
    if index is None:
        i = random.choice(range(sol.problem.n))
    else:
        i = index
    
    # Losowo wybieramy nowe koordynaty wylosowanego elementu rozwiązania
    possible_x = range(sol.problem.size[0])
    possible_y = range(sol.problem.size[1])
    
    # Dopóki nie odnaleźliśmy niezajętego pola
    while True:
        x = random.choice(possible_x)
        y = random.choice(possible_y)
        if (x, y) not in sol.vector:
            sol.vector[i] = (x, y)
            return

'''
Zamienia element rozwiązania na element sąsiadujący z nim w macierzy
'''         
def territorial_mutation(sol: 'Solution') -> None:
    # Dopóki nie odnaleźliśmy niezajętego pola
    while True:
        # Losowo wybieramy element rozwiązania
        i = random.choice(range(sol.problem.n))
        
        # Wybieramy nowe pole macierzy sąsiadujące z poprzednim, pamiętając o ograniczeniach
        new_x = min(sol.problem.size[0], max(0, sol.vector[i][0] + random.choice([-1, 0, 1])))
        new_y = min(sol.problem.size[1], max(0, sol.vector[i][1] + random.choice([-1, 0, 1])))
        
        if (new_x, new_y) not in sol.vector:
            sol.vector[i] = (new_x, new_y)
            return

'''
Zamienia element rozwiązania z etapu i z rozwiązaniem z etapu i+1 (lub i-1)
'''  
def permutation_mutation(sol: 'Solution') -> None:
    # Losowo wybieramy element rozwiązania
    i = random.choice(range(sol.problem.n))
    
    # Zamieniamy wylosowany element z elementem z kolejnego etapu
    # (Lub z etapu poprzedniego w szczególnym przypadku).
    if i == sol.problem.n - 1:
        i_2 = i - 1
    else:
        i_2 = i + 1  
    sol.vector[i], sol.vector[i_2] = sol.vector[i_2], sol.vector[i]
  
'''
Odrzuca element rozwiązania o najmniejszej różnicy wartości nagody z kosztem transportu
'''  
def max_reward_mutation(sol: 'Solution') -> None:
    min_reward = np.inf
    min_element_i = None
    for i in range(len(sol.vector)):
        x, y = sol.vector[i]
        value = sol.problem.matrix[x][y].reward - sol.problem.matrix[x][y].transport_cost
        if value < min_reward:
            min_reward, min_element_i = value, i
    
    basic_mutation(sol, min_element_i)
    
'''
Zamienia element rozwiązania z elementem macierzy lężącym w obszarze, gdzie nie leży żaden element rozwiązania.
Znajduje pole macierzy najbardziej oddalone od obecnych rozwiązań i dodaje je do obecnego wektora rozwiązań.
'''
def expansion_mutation(sol: 'Solution') -> None:
    xs, ys = map(list, zip(*sol.vector))
    xs.extend([0, sol.problem.size[0]-1])
    ys.extend([0, sol.problem.size[1]-1])
    xs.sort()
    ys.sort()
    
    max_d_x = 0
    new_x = None
    for i in range(len(xs)-1):
       d_x = xs[i+1] - xs[i] 
       if d_x > max_d_x:
           max_d_x, new_x = d_x, xs[i] + (xs[i+1] - xs[i])//2
           
    max_d_y = 0
    new_y = None
    for i in range(len(ys)-1):
       d_y = ys[i+1] - ys[i] 
       if d_y > max_d_y:
           max_d_y, new_y = d_y, ys[i] + (ys[i+1] - ys[i])//2
           
    
    if (new_x, new_y) not in sol.vector:
        i = random.choice(range(sol.problem.n))
        sol.vector[i] = (new_x, new_y)



# Deterministyczne krzyżowania
# Kilka krzyżowań z prawdopodobieństwem
# Dodanie do klasy solution
# Dobór osobników do krzyżowania
# Zastanowić się nad interfejsem

def single_point_crossover(parent_1: 'Solution', parent_2: 'Solution', side: str = 'l') -> 'Solution':
    """
    Wykonuje krzyżowanie jednopunktowe na reprezentacji macierzowej rodziców
    --------------------
    Parameters
    ---------- 
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    side: str, 'l' - macierz dzielona od: 'l' lewej, 'p' prawej, 'g' od góry, 'd' od dołu
    return: Solution, child

    Decyduje fitnes rodziców.
    """
    # Zamień rozwiązanie z reprezetnacji wektorowej na reprezentacje macierzową
    parent_1_matrix = np.zeros([parent_1.size[0], parent_1.size[1]])
    parent_1_matrix[parent_1] = 1

    parent_2_matrix = np.zeros([parent_2.size[0], parent_2.size[1]])
    parent_2_matrix[parent_2] = 1

    # Przypisanie takich samych zmiennych jak rodzic_1 dla dziecka
    child = parent_1

    child_matrix = np.zeros([parent_1.size[0],parent_2.size[1]])

    # Punkt podziału na podstawie wartosci fitnes rodziców 
    split_point = max(parent_1.fitness, parent_2.fitness)/(parent_1+parent_2)

    if parent_1.fitness > parent_2.fitness:
        child_matrix = matrix_crossover(parent_1_matrix, parent_2_matrix, split_point, side)
    else:
        child_matrix = matrix_crossover(parent_2_matrix, parent_1_matrix, split_point, side)

    child.vector = np.transpose(np.where(child_matrix))

    return child

def single_point_crossover_random(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Wykonuje losowe krzyżowanie jednopunktowe na reprezentacji macierzowej rodziców
    --------------------
    Parameters
    ---------- 
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    return: Solution, child

    Decyduje losowo.
    """

    # Zamień rozwiązanie z reprezetnacji wektorowej na reprezentacje macierzową
    parent_1_matrix = np.zeros([parent_1.size[0], parent_1.size[1]])
    parent_1_matrix[parent_1] = 1

    parent_2_matrix = np.zeros([parent_2.size[0], parent_2.size[1]])
    parent_2_matrix[parent_2] = 1

    # Przypisanie takich samych zmiennych jak rodzic_1 dla dziecka
    child = parent_1

    child_matrix = np.zeros([parent_1.size[0],parent_2.size[1]])

    # Punkt podziału losowo 
    split_point = np.random.randint(0, parent_1.size[0])
    sides = ['l', 'p', 'd', 'g']

    if parent_1.fitness > parent_2.fitness:
        child_matrix = matrix_crossover(parent_1_matrix, parent_2_matrix, split_point, random.choice(sides))
    else:
        child_matrix = matrix_crossover(parent_2_matrix, parent_1_matrix, split_point, random.choice(sides))

    child.vector = np.transpose(np.where(child_matrix))

    return child


def matrix_crossover(m1: np.array, m2: np.array, split_point, side:str = 'l') -> np.array:
    """
    Funkcja przeprowadzająca krzyżowanie jednopunktowe
    """
    m = np.zeros_like(m1)
    
    if side == 'l':
        m[:split_point][:] = m1[:split_point][:] 
        m[split_point:][:] = m2[split_point:][:]
        return m
    
    if side == 'p':
        m[split_point:][:] = m1[split_point:][:] 
        m[:split_point][:] = m2[:split_point][:] 
        return m

    if side == 'g':
        m[:][:split_point] = m1[:][:split_point] 
        m[:][split_point:] = m2[:][split_point:] 
        return m

    if side == 'd':
        m[:][split_point:] = m1[:][split_point:] 
        m[:][:split_point] = m2[:][:split_point] 
        return m

    ValueError("Wrong side argument")




def single_point_crossover_naive(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Naiwne krzyżowanie jednopunktowe - operuje na wektorze solution
    ----------
    Parameters
    ---------- 
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    return: Solution, child

    Schemat rozwiązania:
    Wybierz losowy indeks jako punkt krzyżowania, rozdziel w tym punkcie 
    obydwojga rodziców, połącz segmenty tworząc dziecko.
    """
    # TODO - sprawdzenie czy rodzice mogą się krzyżować

    crossover_point = np.random.randint(1, parent_1.size) 
    child_vector = parent_1.vector[:crossover_point] + parent_2.vector[crossover_point:]
    child = Solution(vector=child_vector, problem=parent_1.problem)

    # TODO - Zapewnienie czy dziecko jest legalne, rozwiązanie na teraz: dziecko nielegalne -> przeprowadzamy mutacje 
    max_retries = 10
    retries = 0

    while not child.is_legal() and retries < max_retries:
        child._perform_mutation()
        retries += 1

    if not child.is_legal():
        # raise TimeoutError # Mozliwa alternatywa zamiast print'a
        print("Uwaga! Nielegalne dziecko")
        child = parent_1


    return child



def multi_point_crossover_naive(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Wykonuje naiwne krzyżowanie wielopunktowe między dwoma rodzicami - operuje na wektorze solution.
    --------------------
    :param n_points - liczba punktów krzyżowania (domyślnie 2)
    """
    # TODO - sprawdzenie czy rodzice mogą się krzyżować
    # TODO - zmiana w solution tak aby ilość punktów mogła być różna od 2  
    
    n_points = 2

    crossover_points = sorted(np.random.choice(range(1, len(parent_1.size)), size=n_points, replace=False))

    # Krzyżowanie genomu naprzemienie między rodzicami
    # TODO -  początek zawsze z parent_1

    child_vector = []
    current_parent = parent_1.vector
    last_point = 0

    for point in crossover_points:
        child_vector.extend(current_parent[last_point:point])
        current_parent = parent_2.vector if current_parent == parent_1.vector else parent_1.vector
        last_point = point

    child_vector.extend(current_parent[last_point:])
    
    child = Solution(vector=child_vector, problem=parent_1.problem)

    # TODO - Zapewnienie czy dziecko jest legalne, rozwiązanie na teraz: dziecko nielegalne -> przeprowadzamy mutacje 
    max_retries = 10
    retries = 0

    while not child.is_legal() and retries < max_retries:
        child._perform_mutation()
        retries += 1

    if not child.is_legal():
        print("Uwaga! Nielegalne dziecko") 
        child = parent_1

    return child


def uniform_crossover_naive(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Wykonuje naiwne krzyżowanie jednolite - operuje na wektorze solution
    --------------------
    Dla każdego genu, wybiera losowo z parent_1 lub parent_2
    """

    vector_1 = parent_1.vector
    vector_2 = parent_2.vector
    child_vector = []

    child_vector = [
        vector_1[i] if np.random.rand() < 0.5 else vector_2[i]
        for i in range(len(vector_1))
    ]

    child = Solution(vector=child_vector, problem=parent_1.problem)

    # TODO - Zapewnienie czy dziecko jest legalne, rozwiązanie na teraz: dziecko nielegalne -> przeprowadzamy mutacje 
    max_retries = 10
    retries = 0

    while not child.is_legal() and retries < max_retries:
        child._perform_mutation()
        retries += 1

    if not child.is_legal():
        print("Uwaga! Nielegalne dziecko")
        child = parent_1


    return child



