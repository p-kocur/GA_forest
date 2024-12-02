import numpy as np
import random

from data_structures.problem_structure import Solution, Problem, Tile

def basic_mutation(sol: 'Solution') -> None:
    # Losowo wybieramy element rozwiązania
    i = random.choice(range(sol.problem.n))
    
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
        
def permutation_mutation(sol: 'Solution') -> None:
    # Losowo wybieramy element rozwiązania
    i = random.choice(range(sol.problem.n))
    
    # Zamieniamy wylosowany element z elementem z kolejnego etapu
    # (Lub z etapu poprzedniego w szczególnym wypadku)
    if i == sol.problem.n - 1:
        i_2 = i - 1
    else:
        i_2 = i + 1  
    sol.vector[i], sol.vector[i_2] = sol.vector[i_2], sol.vector[i]
        

def single_point_crossover(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Wykonuje krzyżowanie jednopunktowe między dwoma rodzicami.
    --------------------
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
        print("Uwaga! Nielegalne dziecko") # xD

    return child



def multi_point_crossover(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Wykonuje krzyżowanie wielopunktowe między dwoma rodzicami.
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

    return child


def uniform_crossover(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    Wykonuje krzyżowanie jednolite
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

    return child