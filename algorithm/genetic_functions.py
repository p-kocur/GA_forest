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


"""
------------------------
Krzyżowanie
------------------------
"""

def single_point_crossover_random(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    #### Krzyżowanie jednopunktowe na reprezentacji macierzowej rodziców
    ----------
    #### Parametry
    ----------
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    return: Solution, dziecko

    Schemat rozwiązania:
    Losowo wybieramy punkt krzyżowania i dzielimy oba rodzicielskie rozwiązania w tym punkcie,
    tworząc dziecko przez połączenie segmentów macierzy z obu rodziców.
    """
    problem_size = parent_1.problem.size

    parent_1_matrix = np.zeros(problem_size)
    for coord in parent_1.vector:
        parent_1_matrix[coord] = 1

    parent_2_matrix = np.zeros(problem_size)
    for coord in parent_2.vector:
        parent_2_matrix[coord] = 1

    child = Solution([(0,0)], parent_1.problem)
    
    sides = ['l', 'p', 'd', 'g']
    split_point = np.random.randint(0, problem_size[0])

    if parent_1.fitness > parent_2.fitness:
        child_matrix = single_point_matrix_crossover(parent_1_matrix, parent_2_matrix, split_point, random.choice(sides))
    else:
        child_matrix = single_point_matrix_crossover(parent_2_matrix, parent_1_matrix, split_point, random.choice(sides))

    vector = np.transpose(np.nonzero(child_matrix))
    child.vector = [tuple(coord) for coord in vector]

    child = naive_legitimacy(child, parent_1, parent_2)

    child.evaluate_function()

    return child


def single_point_matrix_crossover(m1: np.array, m2: np.array, split_point, side:str = 'l') -> np.array:
    """
    #### Krzyżowanie jednopunktowe macierzy
    ----------
    #### Parametry
    ----------
    m1: np.array, pierwsza macierz rodzica
    m2: np.array, druga macierz rodzica
    split_point: int, punkt krzyżowania
    side: str, strona krzyżowania ('l' - lewa, 'p' - prawa, 'd' - dolna, 'g' - górna)
    return: np.array, macierz dziecka

    Schemat rozwiązania:
    Na podstawie wybranego punktu krzyżowania i strony, łączymy macierze rodziców w jedną nową,
    tworząc dziecko w formie macierzy.
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




def single_point_crossover_vector(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    #### Naiwne krzyżowanie jednopunktowe na wektorze solution
    ----------
    #### Parametry
    ----------
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    return: Solution, dziecko

    Schemat rozwiązania:
    Wybieramy losowy punkt krzyżowania, dzielimy wektory rodziców w tym punkcie i tworzymy dziecko 
    poprzez połączenie segmentów wektorów.
    """
    first_is_parent_1 = np.random.random() < 0.5

    smaller_size = min(len(parent_1.vector), len(parent_2.vector))
    crossover_point = np.random.randint(1, smaller_size)

    if first_is_parent_1:
        child_vector = (
            parent_1.vector[:crossover_point] +
            parent_2.vector[crossover_point:len(parent_2.vector)]
        )
    else:
        child_vector = (
            parent_2.vector[:crossover_point] +
            parent_1.vector[crossover_point:len(parent_1.vector)]
        )

    child = Solution(vector=child_vector, problem=parent_1.problem)

    child = naive_legitimacy(child, parent_1, parent_2)

    return child



def multi_point_crossover_vector(parent_1: 'Solution', parent_2: 'Solution', n_points=2) -> 'Solution':
    """
    #### Krzyżowanie wielopunktowe - operuje na wektorach rodziców
    ----------
    #### Parametry
    ----------
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    return: Solution, dziecko

    Schemat rozwiązania:
    Wybieramy losowo kilka punktów krzyżowania i dzielimy wektory rodziców w tych punktach, 
    tworząc dziecko przez połączenie segmentów z obu rodziców.
    """
    smaller_size = min(len(parent_1.vector), len(parent_2.vector))
    crossover_points = sorted(np.random.choice(range(1, smaller_size), size=n_points, replace=False))

    start_with_parent_1 = np.random.random() < 0.5

    child_vector = []
    current_parent = parent_1.vector if start_with_parent_1 else parent_2.vector
    last_point = 0

    for point in crossover_points:
        child_vector.extend(current_parent[last_point:point])
        current_parent = parent_2.vector if current_parent == parent_1.vector else parent_1.vector
        last_point = point

    child_vector.extend(current_parent[last_point:])

    child = Solution(vector=child_vector, problem=parent_1.problem)

    child = naive_legitimacy(child, parent_1, parent_2)

    return child


def uniform_crossover_naive(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    """
    #### Krzyżowanie jednostajne - operuje na wektorach rodziców
    ----------
    #### Parametry
    ----------
    parent_1: Solution, rodzic 1
    parent_2: Solution, rodzic 2
    return: Solution, dziecko

    Schemat rozwiązania:
    Każdy element wektora dziecka jest losowo wybierany z jednego z dwóch rodziców.
    """
    smaller_size = min(len(parent_1.vector), len(parent_2.vector))
    vector_1 = parent_1.vector
    vector_2 = parent_2.vector

    child_vector = [
        vector_1[i] if np.random.rand() < 0.5 else vector_2[i]
        for i in range(smaller_size)
    ]

    child = Solution(vector=child_vector, problem=parent_1.problem)
    
    child = naive_legitimacy(child, parent_1, parent_2)

    return child


"""
------------------------
Selekcja
------------------------
"""
def elitism_selection(population: list, percentage: int = 25) -> list:
    """
    #### Selekcja elitystyczna - wybór najlepszych elite_size rozwiązań z populacji
    ----------
    #### Parametry
    ----------
    population: list of Solution, populacja rozwiązań
    percentage: int, procent populacji który oznaczamy jako elita
    return: list of Solution, lista najlepszych rozwiązań

    Schemat rozwiązania:
    Sortujemy populację na podstawie wartości funkcji celu i wybieramy najlepszy procent rozwiązań.
    """
    elite_size = int(len(population)*(percentage/100))
    sorted_population = sorted(population, key=lambda x: x.fitness)
    return sorted_population[:elite_size]



def roulette_selection(population: list, percentage: int = 50) -> list:
    """
    #### Selekcja ruletkowa
    ----------
    #### Parametry
    ----------
    population: list, populacja rozwiązań
    percentage: int, procent populacji który zachowujemy
    return: Solution, wybrane rozwiązanie

    Schemat rozwiązania:
    Dla każdej osoby w populacji obliczamy prawdopodobieństwo wyboru, 
    a następnie losowo wybieramy rozwiązanie na podstawie tych prawdopodobieństw.
    """
    
    num_select = int(len(population)*(percentage/100))

    population.sort()
    next_population = []
    total_fitness = sum(solution.fitness for solution in population)

    cumulative_fitness = []
    c_fitness = 0

    for solution in population:
        c_fitness += solution.fitness
        cumulative_fitness.append(c_fitness)

    for _ in range(0, num_select):

        selection_point = random.uniform(0, total_fitness)

        for index, c_fit in enumerate(cumulative_fitness):
            if c_fit >= selection_point:
                next_population.append(population[index])
                for i in range(index+1, len(population)):
                    cumulative_fitness[i] -= population[index].fitness
                break
            
        population.pop(index)
        cumulative_fitness.pop(index)

    return next_population

def elitist_roulette_selection(population: list, elite_percentage: int = 15, percentage: int = 50) -> list:
    """
    #### Elitist Selekcja Ruletkowa - zastosowany bardziej (?) zasobo bierny algorytm.
    ----------
    #### Parametry
    ----------
    population: list, populacja rozwiązań
    elite_percentage: int, procent elitarnych rozwiązań do zachowania
    percentage: int, procent populacji który zachowujemy
    return: Solution, wybrane rozwiązanie

    Schemat rozwiązania:
    Elitarne rozwiązania mają 100% szansę na wybór, a pozostałe rozwiązania 
    są wybierane ruletką w oparciu o ich fitness.
    """
    
    elite_num = int(len(population) * (elite_percentage / 100))

    next_population = []

    population.sort()

    next_population = population[:elite_num]

    remaining_population = population[elite_num:]

    num_select = int(len(remaining_population) * (percentage / 100))

    if not remaining_population or num_select == 0:
        return next_population

    total_fitness = sum(solution.fitness for solution in remaining_population)

    for _ in range(num_select):
        selection_point = random.uniform(0, total_fitness)
        cumulative_fitness = 0

        for index, solution in enumerate(remaining_population):
            cumulative_fitness += solution.fitness
            if cumulative_fitness >= selection_point:
                next_population.append(solution)
                total_fitness -= solution.fitness
                remaining_population.pop(index)
                break

    return next_population


def tournament_selection(population: list, percentage: int = 50, tournament_size: int = 2) -> list:
    """
    #### Selekcja turniejowa
    ----------
    #### Parametry
    ----------
    population: list, populacja rozwiązań
    percentage: int, procent populacji który zachowujemy
    tournament_size: int, liczba rozwiązań biorących udział w turnieju
    return: Solution, wybrane rozwiązanie

    Schemat rozwiązania:
    Losowo wybieramy kilka rozwiązań i wybieramy to z najlepszym fitness.
    """

    num_select = int(len(population) * (percentage/100))

    next_population = []

    for _ in range(num_select):
        tournament_contestants = random.sample(population, tournament_size)
        winner = max(tournament_contestants, key=lambda x: x.fitness)
        next_population.append(winner)
        population.remove(winner)

    return next_population


def elitist_tournament_selection(population: list, percentage: int = 50, elite_percentage: int = 15, tournament_size: int = 2) -> list:
    """
    #### Elitarna Selekcja Turniejowa
    ----------
    #### Parametry
    ----------
    population: list, populacja rozwiązań
    percentage: int, procent populacji który zachowujemy
    elite_percentage: int, procent elitarnych rozwiązań do zachowania
    tournament_size: int, liczba rozwiązań biorących udział w turnieju
    return: Solution, wybrane rozwiązanie

    Schemat rozwiązania:
    Elitarne rozwiązania zawsze biorą udział w selekcji, a pozostałe wybierane są 
    w ramach selekcji turniejowej.
    """
    elite_num = int(len(population) * (elite_percentage/100))

    next_population = []

    population.sort()

    next_population = population[:elite_num]

    remaining_population = population[elite_num:]

    num_select = int(len(remaining_population) * (percentage/100))

    if not remaining_population or num_select == 0:
        return next_population

    for _ in range(num_select):
        tournament_contestants = random.sample(remaining_population, tournament_size)
        winner = max(tournament_contestants, key=lambda x: x.fitness)
        next_population.append(winner)
        remaining_population.remove(winner)
        
    return next_population

"""
------------------------
Inne
------------------------
"""


def naive_legitimacy(child: Solution, parent_1: Solution, parent_2: Solution, max_retries: int = 10) -> Solution:
    """
    #### Sprawdzanie i dostosowywanie poprawności dziecka
    ----------
    #### Parametry
    ----------
    child: Solution, dziecko, które ma zostać zweryfikowane
    parent_1: Solution, rodzic 1, dla porównania
    parent_2: Solution, rodzic 2, dla porównania
    return: Solution, poprawione dziecko, jeśli konieczne

    Schemat rozwiązania:
    Weryfikujemy, czy dziecko spełnia kryteria poprawności. Jeśli jest to konieczne, dostosowujemy jego 
    elementy, aby były zgodne z wymaganiami, np. eliminując duplikaty lub elementy poza dozwolonym obszarem.
    Jeżeli dziecko nie zostanie naprawione w liczbie max_retries przypisz mu dane z rodzica.
    """
    
    if len(child.vector) != len(set(child.vector)):
        unique_child = set(child.vector)
        allowed_values = set(parent_1.vector + parent_2.vector)
        missing_values = allowed_values - unique_child
        
        child.vector = list(unique_child) + random.sample(missing_values, len(child.vector) - len(unique_child))
        child.evaluate_function()

    if child.is_legal():
        return child

    for _ in range(max_retries):
        child._perform_mutation()
        if child.is_legal():
            return child

    better_parent = parent_1 if parent_1.fitness > parent_2.fitness else parent_2
    child = Solution(vector=better_parent.vector.copy(), problem=better_parent.problem)

    return child

