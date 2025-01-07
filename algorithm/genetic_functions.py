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
            return i

'''
Zamienia element rozwiązania na element sąsiadujący z nim w macierzy
'''         
def territorial_mutation(sol: 'Solution', index: int = None) -> None:
    # Dopóki nie odnaleźliśmy niezajętego pola
    while True:
        # Losowo wybieramy element rozwiązania
        if index is None:
            i = random.choice(range(sol.problem.n))
        else:
            i = index
        
        # Wybieramy nowe pole macierzy sąsiadujące z poprzednim, pamiętając o ograniczeniach
        new_x = min(sol.problem.size[0]-1, max(0, sol.vector[i][0] + random.choice([-1, 0, 1])))
        new_y = min(sol.problem.size[1]-1, max(0, sol.vector[i][1] + random.choice([-1, 0, 1])))
        
        if (new_x, new_y) not in sol.vector:
            sol.vector[i] = (new_x, new_y)
            return i

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
    return i
  
'''
Odrzuca element rozwiązania o najmniejszej różnicy wartości nagrody z kosztem transportu
'''  
def max_reward_mutation(sol: 'Solution') -> None:
    min_reward = np.inf
    min_element_i = None
    for i in range(len(sol.vector)):
        x, y = sol.vector[i]
        value = sol.problem.matrix[x][y].reward - sol.problem.matrix[x][y].transport_cost
        if value < min_reward:
            min_reward, min_element_i = value, i
    
    return basic_mutation(sol, min_element_i)
    
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
           
    
    
    i = random.choice(range(sol.problem.n))
    sol.vector[i] = (new_x, new_y)
    
    if len(set(sol.vector)) != len(sol.vector):
        basic_mutation(sol, i)
        
    return i


"""
------------------------
Krzyżowanie
------------------------
"""

def naive_crossover(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
    parents = [parent_1, parent_2]
    child_vector = []
    for i in range(parent_1.problem.n):
        child_vector.append(random.choice(parents).vector[i])
    child = Solution(vector=child_vector, problem=parent_1.problem)
    while not child.is_legal() or len(child.vector) != len(set(child.vector)):
        basic_mutation(child)
    return child

'''def single_point_crossover_random(parent_1: 'Solution', parent_2: 'Solution') -> 'Solution':
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

    ValueError("Wrong side argument")'''




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

    size = len(parent_1.vector)
    crossover_point = np.random.randint(1, size)

    if first_is_parent_1:
        child_vector = (
            parent_1.vector[:crossover_point] +
            parent_2.vector[crossover_point:]
        )
    else:
        child_vector = (
            parent_2.vector[:crossover_point] +
            parent_1.vector[crossover_point:]
        )

    child = Solution(vector=child_vector, problem=parent_1.problem)

    while not child.is_legal() or len(child.vector) != len(set(child.vector)):
        basic_mutation(child)

    child.evaluate_function()

    return child



def multi_point_crossover_vector(parent_1: 'Solution', parent_2: 'Solution', n_points=4) -> 'Solution':
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

    while not child.is_legal() or len(child.vector) != len(set(child.vector)):
        basic_mutation(child)

    child.evaluate_function()
    
    return child


"""
------------------------
Selekcja
------------------------
"""
def elitism_selection(population: list, percentage: int = 20) -> list:
    """
    #### Selekcja elitystyczna - wybór najlepszego procentu rozwiązań z populacji
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
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:elite_size] 



def roulette_selection(population: list, percentage: int = 20) -> list:
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

    population.sort(key=lambda x: x.fitness, reverse=True)
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

def tournament_selection(population: list, percentage: int = 20, tournament_size: int = 4) -> list:
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
    
    remaining_population = population.copy()
    
    lost_solutions = []

    while len(remaining_population) > num_select:
        group_a = random.sample(remaining_population, len(remaining_population)//2)
        group_b = list(set(remaining_population) - set(group_a))
        for i in range(len(group_a)):
            if group_a[i].fitness > group_b[i].fitness:
                remaining_population.remove(group_b[i])
                lost_solutions.append(group_b[i])
            else:
                remaining_population.remove(group_a[i])
                lost_solutions.append(group_a[i])
    
    lost_solutions.sort(key=lambda x: x.fitness, reverse=True)
    
    while len(remaining_population) < num_select:
        remaining_population.append(lost_solutions.pop(0))
    
    return remaining_population


"""
------------------------
Inne
------------------------
"""

def mutate_to_legal(solution: Solution, j: int):
    """
    #### Naprawianie nielegalnego rozwiązania po wykonaniu mutacji
    ----
    #### Parametry
    ----
    solution: rozwiązanie, które chcemy naprawić
    j: indeks elementu w wektorze rozwiązania, który został zmutowany
    
    Schemat działania:
    Znajdujemy pierwsze możliwe do wykorzystania pole w macierzy problemu i wstawiamy je do
    rozwiązania w miejscu wcześniej niepoprawnie zmutowanego elementu.
    """
    solution.vector[j] = None
    for x in range(solution.problem.size[0]):
        for y in range(solution.problem.size[1]):
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            valid = True
            for neighbor, i in zip(neighbors, range(len(neighbors))):
                if neighbor in solution.vector:
                    if i == 0:
                        if (x+2, y) in solution.vector or (x-1, y) in solution.vector:
                            valid = False
                            break
                    if i == 1:
                        if (x-2, y) in solution.vector or (x+1, y) in solution.vector:
                            valid = False
                            break
                    if i == 2:
                        if (x, y+2) in solution.vector or (x, y-1) in solution.vector:
                            valid = False
                            break
                    if i == 3:
                        if (x, y-2) in solution.vector or (x, y+1) in solution.vector:
                            valid = False
                            break
            if valid:
                solution.vector[j] = (x, y)
            return
                    
                

