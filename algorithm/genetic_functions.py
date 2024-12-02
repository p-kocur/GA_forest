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
        

    