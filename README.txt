-------READ ME----------
Plik zawierający wstępne opisy działania funkcji, koncepcje które chcemy wykorzystać, sposoby użycia, etc.
Wstępnie pomysły i założenia oznaczam naszymi imionami, żebym wiedział czy mam zastanawiać się o co mi chodziło czy pytać Ciebie.

------------------------
Lista plików:

/algorithm
algorithm_body.py
genetic_functions.py

/data_structures
problem_structure.py

/other
gui.py

------------------------
Wyjaśnienie zawartości:

-------
Plik:   algorithm_body.py 
Opis:   Zawiera główny kod wykonujący algorytm genetyczny
Założenia:
(Piotr) 
- jedna klasa GeneticAlgorithm:
    Dodatkowe objaśnienie parametrów:
    leave_parents: bool = True, jezeli True wszystkie selekcje działają na zasadzie elity 
                                tj. populacja pozostała po selekcji jest przekazywana do kolejnej generacji, 
                                jezeli False kazda kolejna generacja jest nowa tworzona przez kompletne
                                krzyżowanie poprzedniej
UWAGI:
- TODO: 
new_generation - duze fitness powinno miec większą szanse na stworzenie potomków (?)
multi_point_crossover_vector - przekazywanie n_points

(Paweł)

------
Plik: genetic_functions.py
Opis: Zawiera funkcje mutacji, krzyżowania, selekcji i pomocnicze
Założenia:
(Piotr)
    Wszystkie funkcje są opisane w pliku genetic_functions.py.
    List funkcji:

    - Funkcje mutacji:
        basic_mutation, territorial_mutation, permutation_mutation,
        max_reward_mutation, expansion_mutation 
    
    - Funkcje krzyżowania:
        single_point_crossover_random, single_point_matrix_crossover,
        single_point_crossover_vector, multi_point_crossover_vector,
        uniform_crossover_naive
    
    - Funkcje selekcji 
        elitism_selection, roulette_selection, elitist_roulette_selection,
        tournament_selection, elitist_tournament_selection
    
    - Pomocnicze
    naive_legitimacy

(Paweł)

-------
Plik: problem_structure.py
Opis:

-------
Plik: gui.py
Opis: Skrypt odpowiadający za interfejs graficzny 



