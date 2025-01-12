#import matplotlib.pyplot as plt
from tkinter import Tk, Toplevel
import gui as GUI
from algorithm.algorithm_body import GeneticAlgorithm
from algorithm.genetic_functions import mutate_to_legal, basic_mutation, permutation_mutation, territorial_mutation, max_reward_mutation, expansion_mutation
from data_structures.problem_structure import Problem
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
#import pickle
import dill as pickle

# Zmienna potrzebna w celu przerwania algorytmu w związku z tym jak działa Tkinter 
# Większość kodu w main jest związana właśnie z wielowątkowością


should_interrupt = threading.Event()
results = []


def main():
    problem = Problem(mutation_functions=[basic_mutation, permutation_mutation, territorial_mutation, max_reward_mutation, expansion_mutation], mutation_probs=[1, 0, 0, 0, 0], mutate_to_legal=mutate_to_legal, basic_mutation=basic_mutation)
    with open("problem.pkl", 'wb') as f:
        pickle.dump(problem, f)
        
    def start_callback(parameters):
        '''
        Funkcja wywoływana przez przycisk "Start"
        '''
        
        print(f"{parameters}\n")

        # Funkcja obsługująca prace algorytmu 
        def run_algorithm(starting_parameters, result_callback):
            '''
            Funkcja obsługująca prace algorytmu - tworzy klase GeneticAlgorithm i przekazuje rozwiązanie dalej
            Wywołuwana w odzielnym wątku. 
            '''
            with open("problem.pkl", 'rb') as f:
                problem = pickle.load(f)
            if starting_parameters['new_problem']:
                problem = Problem(size=[starting_parameters['problem_size'], starting_parameters['problem_size']], mutation_probs=starting_parameters['mutations'],
                                  mutation_functions=[basic_mutation, permutation_mutation, territorial_mutation, max_reward_mutation, expansion_mutation],
                                  mutate_to_legal=mutate_to_legal, basic_mutation=basic_mutation, n=starting_parameters['n_elements'])
                with open("problem.pkl", 'wb') as f:
                    pickle.dump(problem, f)
            
            algorithm = GeneticAlgorithm(
                crossover_probs=starting_parameters['crossovers'],
                selection_types=starting_parameters['selection'],
                mutation_probs=starting_parameters['mutations'],
                population_size = starting_parameters['population_size'],
                max_iter = starting_parameters['iteration_limit'],
                mutation_probability = starting_parameters['mutation_probability'],
                elite_percentage = starting_parameters['elite_percentage'],
                load_population=starting_parameters['use_last_population'],
                problem=problem,
                interrupt_flag=should_interrupt.is_set,  # Obsluga przerwania poprzez nacisniecie ESC
            )

            # Uruchomienie algorytmu
            best_sol, bests, avgs, worsts = algorithm.run()
            result_callback(best_sol, bests, avgs, worsts)


        def handle_results(best_sol, bests, avgs, worsts):
            """
            Obsługa rozwiązania.
            Sposób przekazywania przez powinien niwelować problemy z kopiowaniem danych między wątkami
            """
            gui._update_state_label('Stan: rezultaty dostepne')
            gui.best_result = best_sol
            gui.bests = bests
            gui.avgs = avgs
            gui.worsts = worsts

            print(best_sol.vector)

        # Uruchomienie algorytmu w nowym wątku
        algorithm_thread = threading.Thread(
            target=run_algorithm, args=(parameters, handle_results)
        )
        algorithm_thread.start()
        
    
    # Obsługa przerwania -
    def interrupt_callback(event=None):
        """
        Funkcja wywoływana przy naciścnieciu ESC ustawiająca flage przerwania dla algorytmu.
        """
        should_interrupt.set()  # Ustawienie flagi przerwania
        print("Interrupt flag set.")

    def interrupt_reset(event=None):
        """
        Funkcja wywoływana przy naciścnieciu R/r usuwająca flage przerwania dla algorytmu.
        """
        should_interrupt.clear()  # Reset flagi przerwabnia
        print("Interrupt flag reset.")

    # Uruchomienie GUI
    root = Tk()
    gui = GUI.GeneticAlgorithmGUI(root, on_start_callback=start_callback)
    
    root.bind("<Escape>", interrupt_callback)
    root.bind("R", interrupt_reset)
    root.bind("r", interrupt_reset)

    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == "__main__":
    main()
