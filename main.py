import matplotlib.pyplot as plt
from tkinter import Tk, Toplevel
import gui as GUI
from algorithm.algorithm_body import GeneticAlgorithm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np

# Zmienna potrzebna w celu przerwania algorytmu w związku z tym jak działa Tkinter 
# Większość kodu w main jest związana właśnie z wielowątkowością


should_interrupt = threading.Event()
results = []

def main():
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
            algorithm = GeneticAlgorithm(
                crossover_types=starting_parameters['crossovers'],
                selection_types=starting_parameters['selection'],
                mutation_types=starting_parameters['mutations'],
                population_size = starting_parameters['population_size'],
                max_iter = starting_parameters['iteration_limit'],
                mutation_probability = starting_parameters['mutation_probability'],
                elite_percentage = starting_parameters['elite_percentage'],
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
