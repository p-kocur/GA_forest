from tkinter import *
from tkinter import ttk

# Wstępne
window_size = (1200,800)    # Wymiary okna ~ w przyblizeniu
left_size = (1000,720)      # Wymiary lewej części, zakładam że tam będziemy pokazywać naszą macierz rozwiązań

# 
root = Tk()
root.resizable(0,0)
root.title("BO2 - Algorytm Genetyczny")

# Główna rama
content = ttk.Frame(root)

# Częśc przeznaczona na macierz rozwiązań
matrix_space = ttk.Frame(content, borderwidth=5, relief="ridge", width=left_size[0], height=left_size[1])
matrix_space.grid(column=0, row=0, columnspan=3, rowspan=3)

# Mutacje
mutation_options = ttk.Frame(content, borderwidth=5, height=150, width=(window_size[0]-left_size[0]))
mutation_options.grid(column=3, row=0, columnspan=2, sticky='we')

# Krzyżowanie
crossover_options = ttk.Frame(content, borderwidth=5, height=150, width=(window_size[0]-left_size[0]))
crossover_options.grid(column=3, row=1, columnspan=2, sticky="we")

# Selekcja
selection_options = ttk.Frame(content, borderwidth=5, height=150, width=(window_size[0]-left_size[0]))
selection_options.grid(column=3, row=2, columnspan=2, sticky='we')



# Mutacje


# Zmienne
basic_mut = BooleanVar(value=False)
territorial_mut = BooleanVar(value=False)
permutation_mutation = BooleanVar(value=False)
max_reward_mutation = BooleanVar(value=False)
expansion_mutation = BooleanVar(value=False)


mut_label = ttk.Label(mutation_options, text="Opcje mutacji")
mut_label.grid(column=0, row=0, columnspan=2)

h_sep_mut = ttk.Separator(mutation_options, orient="horizontal")
h_sep_mut.grid(column=0, row=1, columnspan=2, sticky="we")

basic_mut_checkbox = ttk.Checkbutton(mutation_options, text="Mutacja podstawowa", variable=basic_mut, onvalue=True)
basic_mut_checkbox.grid(column=0, row=2, sticky="w")

territorial_mutation_checkbox = ttk.Checkbutton(mutation_options, text="Mutacja terytorialna", variable=territorial_mut, onvalue=True)
territorial_mutation_checkbox.grid(column=0, row=3, sticky="w")

permutation_mutation_checkbox = ttk.Checkbutton(mutation_options, text="Mutacja permutacyjna", variable=permutation_mutation, onvalue=True)
permutation_mutation_checkbox.grid(column=0, row=4, sticky="w")

max_reward_mutation_checkbox = ttk.Checkbutton(mutation_options, text="Mutacja maksymalnej nagrody", variable=max_reward_mutation, onvalue=True)
max_reward_mutation_checkbox.grid(column=0, row=5, sticky="w")

expansion_mutation_checkbox = ttk.Checkbutton(mutation_options, text="Mutacja ekspansji", variable=expansion_mutation, onvalue=True)
expansion_mutation_checkbox.grid(column=0, row=6, sticky="w")


# Krzyżowanie
single_point_crossover_random = BooleanVar(value=False)
single_point_matrix_crossover = BooleanVar(value=False)
single_point_crossover_vector = BooleanVar(value=False)
multi_point_crossover_vector = BooleanVar(value=False)
uniform_crossover_naive = BooleanVar(value=False)

cross_label = ttk.Label(crossover_options, text="Opcje krzyżowania")
cross_label.grid(column=0, row=0, columnspan=2)

h_sep_cross = ttk.Separator(crossover_options, orient="horizontal")
h_sep_cross.grid(column=0, row=1, columnspan=2, sticky="we")


single_point_crossover_random_checkbox = ttk.Checkbutton(crossover_options, text="Single Point Crossover Random", variable=single_point_crossover_random, onvalue=True)
single_point_crossover_random_checkbox.grid(column=0, row=2, sticky="w")

single_point_matrix_crossover_checkbox = ttk.Checkbutton(crossover_options, text="Single Point Matrix Crossover", variable=single_point_matrix_crossover, onvalue=True)
single_point_matrix_crossover_checkbox.grid(column=0, row=3, sticky="w")

single_point_crossover_vector_checkbox = ttk.Checkbutton(crossover_options, text="Single Point Crossover Vector", variable=single_point_crossover_vector, onvalue=True)
single_point_crossover_vector_checkbox.grid(column=0, row=4, sticky="w")

multi_point_crossover_vector_checkbox = ttk.Checkbutton(crossover_options, text="Multi Point Crossover Vector", variable=multi_point_crossover_vector, onvalue=True)
multi_point_crossover_vector_checkbox.grid(column=0, row=5, sticky="w")

uniform_crossover_naive_checkbox = ttk.Checkbutton(crossover_options, text="Uniform Crossover Naive", variable=uniform_crossover_naive, onvalue=True)
uniform_crossover_naive_checkbox.grid(column=0, row=6, sticky="w")


# Selekcja
roulette_selection = BooleanVar(value=False)
elitist_roulette_selection = BooleanVar(value=False)
tournament_selection = BooleanVar(value=False)
elitist_tournament_selection = BooleanVar(value=False)

sel_label = ttk.Label(selection_options, text="Opcje selekcji")
sel_label.grid(column=0, row=0, columnspan=2)

h_sep_sel = ttk.Separator(selection_options, orient="horizontal")
h_sep_sel.grid(column=0, row=1, columnspan=2, sticky="we")

roulette_selection_checkbox = ttk.Checkbutton(selection_options, text="Selekcja ruletkowa", variable=roulette_selection, onvalue=True)
roulette_selection_checkbox.grid(column=0, row=2, sticky="w")

elitist_roulette_selection_checkbox = ttk.Checkbutton(selection_options, text="Selekcja elitarnych ruletkowa", variable=elitist_roulette_selection, onvalue=True)
elitist_roulette_selection_checkbox.grid(column=0, row=3, sticky="w")

tournament_selection_checkbox = ttk.Checkbutton(selection_options, text="Selekcja turniejowa", variable=tournament_selection, onvalue=True)
tournament_selection_checkbox.grid(column=0, row=4, sticky="w")

elitist_tournament_selection_checkbox = ttk.Checkbutton(selection_options, text="Selekcja elitarnych turniejowa", variable=elitist_tournament_selection, onvalue=True)
elitist_tournament_selection_checkbox.grid(column=0, row=5, sticky="w")


content.grid(column=0, row=0)

root.mainloop()