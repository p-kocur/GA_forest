from tkinter import BooleanVar, IntVar, ttk, Tk, Button
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Nie wiem dlaczego zmienne nie mają wpływu na wielkość okna
WINDOW_SIZE = (1500, 800)
LEFT_SIZE = (1000, 720)
SIDE_PANEL_WIDTH = WINDOW_SIZE[0] - LEFT_SIZE[0]

root = Tk()
root.resizable(0, 0)
root.title("BO2 - Algorytm Genetyczny")

content = ttk.Frame(root)
content.grid(column=0, row=0)

# Miejsce na reprezentacje graficzną macierzy
matrix_space = ttk.Frame(content, borderwidth=5, relief="ridge", width=LEFT_SIZE[0], height=LEFT_SIZE[1])
matrix_space.grid(column=0, row=0, columnspan=3, rowspan=4)


def draw_matrix():
    n = problem_size.get()

    rows, cols = n, n

    fig, ax = plt.subplots(figsize=(10,10)) 

    for i in range(rows + 1):
        ax.plot([0, cols], [i, i], color='black', lw=1)

    for i in range(cols + 1):
        ax.plot([i, i], [0, rows], color='black', lw=1)

    for i in range(rows + 1):
        for j in range(cols + 1):
            ax.scatter(j, i, color='black', s=35)

    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5) 
    ax.set_aspect('equal')

    ax.axis('off')
    canvas = FigureCanvasTkAgg(fig, master=matrix_space)
    canvas.draw()

    canvas.get_tk_widget().place(x=0, y=0, width=LEFT_SIZE[0], height=LEFT_SIZE[1])


def create_section(parent, title, options, start_row=0):
    ttk.Label(parent, text=title).grid(column=0, row=start_row, columnspan=2)
    ttk.Separator(parent, orient="horizontal").grid(column=0, row=start_row + 1, columnspan=2, sticky="we")

    for i, (text, variable) in enumerate(options, start=start_row + 2):
        ttk.Checkbutton(parent, text=text, variable=variable).grid(column=0, row=i, sticky="nswe")

def create_option_frame(parent, title, options, row):
    frame = ttk.Frame(parent, borderwidth=5, height=150, width=SIDE_PANEL_WIDTH)
    frame.grid(column=3, row=row, columnspan=2, sticky="nswe")
    create_section(frame, title, options)
    return options

def start(options):
    "PLACEHOLDER"
    print(f"Wielkość populacji: {population_size.get()}")
    print(f"Ilość iteracji: {iteration_limit.get()}")
    for option_group in options:
        for text, variable in option_group:
            print(f"{text}: {'Enabled' if variable.get() else 'Disabled'}")


start_frame = ttk.Frame(content, borderwidth=5, height=150, width=SIDE_PANEL_WIDTH)
start_frame.grid(column=3, row=0, columnspan=2, sticky="we")

ttk.Label(start_frame, text="Zmienne startowe").grid(column=0, row=0, columnspan=2)
ttk.Separator(start_frame, orient="horizontal").grid(column=0, row=1, columnspan=2, sticky="we")

ttk.Label(start_frame, text="Wielkość populacji: ").grid(column=0, row=2, sticky="w")
population_size = IntVar()
ttk.Entry(start_frame, textvariable=population_size, width=10).grid(column=1, row=2)

ttk.Label(start_frame, text="Ilość iteracji: ").grid(column=0, row=3, sticky="w")
iteration_limit = IntVar()
ttk.Entry(start_frame, textvariable=iteration_limit, width=10).grid(column=1, row=3)

ttk.Label(start_frame, text="Wielkość problemu: ").grid(column=0, row=4, sticky="w")
problem_size = IntVar()
ttk.Entry(start_frame, textvariable=problem_size, width=10).grid(column=1, row=4)

ttk.Label(start_frame, text="Prawdopo. mutacji: ").grid(column=0, row=5, sticky="w")
mutation_probability = IntVar()
ttk.Entry(start_frame, textvariable=mutation_probability, width=10).grid(column=1, row=5)

ttk.Label(start_frame, text="Prawdopo. krzyżowania: ").grid(column=0, row=6, sticky="w")
crossover_probability = IntVar()
ttk.Entry(start_frame, textvariable=crossover_probability, width=10).grid(column=1, row=6)

ttk.Label(start_frame, text="Prawdopo. selekcji: ").grid(column=0, row=7, sticky="w")
selection_probability = IntVar()
ttk.Entry(start_frame, textvariable=selection_probability, width=10).grid(column=1, row=7)

Button(content, text="Rysuj macierz", relief="raised", border=5, command=lambda: draw_matrix()).grid(column=0, row=4, columnspan=1, sticky="w")

mutation_options = [
    ("Mutacja podstawowa", BooleanVar()),
    ("Mutacja terytorialna", BooleanVar()),
    ("Mutacja permutacyjna", BooleanVar()),
    ("Mutacja maksymalnej nagrody", BooleanVar()),
    ("Mutacja ekspansji", BooleanVar()),
]

crossover_options = [
    ("Single Point Crossover Random", BooleanVar()),
    ("Single Point Matrix Crossover", BooleanVar()),
    ("Single Point Crossover Vector", BooleanVar()),
    ("Multi Point Crossover Vector", BooleanVar()),
    ("Uniform Crossover Naive", BooleanVar()),
]

selection_options = [
    ("Selekcja ruletkowa", BooleanVar()),
    ("Selekcja elitarnych ruletkowa", BooleanVar()),
    ("Selekcja turniejowa", BooleanVar()),
    ("Selekcja elitarnych turniejowa", BooleanVar()),
]

options = [
    create_option_frame(content, "Opcje mutacji", mutation_options, row=1),
    create_option_frame(content, "Opcje krzyżowania", crossover_options, row=2),
    create_option_frame(content, "Opcje selekcji", selection_options, row=3),
]

Button(content, text="Start", relief="raised", border=5, command=lambda: start(options)).grid(column=3, row=4, columnspan=2, sticky="we")

def on_close():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
