from tkinter import BooleanVar, IntVar, ttk, Tk, Button, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GeneticAlgorithmGUI:
    def __init__(self, root, on_start_callback=None):
        self.root = root
        self.on_start_callback = on_start_callback  # Odwołanie do przycisku start

        self.root.resizable(0, 0)
        self.root.title("BO2 - Algorytm Genetyczny")
        
        self.WINDOW_SIZE = (1500, 800)
        self.LEFT_SIZE = (1000, 720)
        self.SIDE_PANEL_WIDTH = self.WINDOW_SIZE[0] - self.LEFT_SIZE[0]

        self.content = ttk.Frame(self.root)
        self.content.grid(column=0, row=0)

        self.matrix_space = ttk.Frame(
            self.content, borderwidth=5, relief="ridge",
            width=self.LEFT_SIZE[0], height=self.LEFT_SIZE[1]
        )
        self.matrix_space.grid(column=0, row=0, columnspan=3, rowspan=4)

        self.population_size = IntVar(value=5)
        self.iteration_limit = IntVar(value=50)
        self.problem_size = IntVar(value=5)
        self.mutation_probability = IntVar(value=100)
        self.crossover_probability = IntVar(value=100)
        self.selection_probability = IntVar(value=100)

        self.options = []

        
        # Wykres rozwiazania
        self.options = []
        self.best_result = None
        self.bests = []
        self.avgs = []
        self.worsts = []
        self.state_label_text = 'Stan: Brak rezultatów'

        self._initialize_gui()

        self.root.bind("<Escape>", self._on_escape)



    def _initialize_gui(self):
        self._create_start_frame()
        self._create_option_frames()
        self._create_buttons()

    def _create_start_frame(self):
        start_frame = ttk.Frame(
            self.content, borderwidth=5, height=150, width=self.SIDE_PANEL_WIDTH
        )
        start_frame.grid(column=3, row=0, columnspan=2, sticky="we")

        ttk.Label(start_frame, text="Zmienne startowe").grid(column=0, row=0, columnspan=2)
        ttk.Separator(start_frame, orient="horizontal").grid(column=0, row=1, columnspan=2, sticky="we")

        labels_and_vars = [
            ("Wielkość populacji: ", self.population_size),
            ("Ilość iteracji: ", self.iteration_limit),
            ("Wielkość problemu: ", self.problem_size),
            ("Prawdopo. mutacji: ", self.mutation_probability),
        ]

        for i, (label, var) in enumerate(labels_and_vars, start=2):
            ttk.Label(start_frame, text=label).grid(column=0, row=i, sticky="w")
            ttk.Entry(start_frame, textvariable=var, width=10).grid(column=1, row=i)

    def _create_option_frames(self):
        mutation_options = [
            ("Mutacja podstawowa", BooleanVar(name='basic_mutation')),
            ("Mutacja terytorialna", BooleanVar(name='territorial_mutation')),
            ("Mutacja permutacyjna", BooleanVar(name='permutation_mutation')),
            ("Mutacja maksymalnej nagrody", BooleanVar(name='max_reward_mutation')),
            ("Mutacja ekspansji", BooleanVar(name='expansion_mutation')),
        ]

        crossover_options = [
            ("Naive crossover", BooleanVar(name="naive_crossover")),
            #("Single Point Crossover Random", BooleanVar(name='single_point_crossover_random')),
            #("Single Point Matrix Crossover", BooleanVar(name='single_point_matrix_crossover')),
            ("Single Point Crossover Vector", BooleanVar(name='single_point_crossover_vector')),
            ("Multi Point Crossover Vector", BooleanVar(name='multi_point_crossover_vector')),
            ("Uniform Crossover Naive", BooleanVar(name='uniform_crossover_naive')),
        ]

        selection_options = [
            ("Selekcja ruletkowa", BooleanVar(name='roulette_selection')),
            ("Selekcja elitarnych ruletkowa", BooleanVar(name='elitist_roulette_selection')),
            ("Selekcja turniejowa", BooleanVar(name='tournament_selection')),
            ("Selekcja elitarnych turniejowa", BooleanVar(name='elitist_tournament_selection')),
        ]

        self.options = [
            self._create_option_frame("Rodzaj mutacji", mutation_options, row=1),
            self._create_option_frame("Rodzaj krzyżowania", crossover_options, row=2),
            self._create_option_frame("Rodzaj selekcji", selection_options, row=3),
        ]

    def _create_option_frame(self, title, options, row):
        frame = ttk.Frame(self.content, borderwidth=5, height=150, width=self.SIDE_PANEL_WIDTH)
        frame.grid(column=3, row=row, columnspan=2, sticky="nswe")
        ttk.Label(frame, text=title).grid(column=0, row=0, columnspan=2)
        ttk.Separator(frame, orient="horizontal").grid(column=0, row=1, columnspan=2, sticky="we")

        for i, (text, variable) in enumerate(options, start=2):
            ttk.Checkbutton(frame, text=text, variable=variable).grid(column=0, row=i, sticky="nswe")

        return options

    def _create_buttons(self):
        Button(
            self.content, text="Rysuj macierz", relief="raised", border=5,
            command=self.draw_matrix
        ).grid(column=0, row=4, columnspan=1, sticky="we")

        Button(
            self.content, text="Wykres", relief="raised", border=5,
            command=self._on_plot_button
        ).grid(column=1, row=4, columnspan=1, sticky="we")

        self.state_label = ttk.Label(
            self.content, text=self.state_label_text
        )
        self.state_label.grid(column=2, row=4, columnspan=1)

        Button(
            self.content, text="Start", relief="raised", border=5,
            command=self._on_start_button
        ).grid(column=3, row=4, columnspan=2, sticky="we")



    def draw_matrix(self):
        self._update_state_label('Stan: Praca')

        n = self.problem_size.get()
        fig, ax = plt.subplots(figsize=(10, 10))

        for i in range(n):
            ax.plot([0, n-1], [i, i], color='black', lw=1)
        for i in range(n):
            ax.plot([i, i], [0, n-1], color='black', lw=1)
        for i in range(n):
            for j in range(n):
                ax.scatter(j, i, color='black', s=10)

        if self.best_result:
            for point in self.best_result.vector:
                ax.scatter(point[0],point[1], color='red', s=50)    

        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-0.5, n + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=self.matrix_space)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, width=self.LEFT_SIZE[0], height=self.LEFT_SIZE[1])

        if len(self.bests)>0:
            self._update_state_label('Stan: rezultaty dostepne')
        else:
            self._update_state_label('Stan: Brak rezultatów')

    def _on_start_button(self):
        self._update_state_label('Stan: Praca')

        if self.on_start_callback:
            
            mutations = [item[1]._name for item in self.options[0] if item[1].get() is True]
            crossovers = [item[1]._name for item in self.options[1] if item[1].get() is True]
            selections = [item[1]._name for item in self.options[2] if item[1].get() is True]

        self.on_start_callback({
            "population_size": self.population_size.get(),
            "iteration_limit": self.iteration_limit.get(),
            "problem_size": self.problem_size.get(),
            "mutation_probability": self.mutation_probability.get(),
            "crossover_probability": self.crossover_probability.get(),
            "selection_probability": self.selection_probability.get(),
            "mutations": mutations, 
            "crossovers": crossovers,
            "selection": selections, 
        })
            
        

    def _on_plot_button(self):
        if not len(self.bests)>0 or not len(self.avgs)>0 or not len(self.worsts)>0:
            self._update_state_label('Stan: Brak rezultatów')
           
            print("No results to plot.")
            return

        self._update_state_label('Stan: Praca')

        fig, ax = plt.subplots(figsize=(10, 10))

        fig, ax = plt.subplots()
        ax.plot(self.bests, label="Najlepszy")
        ax.plot(self.avgs, label="Średni")
        ax.plot(self.worsts, label="Najgorszy")
        ax.legend()
        ax.grid(True)
        ax.set_title("Przebieg algorytmu")
        ax.set_xlabel("Generacja numer")
        ax.set_ylabel("Fitness")

        canvas = FigureCanvasTkAgg(fig, master=self.matrix_space)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, width=self.LEFT_SIZE[0], height=self.LEFT_SIZE[1])

        self._update_state_label('Stan: rezultaty dostepne')
        

    def _on_escape(self, event=None):
        self.root.quit()
        self.root.destroy()

    def _update_state_label(self, update_text):
        self.state_label.config(text=update_text)
