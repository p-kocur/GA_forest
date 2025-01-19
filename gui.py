from tkinter import BooleanVar, IntVar, DoubleVar, ttk, Tk, Button, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

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

        self.population_size = IntVar(value=100)
        self.iteration_limit = IntVar(value=500)
        self.problem_size = IntVar(value=40)
        self.n_elements = IntVar(value=20)
        self.elite_percent = IntVar(value=30)
        self.mutation_probability = DoubleVar(value=0.5)
        self.new_problem = BooleanVar(value=True)
        self.use_last_population = BooleanVar(value=False)
        
        self.button1 = None
        self.button2 = None
        self.button3 = None

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
            ("Liczba iteracji: ", self.iteration_limit),
            ("Wielkość problemu: ", self.problem_size),
            ("Liczba elementów rozw.: ", self.n_elements),
            ("Procent elity: ", self.elite_percent),
            ("Prawdopo. mutacji: ", self.mutation_probability)
        ]

        for i, (label, var) in enumerate(labels_and_vars, start=2):
            ttk.Label(start_frame, text=label).grid(column=0, row=i, sticky="w")
            ttk.Entry(start_frame, textvariable=var, width=10).grid(column=1, row=i)
            
        ttk.Checkbutton(start_frame, text="Nowy problem", variable=self.new_problem).grid(column=0, row=8, columnspan=1, sticky="w")
        ttk.Checkbutton(start_frame, text="Użyj ostatniej populacji", variable=self.use_last_population).grid(column=0, row=9, columnspan=1, sticky="w")

    def _create_option_frames(self):
        mutation_options = [
            ("Mutacja podstawowa", DoubleVar(name='basic_mutation')), 
            ("Mutacja terytorialna", DoubleVar(name='territorial_mutation')),
            ("Mutacja permutacyjna", DoubleVar(name='permutation_mutation')),
            ("Mutacja maksymalnej nagrody", DoubleVar(name='max_reward_mutation')),
            ("Mutacja ekspansji", DoubleVar(name='expansion_mutation')),
        ]

        crossover_options = [
            ("Krzyżowanie losowe", DoubleVar(name="naive_crossover")),
            ("Krzyżowanie jednopunktowe", DoubleVar(name='single_point_crossover_vector')),
            ("Krzyżowanie wielopunktowe", DoubleVar(name='multi_point_crossover_vector')),
            ("Krzyżowanie zachłanne", DoubleVar(name='greeedy_crossover'))
        ]

        selection_options = [
            ("Selekcja ruletkowa", BooleanVar(name='roulette_selection')),
            ("Selekcja elit", BooleanVar(name='elitism_selection')),
            ("Selekcja turniejowa", BooleanVar(name='tournament_selection'))
        ]

        self.options = [
            self._create_option_frame_numerical("Rodzaj mutacji / prawdopodobieństwo", mutation_options, row=1),
            self._create_option_frame_numerical("Rodzaj krzyżowania / prawdopodobieństwo", crossover_options, row=2),
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
    
    def _create_option_frame_numerical(self, title, options, row):
        frame = ttk.Frame(self.content, borderwidth=5, height=150, width=self.SIDE_PANEL_WIDTH)
        frame.grid(column=3, row=row, columnspan=2, sticky="nswe")
        ttk.Label(frame, text=title).grid(column=0, row=0, columnspan=2)
        ttk.Separator(frame, orient="horizontal").grid(column=0, row=1, columnspan=2, sticky="we")

        for i, (text, variable) in enumerate(options, start=2):
            ttk.Label(frame, text=text).grid(column=0, row=i, sticky="w")
            ttk.Entry(frame, textvariable=variable, width=10).grid(column=1, row=i)

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

    def add_button(self, command: callable):
        Button(
            self.content, text="Wczytaj", relief="raised", border=5,
            command=command
        ).grid(column=3, row=3, columnspan=2, sticky="swe") 

    def draw_matrix(self, scale="cost"):
        self._update_state_label('Stan: Praca')

        n = self.problem_size.get()
        '''fig, ax = plt.subplots(figsize=(10, 10))

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
        ax.axis('off')'''
        
        data = np.zeros((n, n))
        data[:] = np.nan
        if self.best_result:
            for i, point in enumerate(self.best_result.vector):
                data[point[0], point[1]] = i
                
        data_color = np.zeros((n, n))
        title = ""
        if self.best_result:
            if scale == "cost":
                title = "Cena drewna"
                for i in range(n):
                    for j in range(n):
                        data_color[i, j] = self.best_result.problem.matrix[i][j].reward
            elif scale == "weather":
                title = "Kara pogodowa"
                for i in range(n):
                    for j in range(n):
                        data_color[i, j] = self.best_result.problem.matrix[i][j].weather_affection
            elif scale == "workers":
                title = "Liczba pracowników"
                for i in range(n):
                    for j in range(n):
                        data_color[i, j] = self.best_result.problem.matrix[i][j].workers_required
            
        
        fig, ax = plt.subplots(figsize=(7,7))
        cax = ax.matshow(data_color, cmap="summer")
        
        ax.set_title(f"Wartość funkcji celu: {int(self.best_result.fitness)}", fontsize=12)
        colorbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(title, fontsize=12)

        for i in range(n):
            for j in range(n):
                if self.best_result and (i, j) in self.best_result.vector:
                    ax.text(j, i, f"{int(data[i, j])}", va='center', ha='center', color="black", fontsize=self.LEFT_SIZE[1]//(2.3*n))

        ax.set_xticks([])
        ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=self.matrix_space)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0, width=self.LEFT_SIZE[0], height=self.LEFT_SIZE[1])
        
        self.button1 = Button(
            self.content, text="Pokaż cenę drewna", relief="raised", border=5,
            command=lambda:[plt.close(fig), self.draw_matrix("cost")]
        )
        self.button2 = Button(
            self.content, text="Pokaż karę pogodową", relief="raised", border=5,
            command=lambda:[plt.close(fig), self.draw_matrix("weather")]
        )
        self.button3 = Button(
            self.content, text="Pokaż wymaganą liczbę pracowników", relief="raised", border=5,
            command=lambda:[plt.close(fig), self.draw_matrix("workers")]
        )
        
        self.button1.grid(column=0, row=0, columnspan=1, sticky="nwe")
        self.button2.grid(column=1, row=0, columnspan=1, sticky="nwe")
        self.button3.grid(column=2, row=0, columnspan=1, sticky="nwe")

        if len(self.bests)>0:
            self._update_state_label('Stan: rezultaty dostepne')
        else:
            self._update_state_label('Stan: Brak rezultatów')


            
    def hide_buttons(self):
        if self.button1:
            self.button1.grid_forget()
            self.button2.grid_forget()
            self.button3.grid_forget()

    def _on_start_button(self):
        self._update_state_label('Stan: Praca')

        if self.on_start_callback:
            
            mutations = [item[1].get() for item in self.options[0]]
            crossovers = [item[1].get() for item in self.options[1]]
            selections = [item[1]._name for item in self.options[2] if item[1].get() is True]

        self.on_start_callback({
            "population_size": self.population_size.get(),
            "iteration_limit": self.iteration_limit.get(),
            "problem_size": self.problem_size.get(),
            "mutation_probability": self.mutation_probability.get(),
            "elite_percentage": self.elite_percent.get(),
            "mutations": mutations, 
            "crossovers": crossovers,
            "selection": selections, 
            "new_problem": self.new_problem.get(),
            "use_last_population": self.use_last_population.get(),
            "n_elements": self.n_elements.get()
        })
            
        

    def _on_plot_button(self):
        self.hide_buttons()
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
