import matplotlib.pyplot as plt

from algorithm.algorithm_body import GeneticAlgorithm
from algorithm.genetic_functions import elitism_selection, elitist_roulette_selection, tournament_selection, elitist_tournament_selection, roulette_selection

def main():
    selection_probabilities = [
            {"selection": tournament_selection, "probability": 1.0}
            ]
    algorithm = GeneticAlgorithm(selection_probabilities=selection_probabilities)
    bests, avgs, worsts = algorithm.run()
    
    plt.figure()
    plt.plot(bests, label="Best")
    plt.plot(avgs, label="Average")
    plt.plot(worsts, label="Worst")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
if __name__ == "__main__":
    main()