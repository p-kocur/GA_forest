import unittest
from data_structures.problem_structure import Solution, Problem
from algorithm.genetic_functions import expansion_mutation

class TestGeneticFuns(unittest.TestCase):
    def test_expansion_mutation(self):
        problem = Problem((4, 4), 3, 1, 2, lambda x: 1, 1)
        solution = Solution([(1, 1), (1, 3), (3, 1)], problem, 1)
        expansion_mutation(solution)
        self.assertTrue((2, 2) in solution.vector)
        