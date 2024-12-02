import unittest
from data_structures.problem_structure import Solution, Problem

class TestStructures(unittest.TestCase):
    def test_is_line_false(self):
        problem = Problem((4, 4), 5, 1, 2, lambda x: 1, 1)
        solution = Solution([(1, 1), (1, 3), (1, 2)], problem)
        self.assertFalse(solution.is_legal())
        
    def test_is_line_true(self):
        problem = Problem((4, 4), 5, 1, 2, lambda x: 1, 1)
        solution = Solution([(1, 0), (1, 2), (1, 3)], problem)
        self.assertTrue(solution.is_legal())
        