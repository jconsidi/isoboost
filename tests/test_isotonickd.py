#!/usr/bin/env python3

import unittest

from isoboost import IsotonicKdRegression

class IsotonicKdRegressionTest(unittest.TestCase):
    """
    Test code for kD isotonic regressions.
    """

    def check(self, training_data, test_data):
        # train model

        regressor = IsotonicKdRegression()
        regressor.fit([r[:-1] for r in training_data], [r[-1] for r in training_data])

        for test_row in test_data:
            test_input = test_row[:-1]
            v_expected = test_row[-1]
            with self.subTest(test_input=test_input):
                v_actual = regressor.predict([test_input])[0]
                self.assertAlmostEqual(v_actual, v_expected)

    def check_isotonic(self, training_data):
        training_data = tuple(training_data)
        self.check(training_data, training_data)

    def test_00_singleton(self):
        expected = 4.0981

        self.check(training_data=[(1.0, expected)],
                   test_data=[(0.0, expected), (1.0, expected), (2.0, expected)])

        self.check(training_data=[(1.0, 1.0, 1.0, 1.0, expected)],
                   test_data = [(0.0, 0.0, 0.0, 0.0, expected),
                                (1.0, 1.0, 1.0, 1.0, expected),
                                (0.0, 2.0, 0.0, 2.0, expected)])

    def test_03_isotonic_small(self):
        self.check_isotonic(((1.0, 1.0, 1.0),
                             (1.0, 2.0, 3.0),
                             (2.0, 1.0, 3.0),
                             (3.0, 3.0, 6.0)))

    def test_04_isotonic_medium(self):
        def f(x, y):
            return x + y ** 2

        data_range = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        self.check_isotonic([(x, y, f(x, y)) for x in data_range for y in data_range])

    def test_05_isotonic_big(self):
        def f(x, y, z):
            return x + y + z # x + y ** 2 + z ** 3

        data_range = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        self.check_isotonic([(x, y, z, f(x, y, z)) for x in data_range for y in data_range for z in data_range])

    def test_06_isotonic_big(self):
        def f(x, y, z):
            return x + y ** 2 + z ** 3

        data_range = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        self.check_isotonic([(x, y, z, f(x, y, z)) for x in data_range for y in data_range for z in data_range])

############################################################
# startup handling #########################################
############################################################

if __name__ == '__main__':
    unittest.main()
