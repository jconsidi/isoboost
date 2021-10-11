#!/usr/bin/env python3

import unittest

from isoboost import isotonic2d

class Isotonic2dBinaryTestCase(unittest.TestCase):
    """
    Test case for 2D L1 binary problems.
    """

    def check(self, training_data, a, b, test_data):
        # validate test answers match the requested answers
        
        for (x0, y0, v0) in test_data:
            self.assertIn(v0, (a, b))

        # validate isotonicity of test data

        for (x0, y0, v0) in test_data:
            for (x1, y1, v1) in test_data:
                if x0 > x1 or y0 > y1:
                    continue

                with self.subTest(x0=x0, y0=y0, v0=v0, x1=x1, y1=y1, v1=v1):
                    self.assertLessEqual(v0, v1)

        # train prediction function

        binary_output = self.fit(training_data, a, b)

        # check predictions for test data

        for (x, y, v_expected) in test_data:
            with self.subTest(x=x, y=y):
                v_actual = binary_output[(x, y)]
                self.assertEqual(v_actual, v_expected)

    def check_isotonic(self, training_data, a, b):
        training_data = tuple(training_data)
        self.check(training_data, a, b, training_data)

    def fit(self, training_data, a, b):
        return isotonic2d._regress_isotonic_2d_l1_binary(training_data, a, b)

    def test_04_isotonic(self):
        """
        Test inputs from a sorted test case.
        """

        training_data = []
        training_data.append((1.0, 1.0, 1.0, 1.0))
        training_data.append((1.0, 2.0, 3.0, 1.0))
        training_data.append((2.0, 1.0, 3.0, 1.0))
        training_data.append((3.0, 3.0, 6.0, 1.0))

        for (a, b) in [(1, 3), (3, 6)]:
            test_data = [(x, y, a if v < a else b) for (x, y, v, w) in training_data]
            self.check(training_data, a, b, test_data)
    
    def test_10_ambiguous(self):
        """Test an input that was broken while implementing L2.

        Tthis particular case should have a unique output value to
        maintain isotonicity, but both output choices have the same
        regression error.

        """

        training_data = []
        training_data.append((1.0, 1.0, 1.0, 0.75))
        training_data.append((1.0, 2.0, 1.0, 0.75))
        training_data.append((2.0, 1.0, 1.0, 0.75))
        training_data.append((3.0, 3.0, 0.0, 2.25))

        a = 0.0
        b = 1.0

        output = self.fit(training_data, a, b)

        output_values = set(output.values())
        self.assertEqual(len(output_values), 1)
        self.assertIn(min(output_values), (a, b))

############################################################
# startup handling #########################################
############################################################

if __name__ == '__main__':
    unittest.main()
