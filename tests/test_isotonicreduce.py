#!/usr/bin/env python3

import unittest

from isoboost import reduce_isotonic


class IsotonicReduceTestCase(unittest.TestCase):
    def check(self, vs, ws, n_values, output_expected):
        self.assertEqual(len(vs), len(ws))

        self.assertEqual(set(vs), set(output_expected.keys()))
        self.assertLessEqual(len(set(output_expected.values())), n_values)

        output_actual = reduce_isotonic(vs, ws, n_values)
        self.assertIsInstance(output_actual, dict)
        self.assertLessEqual(len(set(output_actual.values())), n_values)
        self.assertEqual(output_actual, output_expected)

    def test_00_nop(self):
        self.check([1.0, 2.0], [1.0, 1.0], 2, {1.0: 1.0, 2.0: 2.0})

    def test_01_single_output(self):
        self.check([1.0, 2.0], [6.0, 2.0], 1, {1.0: 1.25, 2.0: 1.25})

    def test_02_paper_example(self):
        # example from https://arxiv.org/abs/1412.2844
        self.check(
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            2,
            {0.0: 2.0, 2.0: 2.0, 4.0: 2.0, 6.0: 8.0, 8.0: 8.0, 10.0: 8.0},
        )

    def test_03_paper_example(self):
        # example from https://arxiv.org/abs/1412.2844
        self.check(
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            {0.0: 1.0, 2.0: 1.0, 4.0: 5.0, 6.0: 5.0, 8.0: 9.0, 10.0: 9.0},
        )


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    unittest.main()
