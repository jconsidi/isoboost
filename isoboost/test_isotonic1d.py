#!/usr/bin/env python3

import unittest

import isotonic1d

class Isotonic1dTestCase(unittest.TestCase):
    def test_singleton(self):
        expected = 3.2349

        inputs = []
        inputs.append((1.0, expected))

        (xs, vs) = zip(*inputs)
        output = isotonic1d.regress_isotonic_1d(xs, vs)

        self.assertEqual(output(0.5), expected)
        self.assertEqual(output(1.0), expected)
        self.assertEqual(output(1.5), expected)

    def test_singleton_weighted(self):
        expected = 5.7268

        inputs = []
        inputs.append((1.0, expected, 3.0))

        (xs, vs, ws) = zip(*inputs)
        output = isotonic1d.regress_isotonic_1d(xs, vs, ws)

        self.assertEqual(output(0.5), expected)
        self.assertEqual(output(1.0), expected)
        self.assertEqual(output(1.5), expected)

    def test_sorted(self):
        inputs = []
        inputs.append((1.0, 1.1))
        inputs.append((2.0, 1.2))
        inputs.append((3.0, 1.3))
        inputs.append((4.0, 1.4))

        (xs, vs) = zip(*inputs)

        output = isotonic1d.regress_isotonic_1d(xs, vs)

        self.assertEqual(output(0.5), 1.1)
        self.assertEqual(output(1.0), 1.1)
        self.assertEqual(output(1.5), 1.1)
        self.assertEqual(output(2.0), 1.2)
        self.assertEqual(output(2.5), 1.2)
        self.assertEqual(output(3.0), 1.3)
        self.assertEqual(output(3.5), 1.3)
        self.assertEqual(output(4.0), 1.4)
        self.assertEqual(output(4.5), 1.4)

############################################################
# startup handling #########################################
############################################################

if __name__ == '__main__':
    unittest.main()
