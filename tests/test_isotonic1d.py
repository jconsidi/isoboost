#!/usr/bin/env python3

import unittest

from isoboost import regress_isotonic_1d


class Isotonic1dTestCase(unittest.TestCase):
    def check_generic(self, inputs, output):
        # copy inputs and append regressed values
        regressed = []
        for input in inputs:
            (x, v) = input[:2]
            w = input[2] if len(input) > 2 else 1.0  # add weight if not specified
            r = output(x)  # add regressed value
            regressed.append((x, v, w, r))

        regressed.sort()
        for i in range(len(regressed) - 1):
            (x0, v0, w0, r0) = regressed[i]
            (x1, v1, w1, r1) = regressed[i + 1]
            with self.subTest(x0=x0, x1=x1):
                self.assertLessEqual(r0, r1)

        level_sets = {}
        for (x0, v0, w0, r0) in regressed:
            level_sets.setdefault(r0, [0.0, 0.0])
            level_sets[r0][0] += v0 * w0  # sum(value * weight)
            level_sets[r0][1] += w0  # sum(weight)

        for r in level_sets.keys():
            with self.subTest(r=r):
                # check level sets match their weighted average.
                self.assertAlmostEqual(r, level_sets[r][0] / level_sets[r][1])

    def test_00_singleton(self):
        expected = 3.2349

        inputs = []
        inputs.append((1.0, expected))

        (xs, vs) = zip(*inputs)
        output = regress_isotonic_1d(xs, vs)

        self.assertEqual(output(0.5), expected)
        self.assertEqual(output(1.0), expected)
        self.assertEqual(output(1.5), expected)

        self.check_generic(inputs, output)

    def test_01_singleton_weighted(self):
        expected = 5.7268

        inputs = []
        inputs.append((1.0, expected, 3.0))

        (xs, vs, ws) = zip(*inputs)
        output = regress_isotonic_1d(xs, vs, ws)

        self.assertEqual(output(0.5), expected)
        self.assertEqual(output(1.0), expected)
        self.assertEqual(output(1.5), expected)

        self.check_generic(inputs, output)

    def test_02_duplicate(self):
        expected = 1.123

        inputs = []
        inputs.append((1.0, expected))
        inputs.append((1.0, expected))

        output = regress_isotonic_1d(*(zip(*inputs)))

        self.assertEqual(output(0.5), expected)
        self.assertEqual(output(1.0), expected)
        self.assertEqual(output(1.5), expected)

    def test_03_duplicate_avoid_bad_merge(self):
        inputs = []
        inputs.append((0.0, 0.0))
        inputs.append((1.0, -1.0))
        inputs.append((1.0, 3.0))

        output = regress_isotonic_1d(*zip(*inputs))

        self.assertEqual(output(0.0), 0.0)
        self.assertEqual(output(1.0), 1.0)

    def test_10_sorted(self):
        inputs = []
        inputs.append((1.0, 1.1))
        inputs.append((2.0, 1.2))
        inputs.append((3.0, 1.3))
        inputs.append((4.0, 1.4))

        (xs, vs) = zip(*inputs)

        output = regress_isotonic_1d(xs, vs)

        self.assertEqual(output(0.5), 1.1)
        self.assertEqual(output(1.0), 1.1)
        self.assertEqual(output(1.5), 1.15)
        self.assertEqual(output(2.0), 1.2)
        self.assertEqual(output(2.5), 1.25)
        self.assertEqual(output(3.0), 1.3)
        self.assertEqual(output(3.5), 1.35)
        self.assertEqual(output(4.0), 1.4)
        self.assertEqual(output(4.5), 1.4)

        self.check_generic(inputs, output)

    def test_20_reduced_one(self):
        inputs = []
        inputs.append((1.0, 1.1))
        inputs.append((2.0, 1.2))
        inputs.append((3.0, 1.3))
        inputs.append((4.0, 1.4))

        (xs, vs) = zip(*inputs)

        output = regress_isotonic_1d(xs, vs, n_values=1)

        self.assertEqual(output(0.5), 1.25)
        self.assertEqual(output(1.0), 1.25)
        self.assertEqual(output(1.5), 1.25)
        self.assertEqual(output(2.0), 1.25)
        self.assertEqual(output(2.5), 1.25)
        self.assertEqual(output(3.0), 1.25)
        self.assertEqual(output(3.5), 1.25)
        self.assertEqual(output(4.0), 1.25)
        self.assertEqual(output(4.5), 1.25)

        self.check_generic(inputs, output)

    def test_21_reduced_two(self):
        inputs = []
        inputs.append((1.0, 1.1))
        inputs.append((2.0, 1.2))
        inputs.append((3.0, 1.3))
        inputs.append((4.0, 1.4))

        (xs, vs) = zip(*inputs)

        output = regress_isotonic_1d(xs, vs, n_values=2)

        self.assertEqual(output(0.5), 1.15)
        self.assertEqual(output(1.0), 1.15)
        self.assertEqual(output(1.5), 1.15)
        self.assertEqual(output(2.0), 1.15)
        self.assertEqual(output(2.5), 1.25)
        self.assertEqual(output(3.0), 1.35)
        self.assertEqual(output(3.5), 1.35)
        self.assertEqual(output(4.0), 1.35)
        self.assertEqual(output(4.5), 1.35)

        self.check_generic(inputs, output)


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    unittest.main()
