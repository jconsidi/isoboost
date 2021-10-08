#!/usr/bin/env python3

import unittest

from isoboost import isotonic2d

class Isotonic2dBase(object):
    """
    Shared test cases where the choice of Lp norm does not matter.
    """

    def check(self, inputs, output):
        # copy inputs and append regressed values
        regressed = []
        for input in inputs:
            (x, y, v) = input[:3]
            w = input[3] if len(input) > 3 else 1.0 # add weight if not specified
            r = output(x, y) # add regressed value
            regressed.append((x, y, v, w, r))

        for (x0, y0, v0, w0, r0) in regressed:
            for (x1, y1, v1, w1, r1) in regressed:
                with self.subTest(x0 = x0, y0 = y0, x1 = x1, y1 = y1):
                    # check isotonicity
                    if x0 <= x1 and y0 <= y1:
                        self.assertLessEqual(r0, r1)

class Isotonic2dLpBase(Isotonic2dBase):
    def test_singleton(self):
        expected = 4.0981

        inputs = []
        inputs.append((1.0, 1.0, expected))

        (xs, ys, vs) = zip(*inputs)
        output = self.regress(xs, ys, vs)

        self.check(inputs, output)

        self.assertEqual(output(0.5, 0.5), expected)
        self.assertEqual(output(0.5, 1.0), expected)
        self.assertEqual(output(0.5, 1.5), expected)
        self.assertEqual(output(1.0, 0.5), expected)
        self.assertEqual(output(1.0, 1.0), expected)
        self.assertEqual(output(1.0, 1.5), expected)
        self.assertEqual(output(1.5, 0.5), expected)
        self.assertEqual(output(1.5, 1.0), expected)
        self.assertEqual(output(1.5, 1.5), expected)

    def test_singleton_weighted(self):
        expected = 6.1279

        inputs = []
        inputs.append((1.0, 1.0, expected))

        (xs, ys, vs) = zip(*inputs)
        output = self.regress(xs, ys, vs)

        self.check(inputs, output)

        self.assertEqual(output(0.5, 0.5), expected)
        self.assertEqual(output(0.5, 1.0), expected)
        self.assertEqual(output(0.5, 1.5), expected)
        self.assertEqual(output(1.0, 0.5), expected)
        self.assertEqual(output(1.0, 1.0), expected)
        self.assertEqual(output(1.0, 1.5), expected)
        self.assertEqual(output(1.5, 0.5), expected)
        self.assertEqual(output(1.5, 1.0), expected)
        self.assertEqual(output(1.5, 1.5), expected)

    def test_sorted(self):
        inputs = []
        inputs.append((1.0, 1.0, 1.0))
        inputs.append((1.0, 2.0, 3.0))
        inputs.append((2.0, 1.0, 3.0))
        inputs.append((3.0, 3.0, 6.0))

        (xs, ys, vs) = zip(*inputs)

        output = self.regress(xs, ys, vs)

        self.check(inputs, output)

        for (x, y, expected) in inputs:
            with self.subTest(x = x, y = y):
                self.assertEqual(output(x, y), expected)

class Isotonic2dL1BinaryTestCase(Isotonic2dBase, unittest.TestCase):
    def check(self, inputs, a, b, output):
        super(Isotonic2dL1BinaryTestCase, self).check(inputs, lambda x, y: output[(x, y)])

        for r in output.values():
            self.assertIn(r, (a, b), msg = 'output not limited to binary values specified')

    def regress(self, inputs, a, b):
        return isotonic2d._regress_isotonic_2d_l1_binary(inputs, a, b)

    def test_0(self):
        """
        Test an input that was broken while implementing L2.
        """

        inputs = []
        inputs.append((1.0, 1.0, 1.0, 0.75))
        inputs.append((1.0, 2.0, 1.0, 0.75))
        inputs.append((2.0, 1.0, 1.0, 0.75))
        inputs.append((3.0, 3.0, 0.0, 2.25))

        a = 0.0
        b = 1.0

        output = self.regress(inputs, a, b)

        self.check(inputs, a, b, output)

        # this particular case should have a unique output value, but
        # both choices have the same regression error.
        self.assertEqual(len(set(output.values())), 1, msg = 'output value should be unique')

    def test_sorted(self):
        """
        Test inputs from a sorted test case.
        """

        inputs = []
        inputs.append((1.0, 1.0, 1.0, 1.0))
        inputs.append((1.0, 2.0, 3.0, 1.0))
        inputs.append((2.0, 1.0, 3.0, 1.0))
        inputs.append((3.0, 3.0, 6.0, 1.0))

        for (a, b) in [(1, 3), (3, 6)]:
            output = self.regress(inputs, a, b)

            self.check(inputs, a, b, output)

            for (x, y, v, _) in inputs:
                with self.subTest(a = a, b = b, x = x, y = y):
                    r = output[(x, y)]
                    if v <= a:
                        self.assertEqual(r, a)
                    elif v >= b:
                        self.assertEqual(r, b)
    
class Isotonic2dL1TestCase(Isotonic2dLpBase, unittest.TestCase):
    """
    Test L1 support from isotonic2d.regress_isotonic_2d_l1.
    """

    def regress(self, *args, **kwargs):
        return isotonic2d.regress_isotonic_2d_l1(*args, **kwargs)

class Isotonic2dL2TestCase(Isotonic2dLpBase, unittest.TestCase):
    """
    Test L2 support from isotonic2d.regress_isotonic_2d_l2.
    """

    def check(self, inputs, output):
        super(Isotonic2dL2TestCase, self).check(inputs, output)

        # check level sets match average of their members.

        level_sets = {}
        for row in inputs:
            (x0, y0, v0) = row[:3]
            w0 = row[3] if len(row) > 3 else 1.0

            r0 = output(x0, y0)
            level_sets.setdefault(r0, [0.0, 0.0])
            level_sets[r0][0] += v0 * w0 # sum(value * weight)
            level_sets[r0][1] += w0 # sum(weight)

        for r in level_sets.keys():
            with self.subTest(r = r):
                # check level sets match their weighted average.
                self.assertAlmostEqual(r, level_sets[r][0] / level_sets[r][1])

    def regress(self, *args, **kwargs):
        return isotonic2d.regress_isotonic_2d_l2(*args, **kwargs)

    def test_unsorted(self):
        expected = 0.25

        inputs = []
        inputs.append((1.0, 1.0, 1.0))
        inputs.append((1.0, 2.0, 1.0))
        inputs.append((2.0, 1.0, 1.0))
        inputs.append((3.0, 3.0, -2.0))

        (xs, ys, vs) = zip(*inputs)

        output = self.regress(xs, ys, vs)

        self.check(inputs, output)

        test_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        for x in test_range:
            for y in test_range:
                with self.subTest(x = x, y = y):
                    self.assertAlmostEqual(output(x, y), expected, places = 5)

class Isotonic2dTestCase(Isotonic2dL2TestCase):
    """
    Test isotonic2d.regress_isotonic_2d with the same L2 test cases.
    """
    def regress(self, *args, **kwargs):
        return isotonic2d.regress_isotonic_2d(*args, **kwargs)

class Isotonic2dRegressionTestCase(unittest.TestCase):
    """
    Test isotonic2d.Isotonic2dRegression.
    """

    def check(self, X, y, output_expected):
        model = isotonic2d.Isotonic2dRegression()
        model.fit(X, y)

        output = model.predict(X)
        for i in range(len(output_expected)):
            with self.subTest(i = i, X_i = X[i]):
                self.assertAlmostEqual(output[i], output_expected[i])

    def test_00_constant(self):
        self.check([[0, 0]], [0], [0])
        self.check([[0, 0], [1, 1]], [1, 1], [1, 1])

    def test_01_sorted(self):
        self.check([[0, 0], [1, 1]], [3, 5], [3, 5])

    def test_01_unsorted(self):
        self.check([[0, 0], [1, 1]], [2, 0], [1, 1])

############################################################
# startup handling #########################################
############################################################

if __name__ == '__main__':
    unittest.main()
