#!/usr/bin/env python3

import unittest

from isoboost import isotonic2d

class Isotonic2dBase(object):
    """
    Shared test code for 2D isotonic regressions.
    """

    def check(self, training_data, test_data):
        # validate isotonicity of test data

        for (x0, y0, v0) in test_data:
            for (x1, y1, v1) in test_data:
                if x0 > x1 or y0 > y1:
                    continue

                with self.subTest(x0=x0, y0=y0, v0=v0, x1=x1, y1=y1, v1=v1):
                    self.assertLessEqual(v0, v1)

        # train prediction function

        predict_func = self.fit(training_data)

        # TODO: check level sets for training data

        # check predictions for test data

        for (x, y, v_expected) in test_data:
            with self.subTest(x=x, y=y):
                v_actual = predict_func(x, y)
                self.assertAlmostEqual(v_actual, v_expected)

    def check_isotonic(self, training_data):
        training_data = tuple(training_data)
        self.check(training_data, training_data)

    def fit(self, training_data):
        """Build model for training data and return function to predict a single point.
        """
        raise NotImplemented()

class Isotonic2dLpBase(Isotonic2dBase):
    """
    Shared test cases where the choice of Lp norm does not matter.
    """

    def test_00_singleton(self):
        expected = 4.0981

        self.check(training_data=[(1.0, 1.0, expected)],
                   test_data=[(x, y, expected) for x in (0.5, 1.0, 1.5) for y in (0.5, 1.0, 1.5)])

    def test_01_singleton_weighted(self):
        expected = 6.1279

        self.check(training_data=[(1.0, 1.0, expected)],
                   test_data=[(x, y, expected) for x in (0.5, 1.0, 1.5) for y in (0.5, 1.0, 1.5)])

    def test_02_isotonic_small(self):
        self.check_isotonic(((1.0, 1.0, 1.0), (1.0, 2.0, 3.0), (2.0, 1.0, 3.0), (3.0, 3.0, 6.0)))

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

        #self.check(inputs, a, b, output)

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

            #self.check(inputs, a, b, output)

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

    def fit(self, training_data):
        return isotonic2d.regress_isotonic_2d_l1(*zip(*training_data))

class Isotonic2dL2TestCase(Isotonic2dLpBase, unittest.TestCase):
    """
    Test L2 support from isotonic2d.regress_isotonic_2d_l2.
    """

    # def check(self, inputs, output):
    #     super(Isotonic2dL2TestCase, self).check(inputs, output)

    #     # check level sets match average of their members.

    #     level_sets = {}
    #     for row in inputs:
    #         (x0, y0, v0) = row[:3]
    #         w0 = row[3] if len(row) > 3 else 1.0

    #         r0 = output(x0, y0)
    #         level_sets.setdefault(r0, [0.0, 0.0])
    #         level_sets[r0][0] += v0 * w0 # sum(value * weight)
    #         level_sets[r0][1] += w0 # sum(weight)

    #     for r in level_sets.keys():
    #         with self.subTest(r = r):
    #             # check level sets match their weighted average.
    #             self.assertAlmostEqual(r, level_sets[r][0] / level_sets[r][1])

    def fit(self, training_data):
        return isotonic2d.regress_isotonic_2d_l2(*zip(*training_data))

    def test_10_unsorted(self):
        expected = 0.25

        test_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.check(training_data = [(1.0, 1.0, 1.0),
                                    (1.0, 2.0, 1.0),
                                    (2.0, 1.0, 1.0),
                                    (3.0, 3.0, -2.0)],
                   test_data = [(x, y, expected) for x in test_range for y in test_range])

class Isotonic2dTestCase(Isotonic2dL2TestCase):
    """
    Test isotonic2d.regress_isotonic_2d with the same L2 test cases.
    """
    def fit(self, training_data):
        return isotonic2d.regress_isotonic_2d(*zip(*training_data))

class Isotonic2dRegressionTestCase(Isotonic2dTestCase):
    """
    Test isotonic2d.Isotonic2dRegression.
    """

    def fit(self, training_data):
        model = isotonic2d.Isotonic2dRegression()
        model.fit(X=[r[:2] for r in training_data], y = [r[2] for r in training_data])

        return lambda x, y: model.predict([(x, y)])[0]

############################################################
# startup handling #########################################
############################################################

if __name__ == '__main__':
    unittest.main()
