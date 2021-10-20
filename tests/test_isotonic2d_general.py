#!/usr/bin/env python3

import unittest

from isoboost import Isotonic2dRegression
from isoboost import regress_isotonic_2d
from isoboost import regress_isotonic_2d_l1
from isoboost import regress_isotonic_2d_l2


class Isotonic2dBase(object):
    """
    Shared test code for 2D isotonic regressions.
    """

    def check(self, training_data, test_data, *, n_values=None):
        # validate isotonicity of test data

        if len(test_data) <= 100:
            for (x0, y0, v0) in test_data:
                for (x1, y1, v1) in test_data:
                    if x0 > x1 or y0 > y1:
                        continue

                    with self.subTest(x0=x0, y0=y0, v0=v0, x1=x1, y1=y1, v1=v1):
                        self.assertLessEqual(v0, v1)

        # train prediction function

        predict_func = self.fit(training_data, n_values=n_values)

        # TODO: check level sets for training data

        # check predictions for test data

        for (x, y, v_expected) in test_data:
            with self.subTest(x=x, y=y):
                v_actual = predict_func(x, y)
                self.assertAlmostEqual(v_actual, v_expected)

    def check_isotonic(self, training_data):
        training_data = tuple(training_data)
        test_data = tuple(r[:3] for r in training_data)
        self.check(training_data, test_data)

    def fit(self, training_data, *, n_values=None):
        """Build model for training data and return function to predict a single point.
        """
        raise NotImplemented()


class Isotonic2dLpBase(Isotonic2dBase):
    """
    Shared test cases where the choice of Lp norm does not matter.
    """

    def test_00_singleton(self):
        expected = 4.0981

        self.check(
            training_data=[(1.0, 1.0, expected)],
            test_data=[
                (x, y, expected) for x in (0.5, 1.0, 1.5) for y in (0.5, 1.0, 1.5)
            ],
        )

    def test_01_singleton_weighted(self):
        expected = 6.1279

        self.check(
            training_data=[(1.0, 1.0, expected)],
            test_data=[
                (x, y, expected) for x in (0.5, 1.0, 1.5) for y in (0.5, 1.0, 1.5)
            ],
        )

    def test_02_isotonic_tiny(self):
        def f(x, y):
            return x + y ** 2

        data = [(x, y, f(x, y)) for (x, y) in [(0.0, 0.0), (0.0, 0.2)]]
        self.check_isotonic(data)

    def test_03_isotonic_small(self):
        self.check_isotonic(
            ((1.0, 1.0, 1.0), (1.0, 2.0, 3.0), (2.0, 1.0, 3.0), (3.0, 3.0, 6.0))
        )

    def test_04_isotonic_medium(self):
        def f(x, y):
            return x + y ** 2

        data_range = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        self.check_isotonic(((x, y, f(x, y)) for x in data_range for y in data_range))

    def test_05_isotonic_unstable(self):
        """This test case triggers numerical error with the initial L1 binary
        split on 0.3 vs 0.30000000000000004. This can inadvertently
        put "0.2 cases" on the wrong side of the split which is then
        handled poorly.

        """

        training_data = [
            (0.0, 0.1, 0.1),
            (0.1, 0.0, 0.1),
            (0.1, 0.1, 0.2),
            (0.2, 0.0, 0.2),
            (0.2, 0.1, 0.30000000000000004),
            (0.3, 0.0, 0.3),
            (0.3, 0.1, 0.4),
            (0.4, 0.0, 0.4),
        ]

        self.check_isotonic(training_data)

    def test_06_isotonic_big(self):
        def f(x, y):
            return x + y

        n = 30
        data_range = [i / n for i in range(n)]
        self.check_isotonic(((x, y, f(x, y)) for x in data_range for y in data_range))


class Isotonic2dL1TestCase(Isotonic2dLpBase, unittest.TestCase):
    """
    Test L1 support from regress_isotonic_2d_l1.
    """

    def fit(self, training_data, *, n_values=None):
        return regress_isotonic_2d_l1(*zip(*training_data))


class Isotonic2dL2TestCase(Isotonic2dLpBase, unittest.TestCase):
    """
    Test L2 support from regress_isotonic_2d_l2.
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

    def fit(self, training_data, *, n_values=None):
        return regress_isotonic_2d_l2(*zip(*training_data), n_values=n_values)

    def test_10_unsorted(self):
        expected = 0.25

        test_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.check(
            training_data=[
                (1.0, 1.0, 1.0),
                (1.0, 2.0, 1.0),
                (2.0, 1.0, 1.0),
                (3.0, 3.0, -2.0),
            ],
            test_data=[(x, y, expected) for x in test_range for y in test_range],
        )

    def test_20_reduced(self):
        low = 2.0 / 3.0
        high = 2.0 + 4.0 / 7.0
        self.check(
            training_data=[
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 1.0),
                (0.0, 2.0, 2.0),
                (0.0, 3.0, 3.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 2.0),
                (1.0, 2.0, 3.0),
                (2.0, 0.0, 2.0),
                (2.0, 1.0, 3.0),
                (3.0, 0.0, 3.0),
            ],
            test_data=[
                (0.0, 0.0, low),
                (0.0, 1.0, low),
                (0.0, 2.0, high),
                (0.0, 3.0, high),
                (1.0, 0.0, low),
                (1.0, 1.0, high),
                (1.0, 2.0, high),
                (2.0, 0.0, high),
                (2.0, 1.0, high),
                (3.0, 3.0, high),
            ],
            n_values=2,
        )


class Isotonic2dTestCase(Isotonic2dL2TestCase):
    """
    Test regress_isotonic_2d with the same L2 test cases.
    """

    def fit(self, training_data, *, n_values=None):
        return regress_isotonic_2d(*zip(*training_data), n_values=n_values)


class Isotonic2dRegressionTestCase(Isotonic2dTestCase):
    """
    Test Isotonic2dRegression.
    """

    def fit(self, training_data, *, n_values=None):
        model = Isotonic2dRegression(n_values=n_values)
        model.fit(X=[r[:2] for r in training_data], y=[r[2] for r in training_data])

        return lambda x, y: model.predict([(x, y)])[0]


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    unittest.main()
