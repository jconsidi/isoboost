# isotonic2d.py

# Based on
# Isotonic Regression via Partitioning
# by Quentin F. Stout
# in Algorithmica 66 (2013), pp. 93–112.

import bisect
import itertools
import math

from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.utils import check_array

from . import rangemap

def _build_output_function(regressed):
    """
    Helper function returning output function applying regression
    output to arbitrary points.
    """

    if len(regressed) <= 0:
        raise ValueError('no regressed values')

    regressed_min = min(regressed.values())

    regressed_by_x = {}
    for ((x, y), r) in regressed.items():
        regressed_by_x.setdefault(x, []).append((y, r))

    for (x, yrs) in regressed_by_x.items():
        yrs.sort()

    # switch to list of (x, [(y,r),...])
    regressed_by_x = sorted(regressed_by_x.items())

    def fit(x, y):
        # TODO make this run in O(log n) time.
        output = regressed_min # TODO make this an option
        for (x0, yrs) in regressed_by_x:
            if x0 > x:
                break

            if yrs[0][0] > y:
                continue

            y_index = bisect.bisect_right(yrs, (y, math.inf)) - 1
            output = max(output, yrs[y_index][1])

        return output

    return fit

def _regress_isotonic_2d_l1_binary(inputs, a, b):
    """
    Helper function used for L1/L2 regression. Will be used to
    partition vertices into those which will have regression values at
    most a or at least b.
    """

    if a >= b:
        raise ValueError('a < b is required')

    # collect and sort distinct x/y values

    x_values = set(x for (x, _, _, _) in inputs)
    y_values = set(y for (_, y, _, _) in inputs)

    x_values = sorted(x_values)
    y_values = sorted(y_values)

    x_indexes = {x : u for (u, x) in enumerate(x_values)}
    y_indexes = {y : v for (v, y) in enumerate(y_values)}

    # calculate the error function following the sketch in
    # 
    # Isotonic Regression via Partitioning, section 4.2

    inputs = sorted(inputs)

    # min_error[k][c] is the minimum error from
    # 
    # any isotonic regression f of the first k points (0 <= i < k)
    # 
    # where f(x_i, y_i) = b implies c_i >= c
    #
    # and for at least one i, f(x_i, y_i) = b and c_i = c

    c_max = len(y_indexes) # used for no b usage case

    # base case: no points regressed => zero error
    min_error = [rangemap.RangeMap(0, c_max, 0.0)]

    # inductive steps: add one point at a time

    for (i, (x_i, y_i, v_i, w_i)) in enumerate(inputs):
        c_i = y_indexes[y_i]

        previous_error = min_error[-1]

        a_error = abs(v_i - a) * w_i # error from picking a for v_i
        b_error = abs(v_i - b) * w_i # error from picking b for v_i

        # three cases:
        #
        # 1) f(x_i, y_i) = b, but not the lowest use of b so it was forced.
        # 2) f(x_i, y_i) = b, is the lowest use of b (could be a tie)
        # 3) f(x_i, y_i) = a, so earliest use of b has to be with higher c_i values.

        case_1_error = previous_error.get_range(0, c_i - 1) + b_error if c_i > 0 else None
        case_2_error = rangemap.RangeMap(c_i, c_i, previous_error.get_min(c_i, c_max) + b_error)
        case_3_error = previous_error.get_range(c_i + 1, c_max) + a_error

        if case_1_error is None:
            combined_error = case_2_error + case_3_error
        else:
            combined_error = case_1_error + case_2_error + case_3_error

        min_error.append(combined_error)

    # reverse error optimization to get regression choices

    regressed = {}

    current_choice = min_error[-1].get_min_x()
    for i in range(len(inputs) - 1, -1, -1):
        (x_i, y_i, _, _) = inputs[i]
        c_i = y_indexes[y_i]

        regressed[(x_i, y_i)] = b if c_i >= current_choice else a

        if current_choice < c_i:
            # case 1 - no change.
            pass
        elif current_choice == c_i:
            # case 2 - need to figure out was optimal before.
            current_choice = min_error[i].get_range(c_i, c_max).get_min_x()
        else:
            # case 3 - no change.
            pass

    return regressed

def regress_isotonic_2d(xs, ys, vs, ws = None, p = 2):
    if p == 1:
        return regress_isotonic_2d_l1(xs = xs, ys = ys, vs = vs, ws = ws)
    if p == 2:
        return regress_isotonic_2d_l2(xs = xs, ys = ys, vs = vs, ws = ws)

    raise ValueError('only L1 and L2 norms supported')

def regress_isotonic_2d_l1(xs, ys, vs, ws = None):
    # xs/ys/vs/ws = iterators of values for respective parameters below.
    # x,y = independent variables
    # v = dependent variable
    # w = weight. defaults to 1 if ws is None.
    # where regressed estimates must be isotonic in x and y

    if ws is None:
        ws = itertools.repeat(1.0, len(xs))

    # consume input iterators and match their values.
    xs = list(xs)
    ys = list(ys)
    vs = list(vs)
    ws = list(ws)

    if len(xs) != len(ys) or len(ys) != len(vs) or len(vs) != len(ws):
        raise ValueError('input lengths do not match')

    # duplicate vertexes not allowed.
    if len(set(zip(xs, ys))) < len(xs):
        raise ValueError('duplicate vertexes not supported for L1 regression')

    # L1 regressions can always return input values. Use binary
    # regression (above) to repeatedly half the choices available for
    # each vertex until each vertex has a unique output.

    regressed = {}
    def partition(partition_inputs, partition_choices):
        partition_inputs = list(partition_inputs)

        if len(partition_inputs) <= 0:
            return
        if len(partition_inputs) == 1:
            ((x, y, v, _),) = partition_inputs
            regressed[(x, y)] = v
            return

        if len(partition_choices) == 0:
            raise RuntimeError('invariant failed')
        if len(partition_choices) == 1:
            (v,) = partition_choices
            for (x, y, _, _) in partition_inputs:
                regressed[(x, y)] = v
            return

        mid_index = (len(partition_choices) + 1) // 2
        partition_a = partition_choices[mid_index - 1]
        partition_b = partition_choices[mid_index]

        binary_values = _regress_isotonic_2d_l1_binary(partition_inputs,
                                                       partition_a,
                                                       partition_b)

        # recursively split based on the binary regression

        # low partition
        partition([r for r in partition_inputs if binary_values[(r[0], r[1])] < partition_b],
                  partition_choices[:mid_index])

        # high partition
        partition([r for r in partition_inputs if binary_values[(r[0], r[1])] >= partition_b],
                  partition_choices[mid_index:])

    partition(list(zip(xs, ys, vs, ws)), sorted(set(vs)))

    return _build_output_function(regressed)

def regress_isotonic_2d_l2(xs, ys, vs, ws = None):
    # xs/ys/vs/ws = iterators of values for respective parameters below.
    # x,y = independent variables
    # v = dependent variable
    # w = weight. defaults to 1 if ws is None.
    # where regressed estimates must be isotonic in x and y

    if ws is None:
        ws = itertools.repeat(1.0, len(xs))

    # consume input iterators and match their values.
    xs = list(xs)
    ys = list(ys)
    vs = list(vs)
    ws = list(ws)

    if len(xs) != len(ys) or len(ys) != len(vs) or len(vs) != len(ws):
        raise ValueError('input lengths do not match')

    # consume input iterators and match their values.
    inputs = zip(xs, ys, vs, ws)
    inputs = list(inputs)

    # combine duplicate vertexes

    values = {}
    weights = {}

    for (x, y, v, w) in inputs:
        k = (x, y)
        values[k] = values.get(k, 0.0) + v * w # sum(v * w)
        weights[k] = weights.get(k, 0.0) + w # sum(w)

    inputs = [(x, y, values[(x, y)], weights[(x,y)]) for (x, y) in values.keys()]

    # Optimal L2 regressions can not be restricted to input values, so
    # we can not use the same binary search over input values used for
    # L1. As noted by Stout, we can split on the L2 norm of the
    # current partition, and get a split unless there is only one
    # regression value remaining. This split choice may require n-1
    # rounds.

    regressed = {}
    def partition(partition_inputs):
        partition_inputs = list(partition_inputs)

        if len(partition_inputs) <= 0:
            raise RuntimeError('empty partition inputs')
        if len(partition_inputs) == 1:
            ((x, y, v, _),) = partition_inputs
            regressed[(x, y)] = v
            return

        def try_binary(v_split):
            # Using epsilon-partitioning from
            # 
            # Isotonic Regression via Partitioning, section 5.1.

            binary_inputs = []
            for (x, y, v, w) in partition_inputs:
                # no pow() in weight formulas since this is L2, so
                # power would be one.
                if v <= v_split:
                    binary_inputs.append((x, y, 0.0, w * (v_split - v)))
                else:
                    binary_inputs.append((x, y, 1.0, w * (v - v_split)))

            return _regress_isotonic_2d_l1_binary(binary_inputs, 0.0, 1.0)

        partition_norm = sum(v * w for (_, _, v, w) in partition_inputs) / sum(w for (_, _, _, w) in partition_inputs)

        binary_values = try_binary(partition_norm)
        if len(set(binary_values.values())) == 1:
            # no more splits
            for (x, y, _, _) in partition_inputs:
                regressed[(x, y)] = partition_norm
            return

        # recursively split based on the binary regression

        # low partition
        partition(r for r in partition_inputs if binary_values[(r[0], r[1])] < 1.0)

        # high partition
        partition(r for r in partition_inputs if binary_values[(r[0], r[1])] >= 1.0)

    partition(list(inputs))

    return _build_output_function(regressed)

class Isotonic2dRegression(RegressorMixin, TransformerMixin):
    """Isotonic 2d regression model.

    Interface based on sklearn.isotonic.IsotonicRegression
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/isotonic.py
    """

    def __init__(self):
        self.f_ = None

    def fit(self, X, y, sample_weight=None):
        # TODO: shape checks
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        self.f_ = regress_isotonic_2d_l2(xs=X[:,0], ys=X[:,1], vs=y, ws = sample_weight)

    def predict(self, T):
        """Predict new data by bilinear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples, 2)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self.transform(T)

    def transform(self, T):
        """Transform new data by bilinear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples, 2)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """

        T = check_array(T)
        return [self.f_(T[i,0], T[i,1]) for i in range(T.shape[0])]
