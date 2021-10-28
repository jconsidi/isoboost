# isotonic1d.py

import itertools

from .isotonicreduce import reduce_isotonic_l2
from .piecewise import PiecewiseLinear


def regress_isotonic_1d(xs, vs, ws=None, *, n_values=None):
    # xs/vs/ws = iterators of values for respective parameters below.
    # x = independent variable
    # v = dependent variable
    # w = weight. defaults to 1 if ws is None.
    # where regressed estimates must be isotonic in x

    if ws is None:
        ws = itertools.repeat(1.0)

    # consume input iterators and match their values.
    inputs = zip(xs, vs, ws)
    inputs = list(inputs)

    # run Principal Adjacent Violators Algorithm

    bucket_starts = []
    bucket_ends = []
    bucket_sums = []
    bucket_values = []
    bucket_weights = []

    inputs.sort()
    for (x, v, w) in inputs:
        if w == 0.0:
            continue

        # add new data point as a new bucket
        bucket_starts.append(x)
        bucket_ends.append(x)
        bucket_sums.append(v * w)
        bucket_values.append(v)
        bucket_weights.append(w)

        # merge buckets as long as isotonicity is violated.

        while len(bucket_values) > 1 and bucket_values[-2] > bucket_values[-1]:
            bucket_sums[-2] += bucket_sums[-1]
            bucket_weights[-2] += bucket_weights[-1]
            if bucket_weights[-2] > 0:
                bucket_values[-2] = bucket_sums[-2] / bucket_weights[-2]

            bucket_ends[-2] = bucket_ends[-1]

            bucket_starts.pop()
            bucket_ends.pop()
            bucket_sums.pop()
            bucket_values.pop()
            bucket_weights.pop()

    if n_values:
        reduced = reduce_isotonic_l2(bucket_values, bucket_weights, n_values)
        bucket_values = [reduced[v] for v in bucket_values]

    points = []
    for i in range(len(bucket_starts)):
        if i <= 0 or bucket_starts[i] > points[-1][0]:
            points.append((bucket_starts[i], bucket_values[i]))
        if bucket_ends[i] != bucket_starts[i]:
            points.append((bucket_ends[i], bucket_values[i]))

    return PiecewiseLinear(points).interpolate
