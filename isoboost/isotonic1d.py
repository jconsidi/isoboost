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

    # sort and merge repeated independent variables.
    # also converting to (x, sum(v*w), sum(w)) representation

    inputs.sort()

    inputs[0] = [inputs[0][0], inputs[0][1] * inputs[0][2], inputs[0][2]]
    distinct_xs = 1
    for i in range(1, len(inputs)):
        if inputs[i][0] > inputs[distinct_xs - 1][0]:
            # new x value
            inputs[distinct_xs] = [
                inputs[i][0],
                inputs[i][1] * inputs[i][2],
                inputs[i][2],
            ]
            distinct_xs += 1
        else:
            # duplicate x value
            inputs[distinct_xs - 1][1] += inputs[i][1] * inputs[i][2]
            inputs[distinct_xs - 1][2] += inputs[i][2]

    inputs[distinct_xs:] = []

    # run Principal Adjacent Violators Algorithm

    bucket_starts = []
    bucket_ends = []
    bucket_sums = []
    bucket_values = []
    bucket_weights = []

    inputs.sort()
    for (x, vw, w) in inputs:
        if w == 0.0:
            continue

        # add new data point as a new bucket
        bucket_starts.append(x)
        bucket_ends.append(x)
        bucket_sums.append(vw)
        bucket_values.append(vw / w)
        bucket_weights.append(w)

        # merge buckets as long as isotonicity is violated.

        while len(bucket_values) > 1 and (
            bucket_ends[-2] >= bucket_starts[-1]
            or bucket_values[-2] > bucket_values[-1]
        ):
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
