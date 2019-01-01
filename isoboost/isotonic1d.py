# isotonic1d.py

import itertools

def _isotonic_helper(x, xs):
    # finds index to use looking up values in isotonic models.

    i_min = 0
    i_max = len(xs) - 1

    while i_min < i_max:
        i_mid = (i_min + i_max + 1) // 2
        if xs[i_mid] <= x:
            i_min = i_mid
        else:
            i_max = i_mid - 1

    return i_min

def regress_isotonic_1d(xs, vs, ws = None):
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
    bucket_sums = []
    bucket_values = []
    bucket_weights = []

    inputs.sort()
    for (x, v, w) in inputs:
        if w == 0.0:
            continue

        # add new data point as a new bucket
        bucket_starts.append(x)
        bucket_sums.append(v * w)
        bucket_values.append(v)
        bucket_weights.append(w)

        # merge buckets as long as isotonicity is violated.

        while len(bucket_values) > 1 and bucket_values[-2] > bucket_values[-1]:
            bucket_sums[-2] += bucket_sums[-1]
            bucket_weights[-2] += bucket_weights[-1]
            if bucket_weights[-2] > 0:
                bucket_values[-2] = bucket_sums[-2] / bucket_weights[-2]

            bucket_starts.pop()
            bucket_sums.pop()
            bucket_values.pop()
            bucket_weights.pop()

    def m(x):
        return bucket_values[_isotonic_helper(x, bucket_starts)]

    return m
