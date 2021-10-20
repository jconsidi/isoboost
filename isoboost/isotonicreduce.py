# isotonicreduce.py

from .smawk import smawk_min


def reduce_isotonic(vs, ws, n_values, p=2):
    if p == 2:
        return reduce_isotonic_l2(vs, ws, n_values)

    raise NotImplentedError("only p=2 implemented")


def reduce_isotonic_l2(vs, ws, n_values):
    """Given the output values and weights from an isotonic regression,
    and a bound on the number of output values, optimally reduce the
    set of values using L2 norms.

    Follows the approach of Optimal Reduced Isotonic Regression by
    Janis Hardwick and Quentin Stout. https://arxiv.org/abs/1412.2844
    (the 2014 arxiv version, not the 2012 proceedings version with a
    slower result.

    """

    value_weights = {}
    for (v, w) in zip(vs, ws):
        value_weights[v] = value_weights.get(v, 0.0) + w

    m = len(value_weights)
    if m <= n_values:
        return {v: v for v in value_weights}

    # get sorted list of distinct values and matching weights
    vws = sorted(value_weights.items())
    (vs, ws) = zip(*vws)

    def calculate_range_mean(i, j):
        return sum(v * w for (v, w) in vws[i:j]) / sum(ws[i:j])

    def calculate_range_error(i, j):
        if j <= i:
            return 0.0

        v_mean = calculate_range_mean(i, j)
        return sum((v - v_mean) ** 2 * w for (v, w) in vws[i:j]) / sum(ws[i:j])

    errs = []
    errs.append([calculate_range_error(i, m) for i in range(m)])

    splits = []

    while len(errs) < n_values:

        def err_i_j(i, j):
            return calculate_range_error(i, j) + errs[-1][j]

        splits.append(smawk_min(err_i_j, m, m))
        errs.append([err_i_j(i, splits[-1][i]) for i in range(m)])

    output = {}
    i = 0
    for b in range(n_values - 1, 0, -1):
        j = splits[b - 1][i]

        v_mean = calculate_range_mean(i, j)
        for v in vs[i:j]:
            output[v] = v_mean

        i = j

    v_mean = calculate_range_mean(i, m)
    for v in vs[i:]:
        output[v] = v_mean

    return output
