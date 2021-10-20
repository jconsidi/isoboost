# smawk.py

"""Implementations of the SMAWK algorithm
https://en.wikipedia.org/wiki/SMAWK_algorithm

Finds the column with the minimum or maximum of each row in a
(logical) matrix with monotonicity constraints.

"""


def smawk_min(f, n, m):
    # TODO: real implementation instead of this quadratic placeholder

    output = [None for i in range(n)]

    def solve(level, columns):
        rows = range(0, n, 1 << level)

        if len(rows) <= 1:
            # base case: linear scan over candidates
            output[rows[0]] = min(columns, key=lambda j: f(rows[0], j))
            return

        columns_inverse = {j: c for (c, j) in enumerate(columns)}
        if len(rows) < len(columns):
            # REDUCE: reduce number of columns to match number of rows
            # and recurse

            candidates = [columns[0]]
            for j in columns[1:]:
                while candidates:
                    i = rows[len(candidates) - 1]
                    if f(i, j) >= f(i, candidates[-1]):
                        break

                    candidates.pop()

                if len(candidates) < len(rows):
                    candidates.append(j)

            return solve(level, candidates)

        # INTERPOLATE: recursively handle the even rows, then
        # interpolate the odd rows.

        solve(level + 1, columns)

        for r in range(1, len(rows) - 1, 2):
            # interpolate internal odd rows
            i = rows[r]
            c_prev = columns_inverse[output[rows[r - 1]]]
            c_next = columns_inverse[output[rows[r + 1]]]

            output[i] = min(columns[c_prev : c_next + 1], key=lambda j: f(i, j))

        if len(rows) % 2 == 0:
            # trailing odd row
            i = rows[-1]
            c_prev = columns_inverse[output[rows[-2]]]
            output[i] = min(columns[c_prev:], key=lambda j: f(i, j))

    solve(level=0, columns=list(range(m)))

    return output
