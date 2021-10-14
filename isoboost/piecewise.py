# piecewise.py

from bisect import bisect_right


class PiecewiseLinear:
    def __init__(self, points):
        points = sorted(points)
        for i in range(1, len(points)):
            if points[i][0] == points[i - 1][0]:
                raise ValueError("points must have distinct coordinates")

        (self.ys, self.vs) = zip(*points)

    def interpolate(self, y):
        i = bisect_right(self.ys, y)
        if i == 0:
            return self.vs[0]
        elif i < len(self.ys):
            y0 = self.ys[i - 1]
            y1 = self.ys[i]

            v0 = self.vs[i - 1]
            v1 = self.vs[i]

            return v0 + (v1 - v0) * (y - y0) / (y1 - y0)
        else:
            return self.vs[i - 1]


class PiecewiseBilinear:
    def __init__(self, points):
        points = sorted(points)
        for i in range(1, len(points)):
            if points[i][:2] == points[i - 1][:2]:
                raise ValueError("points must have distinct coordinates")

        self.xs = []
        self.yvs = []

        curr_x = points[0][0]
        curr_ys = [points[0][1]]
        curr_vs = [points[0][2]]
        for (next_x, next_y, next_v) in points[1:]:
            if next_x == curr_x:
                curr_ys.append(next_y)
                curr_vs.append(next_v)
            else:
                self.xs.append(curr_x)
                self.yvs.append(PiecewiseLinear(zip(curr_ys, curr_vs)))

                curr_x = next_x
                curr_ys = [next_y]
                curr_vs = [next_v]

        self.xs.append(curr_x)
        self.yvs.append(PiecewiseLinear(zip(curr_ys, curr_vs)))

    def interpolate(self, x, y):
        i = bisect_right(self.xs, x)
        if i == 0:
            return self.yvs[0].interpolate(y)
        elif i < len(self.xs):
            x0 = self.xs[i - 1]
            x1 = self.xs[i]

            v0 = self.yvs[i - 1].interpolate(y)
            v1 = self.yvs[i].interpolate(y)

            return v0 + (v1 - v0) * (x - x0) / (x1 - x0)
        else:
            return self.yvs[i - 1].interpolate(y)
