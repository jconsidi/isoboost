# rangemap.py

import functools

class RangeMap(object):
    """
    This class maps ranges of integers to numeric values.
    """

    def __init__(self, x_min = None, x_max = None, v = None, left = None, right = None):
        """
        Make a new map. If both v and left/right are specified, v is added to all child values.
        """

        if left is not None or right is not None:
            if left is None or right is None:
                raise ValueError('both left and right children must be specified')

            if left.x_max + 1 != right.x_min:
                raise ValueError('left and right children are not adjacent')

            if x_min is not None and x_min != left.x_min:
                raise ValueError('x_min specified does not match left x_min')

            if x_max is not None and x_max != right.x_max:
                raise ValueError('x_max specified does not match right x_max')

            if v is None:
                # default nothing added to child values
                v = 0
        elif v is not None:
            if x_min is None or x_max is None:
                raise ValueError('both x_min and x_max must be specified for leaf values')
        else:
            raise ValueError('neither leaf value nor children specified')

        # leaf value or children
        self.v = v
        self.left = left
        self.right = right

        # aggregate stats
        if left is not None:
            # internal node case
            self.v_min = min(left.v_min, right.v_min) + v

            self.x_min = left.x_min
            self.x_max = right.x_max
        else:
            self.v_min = v

            self.x_min = x_min
            self.x_max = x_max

    def __add__(self, other):
        if other is None:
            raise ValueError('null values not supported')

        if not isinstance(other, RangeMap):
            # assume other is a number
            return RangeMap(x_min = self.x_min, x_max = self.x_max, v = self.v + other, left = self.left, right = self.right)

        # sort the ranges and make sure they line up

        if self.x_min < other.x_min:
            (a, b) = (self, other)
        else:
            (a, b) = (other, self)

        if a.x_max + 1 != b.x_min:
            raise ValueError('ranges are not adjacent')

        return self._balance(a, b)

    def _balance(self, *children):
        # TODO : balance this new tree
        return functools.reduce(lambda a, b: RangeMap(left = a, right = b), children)

    def _check_range(self, x_min, x_max):
        """
        Confirm that we have

        self.x_min <= x_min <= x_max <= self.x_max
        """

        if x_min > x_max:
            raise ValueError('requested range [%d, %d] is degenerate' % (x_min, x_max))

        if x_min < self.x_min or self.x_max < x_max:
            raise ValueError('requested range [%d, %d] does not fit in current range [%d, %d]' % (x_min, x_max, self.x_min, self.x_max))

    def get_min(self, x_min, x_max):
        self._check_range(x_min, x_max)

        if self.left is None:
            return self.v_min

        # internal node case

        if x_max <= self.left.x_max:
            # entirely contained within left child
            return self.left.get_min(x_min, x_max) + self.v

        if self.right.x_min <= x_min:
            # entirely contained within right child
            return self.right.get_min(x_min, x_max) + self.v

        return min(self.left.get_min(x_min, self.left.x_max), self.right.get_min(self.right.x_min, x_max)) + self.v

    def get_min_x(self):
        if self.left is None:
            return self.x_min

        if self.left.v_min + self.v == self.v_min:
            return self.left.get_min_x()
        else:
            return self.right.get_min_x()

    def get_range(self, x_min, x_max):
        self._check_range(x_min, x_max)

        if self.x_min == x_min and x_max == self.x_max:
            # identity case
            return self

        if self.left is None:
            # leaf case
            return RangeMap(x_min = x_min, x_max = x_max, v = self.v)

        # internal node case

        if x_max <= self.left.x_max:
            # entirely contained within left child
            return self.left.get_range(x_min, x_max) + self.v

        if self.right.x_min <= x_min:
            # entirely contained within right child
            return self.right.get_range(x_min, x_max) + self.v

        return self._balance(self.left.get_range(x_min, self.left.x_max), self.right.get_range(self.right.x_min, x_max)) + self.v
