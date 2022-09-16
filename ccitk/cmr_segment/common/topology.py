import math
from typing import Union, Tuple, List
import numpy as np


class Point:
    def __init__(self, x: Tuple):
        self.x = x

    def __getitem__(self, index: int):
        return self.x[index]

    def __repr__(self):
        return f"Point({self.x})"

    def __hash__(self):
        return hash(self.x)

    def __iter__(self):
        return iter(self.x)

    def dim(self):
        return len(self.x)

    def __eq__(self, other: 'Point'):
        if self.dim() != other.dim():
            raise ValueError("Dimension not the same: cannot compare dim {} to dim {} point.".format(self.dim(), other.dim()))
        if all([xi == yi for xi, yi in zip(self, other)]):
            return True
        return False

    def __add__(self, other: 'Point'):
        if self.dim() != other.dim():
            raise ValueError("Dimension not the same: cannot compare dim {} to dim {} point.".format(self.dim(), other.dim()))
        x = tuple((xi + yi for xi, yi in zip(self, other)))
        return Point(x)

    def __sub__(self, other):
        if self.dim() != other.dim():
            raise ValueError("Dimension not the same: cannot compare dim {} to dim {} point.".format(self.dim(), other.dim()))
        x = tuple((xi - yi for xi, yi in zip(self, other)))
        return Point(x)

    def __mul__(self, other):
        """Element wise"""
        if isinstance(other, Point):
            if self.dim() != other.dim():
                raise ValueError("Dimension not the same: cannot compare dim {} to dim {} point.".format(self.dim(), other.dim()))
            x = tuple((xi * yi for xi, yi in zip(self, other)))
            return Point(x)
        # scalar product
        x = tuple((xi * other for xi in self))
        return Point(x)

    def __truediv__(self, other):
        """Element wise"""
        if isinstance(other, Point):
            if self.dim() != other.dim():
                raise ValueError("Dimension not the same: cannot compare dim {} to dim {} point.".format(self.dim(), other.dim()))
            x = tuple((xi / yi for xi, yi in zip(self, other)))
            return Point(x)
        # scalar product
        x = tuple((xi / other for xi in self))
        return Point(x)

    def inner_product(self, other):
        if self.dim() != other.dim():
            raise ValueError("Dimension not the same: cannot compare dim {} to dim {} point.".format(self.dim(), other.dim()))
        x = [ix * jx for ix, jx in zip(self, other)]
        return sum(x)

    def norm(self):
        return self.inner_product(self)

    def normalise(self):
        return self / self.inner_product(self)


def l2_distance(point1: Point, point2: Point):
    return math.sqrt((point1 - point2).norm())


class DistanceFunction:
    def __call__(self, point1: Point, point2: Point) -> float:
        raise NotImplementedError("Must be implemented by subclasses.")


class L2Distance(DistanceFunction):
    def __call__(self, point1: Point, point2: Point):
        return math.sqrt((point1 - point2).norm())


class L1Distance(DistanceFunction):
    def __call__(self, point1: Point, point2: Point):
        x = [abs(xi - yi) for xi, yi in zip(point1, point2)]
        return sum(x)


class StandardDistance(DistanceFunction):
    def __call__(self, point1: Point, point2: Point):
        x = [abs(xi - yi) for xi, yi in zip(point1, point2)]
        return max(x)


class PointSet:
    def __init__(self, *points: Point):
        self._set = {point for point in points}

    def __iter__(self):
        return iter(self._set)

    def __len__(self):
        return len(self._set)

    def set(self):
        return self._set

    def __contains__(self, item: Union[Point, 'PointSet']):
        if isinstance(item, Point):
            return item in self.set()
        if isinstance(item, PointSet):
            for point in item:
                if point not in self:
                    return False
            return True

    def __and__(self, other: Union['PointSet', 'OpenBall']):
        if isinstance(other, OpenBall):
            return other & self
        new_set = self.set().intersection(other.set())
        return PointSet(*new_set)

    def __or__(self, other: 'PointSet'):
        new_set = self.set().union(other.set())
        return PointSet(*new_set)

    def open_ball(self, center: Point, radius: float, distance: DistanceFunction = None):
        if distance is None:
            distance = StandardDistance()
        new_set = []
        for point in self:
            if distance(point, center) < radius:
                new_set.append(point)
        return PointSet(*new_set)

    def is_empty(self):
        return len(self) == 0


class PointSequence:
    def __init__(self, points: List[Point]):
        self.points = points

    def append(self, point: Point):
        self.points.append(point)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, index: int):
        return self.points[index]

    def __setitem__(self, key, value):
        self.points[key] = value

    def __len__(self):
        return len(self.points)

    @classmethod
    def from_numpy_array(cls, points: np.ndarray):
        """points.shape = (N, D)"""
        points = points.tolist()
        points = [Point(tuple(x)) for x in points]
        return cls(points)

    def to_numpy_array(self):
        points = [point.x for point in self]
        return np.array(points)

    def set(self) -> PointSet:
        return PointSet(*self.points)

    def index(self, point: Point):
        return self.points.index(point)

    def tangent(self, point: Point):
        return self.first_derivative(point).normalise()

    def first_derivative(self, point: Point):
        index = self.index(point)
        if index == 0:
            # tangent = (C[1].y - C[0].y) / (C[1].x - C[0].x)
            next_point = self[index + 1]
            this_point = self[index]
        elif index == len(self) - 1:
            # tangent = (C[-1].y - C[-2].y) / (C[-1].x - C[-2].x)
            next_point = self[index]
            this_point = self[index - 1]
        else:
            # tangent = (C[i+1].y - C[i-1].y) / (C[i+1].x - C[i-1].x)
            next_point = self[index + 1]
            this_point = self[index - 1]
        dx = next_point[0] - this_point[0]
        dy = next_point[1] - this_point[1]
        return Point((dx, dy))

    def second_derivative(self, point: Point):
        index = self.index(point)
        if index == 0:
            # tangent = (C[1].y - C[0].y) / (C[1].x - C[0].x)
            next_point = self[index + 1]
            this_point = self[index]
        elif index == len(self) - 1:
            # tangent = (C[-1].y - C[-2].y) / (C[-1].x - C[-2].x)
            next_point = self[index]
            this_point = self[index - 1]
        else:
            # tangent = (C[i+1].y - C[i-1].y) / (C[i+1].x - C[i-1].x)
            next_point = self[index + 1]
            this_point = self[index - 1]
        next_tangent = self.first_derivative(next_point)
        this_tangent = self.first_derivative(this_point)
        dx = next_tangent[0] - this_tangent[0]
        dy = next_tangent[1] - this_tangent[1]
        return Point((dx, dy))

    def normal(self, point: Point):
        return self.second_derivative(point).normalise()


class Arc(PointSequence):
    """Define discrete arc as C[i] = Point_i. Each point is distinct and an arc has two endpoints"""
    def __init__(self, points: List[Point]):
        if len(set(points)) != len(points):
            new_points = []
            for point in points:
                if point not in new_points:
                    new_points.append(point)
        else:
            new_points = points
        super().__init__(new_points)

    def delete_point(self, point: Point) -> 'Arc':
        """Deleting a point from arc."""
        index = self.index(point)
        new_points = []
        for i in range(0, index):
            new_points.append(self[i])
        for i in range(index + 1, len(self)):
            new_points.append(self[i])
        return Arc(new_points)


class Curve(Arc):
    """Define discrete arc as C[i] = Point_i. Each point is distinct and the end of the sequence connects
    with the start"""
    def __getitem__(self, index: int):
        _, mod = divmod(index, len(self))
        return self.points[mod]

    def __setitem__(self, index: int, value: Point):
        _, mod = divmod(index, len(self))
        self.points[mod] = value

    def neighbours(self, point: Point):
        index = self.index(point)
        return self[index-1], self[index+1]

    def delete_point(self, point: Point) -> Arc:
        """Deleting a point from curve resulting in an arc, which has two endpoints that are the neighbours
        of the deleted point"""
        index = self.index(point)
        new_points = []
        for i in range(index + 1, len(self)):
            new_points.append(self[i])
        for i in range(0, index):
            new_points.append(self[i])
        return Arc(new_points)

    def first_derivative(self, point: Point):
        index = self.index(point)
        next_point = self[index + 1]
        this_point = self[index - 1]
        dx = next_point[0] - this_point[0]
        dy = next_point[1] - this_point[1]
        return Point((dx, dy))

    def second_derivative(self, point: Point):
        index = self.index(point)
        next_point = self[index + 1]
        this_point = self[index - 1]
        next_tangent = self.first_derivative(next_point)
        this_tangent = self.first_derivative(this_point)
        dx = next_tangent[0] - this_tangent[0]
        dy = next_tangent[1] - this_tangent[1]
        return Point((dx, dy))


class OpenBall:
    """Open ball in R^n"""
    def __init__(self, point: Point, radius: float, distance: DistanceFunction = None):
        self.point = point
        self.radius = radius
        if distance is None:
            distance = StandardDistance()
        self.distance = distance

    def __contains__(self, item: Union[Point, PointSet]):
        if isinstance(item, Point):
            distance = self.distance(self.point, item)
            if distance < self.radius:
                return True
            return False
        if isinstance(item, PointSet):
            for point in item:
                if point not in self:
                    return False
            return True

    def __and__(self, point_set: PointSet) -> PointSet:
        new_set = []
        for point in point_set:
            if self.distance(point, self.point) < self.radius:
                new_set.append(point)
        return PointSet(*new_set)


def tangent(this_point: Point, next_point: Point):
    """tangent vector of two points"""
    dx = next_point[0] - this_point[0]
    dy = next_point[1] - this_point[1]
    ds = math.sqrt(dx ** 2 + dy ** 2)
    tangent = [dx / ds, dy / ds]
    return Point(tuple(tangent))


def normal(this_tangent, next_tangent):
    """normal vector of two tangents"""
    dx = next_tangent[0] - this_tangent[0]
    dy = next_tangent[1] - this_tangent[1]
    ds = math.sqrt(dx ** 2 + dy ** 2)

    normal = [dx / ds, dy / ds]
    return Point(tuple(normal))
