import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random 2D points.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        List of N tuples, each containing two random floats between 0 and 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Represents a dataset of 2D points and their classifications."""

    N: int  # Number of points
    X: List[Tuple[float, float]]  # List of 2D points
    y: List[int]  # List of classifications (0 or 1)


def simple(N: int) -> Graph:
    """Generate a simple dataset where points are classified based on x1 < 0.5.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing N points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a diagonal dataset where points are classified based on x1 + x2 < 0.5.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing N points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a split dataset where points are classified based on x1 < 0.2 or x1 > 0.8.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing N points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate an XOR dataset where points are classified based on x1 < 0.5 and x2 > 0.5 or x1 > 0.5 and x2 < 0.5.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing N points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a circle dataset where points are classified based on x1^2 + x2^2 > 0.1.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing N points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a spiral dataset where points are classified based on the spiral pattern.

    Args:
    ----
        N: Number of points to generate.

    Returns:
    -------
        A Graph object containing N points and their classifications.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
