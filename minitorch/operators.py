"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x: First number to multiply.
        y: Second number to multiply.

    Returns:
        f(x, y) = x * y
        Product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
        x: Input value.

    Returns:
        f(x) = x
        The input value unchanged.
    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x: First number to add.
        y: Second number to add.

    Returns:
        f(x, y) = x + y
        Sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x: Input value.

    Returns:
        Negated value of x.
        f(x) = -x
    """
    return float(-x)


def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
        x: First value to compare.
        y: Second value to compare.

    Returns:
        1.0 if x < y, else 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x equals y.

    Args:
        x: First value to compare.
        y: Second value to compare.

    Returns:
        1.0 if x == y, else 0.0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Get the maximum of two numbers.

    Args:
        x: First value to compare.
        y: Second value to compare.

    Returns:
        The larger of x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close within tolerance.

    Args:
        x: First value to compare.
        y: Second value to compare.

    Returns:
        1.0 if |x - y| < 1e-2, else 0.0.
    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
        x: Input value.

    Returns:
        Sigmoid of x, computed as 1/(1 + e^-x) for x >= 0 or e^x/(1 + e^x) for x < 0
        for numerical stability.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU function.

    Args:
        x: Input value.

    Returns:
        x if x > 0, else 0.
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute derivative of log times a second argument.

    Args:
        x: Input to log function.
        d: Multiplicative factor.

    Returns:
        d multiplied by derivative of log at x (1/x).
    """
    return d * (1 / (x + EPS))


def inv(x: float) -> float:
    """Compute reciprocal.

    Args:
        x: Input value.

    Returns:
        f(x) = 1/x
        1 divided by x.
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute derivative of reciprocal times a second argument.

    Args:
        x: Input to reciprocal function.
        d: Multiplicative factor.

    Returns:
        d multiplied by derivative of 1/x at x (-1/x²).
    """
    return d * (-1.0 / (x * x))


def relu_back(x: float, d: float) -> float:
    """Compute derivative of ReLU times a second argument.

    Args:
        x: Input to ReLU function.
        d: Multiplicative factor.

    Returns:
        d multiplied by derivative of ReLU at x (1 if x > 0 else 0).
    """
    return d * (1.0 if x > 0 else 0.0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list.

    Args:
        ls: List of numbers to negate.

    Returns:
        New list with each element negated.
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    """Element-wise addition of two lists.

    Args:
        ls1: First list of numbers.
        ls2: Second list of numbers.

    Returns:
        New list with element-wise sums.
    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    def _reduce(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(x, result)
        return result
    return _reduce


def sum(ls: Iterable[float]) -> float:
    """Sum elements of a list.

    Args:
        ls: List of numbers to sum.

    Returns:
        Sum of all elements.
    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Compute product of elements in a list.

    Args:
        ls: List of numbers to multiply.

    Returns:
        Product of all elements.
    """
    return reduce(mul, 1.0)(ls)
